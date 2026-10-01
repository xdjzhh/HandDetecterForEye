import time
import numpy as np

"""
离摄像头距离变化 → 点击/拖拽 状态检测器（Push-Click Detector）
================================================================

原理（单目「伪深度」代理，无需深度相机）：
  MediaPipe 只给 2D 关键点，但同一只手的真实掌宽是固定的 ~9cm。
  所以手在画面中的「像素掌宽」与「到摄像头的真实距离」成严格反比：
    距离越近 → 掌宽像素越大  (palm_px ↑)
    距离越远 → 掌宽像素越小  (palm_px ↓)

  因此只用「掌宽 index_mcp(5) → pinky_mcp(17) 的像素距离」作为深度代理，
  结合 EMA 平滑 + 迟滞双阈值 + 基线自适应，就能稳定识别「向前戳一下 = 点击」。

状态机（迟滞阈值，避免抖动误触）：
  IDLE ───(palm_px > BASE * 1.35)────────▶ PUSH_DOWN (触发 mouseDown)
    ▲                                             │
    │ (palm_px < BASE * 1.15)                     │ palm_px 仍 > BASE*1.15
    │                                             ▼
 RELEASE ◀──────────────────────────────────── HOLD (保持按住=拖拽)
  (mouseUp)                                     (持续移动光标=拖拽跟随)

附加：
  1) 基线 BASE 仅在 IDLE 状态下以 α=0.03 缓慢漂移（适应手的基础距离变化），
     在 PUSH_DOWN/HOLD 期间完全冻结，保证退出阈值相对稳定；
  2) EMA(α=0.3) 先对掌宽像素做滤波，抑制单帧尖峰；
  3) COOLDOWN_SEC = 0.6：刚退出 PUSH/HOLD 后的冷却墙钟时长，冷却中不响应深度事件（原 18 帧 @30fps，2026-10-01 统一墙钟）；
  4) 只有主控手（右手 = MediaPipe "Left"）参与深度点击判定，左手忽略。
"""

# 阈值设计（迟滞：ENTER > EXIT，保证状态退出滞后于进入）
ENTER_RATIO = 1.35   # 进入按下：当前掌宽 > 基线 × 1.35  (向前戳 35% 尺度增幅 = 约 26% 距离减小)
EXIT_RATIO  = 1.15   # 释放回到基线范围内：当前掌宽 < 基线 × 1.15

# 平滑
PALM_SMOOTH_ALPHA = 0.30      # 掌宽 EMA 滤波系数（越大越跟手，越小越稳）
BASELINE_ALPHA_IDLE = 0.03    # IDLE 态下基线缓慢更新速率（越小越慢）

# 防抖/冷却（墙钟秒，物理时长与帧率无关）
MIN_HOLD_SEC = 0.067         # ARMED 后必须持续多久满足 ENTER 才真正触发（原 2 帧 @30fps，抑制瞬间尖峰）
COOLDOWN_SEC  = 0.6           # 触发/释放后的冷却时长，冷却中不响应深度事件

class _S:
    """状态枚举"""
    IDLE      = "IDLE"       # 空闲，基线自适应进行中
    ARMED      = "ARMED"      # 已达到 ENTER 阈值，正在等待 MIN_HOLD_SEC
    PUSH_DOWN = "PUSH_DOWN"  # 刚触发按下（mouseDown 触发的这一帧）
    HOLD      = "HOLD"       # 保持按住（拖拽中）
    COOLDOWN  = "COOLDOWN"   # 刚释放的冷却期（墙钟秒）

# 对外发射的事件（供 MouseController / main.py 使用）
class E:
    NONE        = 0
    DEPRESS     = 1   # 按下（仅触发 1 帧）→ 执行 pyautogui.mouseDown
    RELEASE     = 2   # 释放（仅触发 1 帧）→ 执行 pyautogui.mouseUp
    DRAG_MOVE   = 3   # 保持按住移动（每帧都发）→ pyautogui.moveTo


class DepthClickDetector:
    def __init__(self):
        # 状态
        self.state = _S.IDLE
        self.event_this_frame = E.NONE

        # 滤波/基线
        self._smoothed_palm = None   # EMA 后的掌宽（像素）
        self._baseline = None        # 当前基线掌宽（像素）
        self._armed_sec = 0.0       # ARMED 态累计秒
        self._cooldown_until = 0.0  # 冷却截止墙钟（monotonic）

        # 统计（画面展示用）
        self.depress_count = 0       # 累计按下次数（=点击/拖拽次数）
        self.last_press_time = None  # 上次按下时间
        self.last_release_time = None

    # ---------------------- 外部主接口 ----------------------
    def update(self, palm_pixel_width: float | None, dt: float = 0.0, now: float = -1.0) -> int:
        """每帧调用一次，返回本帧事件 E.{NONE,DEPRESS,RELEASE,DRAG_MOVE}。

        Args:
            palm_pixel_width: 主控手 掌宽(像素)，无手时传 None
            dt: 距上一帧墙钟秒数（防误触用）
            now: 当前墙钟时间（time.monotonic），<0 表示函数内自取
        Returns:
            int: 本帧事件
        """
        self.event_this_frame = E.NONE
        if now < 0.0:
            now = time.monotonic()

        # ---- 冷却状态优先处理（墙钟秒） ----
        if self.state == _S.COOLDOWN:
            if now >= self._cooldown_until:
                self.state = _S.IDLE
                self._armed_sec = 0.0
            # 冷却中忽略任何输入
            self._update_baseline(palm_pixel_width, can_update=False)  # 仅平滑，但不更新基线
            return E.NONE

        # ---- 无手：重置到 IDLE，基线不更新 ----
        if palm_pixel_width is None:
            if self.state in (_S.ARMED, _S.PUSH_DOWN, _S.HOLD):
                # 如果之前是按下态，手突然丢了 → 自动释放
                self.event_this_frame = E.RELEASE
                self.last_release_time = time.time()
                self._enter_cooldown(now)
            else:
                self.state = _S.IDLE
                self._armed_sec = 0.0
                self._smoothed_palm = None
            return self.event_this_frame

        # ---- 有手：EMA 平滑掌宽 ----
        if self._smoothed_palm is None:
            self._smoothed_palm = float(palm_pixel_width)
        else:
            self._smoothed_palm = (
                PALM_SMOOTH_ALPHA * float(palm_pixel_width)
                + (1.0 - PALM_SMOOTH_ALPHA) * self._smoothed_palm
            )

        # ---- 基线初始化（首次有手）----
        if self._baseline is None:
            self._baseline = self._smoothed_palm
            self.state = _S.IDLE

        ratio = self._smoothed_palm / max(self._baseline, 1e-6)

        # ================= 状态流转 =================
        if self.state == _S.IDLE:
            self._update_baseline(self._smoothed_palm, can_update=True)
            if ratio >= ENTER_RATIO:
                self.state = _S.ARMED
                self._armed_sec = 0.0

        elif self.state == _S.ARMED:
            # 冻结基线（进入后不再自适应，保证判定稳定）
            if ratio >= ENTER_RATIO:
                self._armed_sec += float(dt)
                if self._armed_sec >= MIN_HOLD_SEC:
                    self.state = _S.PUSH_DOWN
                    self.event_this_frame = E.DEPRESS
                    self.depress_count += 1
                    self.last_press_time = time.time()
            else:
                # 还没保持够就退了 → 回IDLE
                self.state = _S.IDLE
                self._armed_sec = 0.0

        elif self.state == _S.PUSH_DOWN:
            # PUSH_DOWN 是单帧触发态，之后立刻进入 HOLD
            if ratio > EXIT_RATIO:
                self.state = _S.HOLD
                self.event_this_frame = E.DRAG_MOVE
            else:
                # 极短按下立刻松开（< MIN_HOLD_SEC）→ 也算一次完整 click = depress + release
                self.event_this_frame = E.RELEASE
                self.last_release_time = time.time()
                self._enter_cooldown(now)

        elif self.state == _S.HOLD:
            if ratio <= EXIT_RATIO:
                self.event_this_frame = E.RELEASE
                self.last_release_time = time.time()
                self._enter_cooldown(now)
            else:
                self.event_this_frame = E.DRAG_MOVE

        return self.event_this_frame

    # ---------------------- 内部辅助 ----------------------
    def _update_baseline(self, smoothed, *, can_update: bool):
        if smoothed is None or self._baseline is None:
            return
        if not can_update:
            return
        self._baseline = (
            BASELINE_ALPHA_IDLE * smoothed
            + (1.0 - BASELINE_ALPHA_IDLE) * self._baseline
        )

    def _enter_cooldown(self, now: float):
        self.state = _S.COOLDOWN
        self._cooldown_until = float(now) + COOLDOWN_SEC
        self._armed_sec = 0.0

    # ---------------------- 状态查询（画面展示）----------------------
    @property
    def ratio_now(self) -> float:
        """当前掌宽 / 基线 比值（>1 = 比基线近）。"""
        if self._baseline is None or self._smoothed_palm is None:
            return 1.0
        return self._smoothed_palm / max(self._baseline, 1e-6)

    @property
    def cooldown_left_sec(self) -> float:
        """冷却剩余秒数（COOLDOWN 态以外返回 0）"""
        if self.state != _S.COOLDOWN:
            return 0.0
        return max(0.0, self._cooldown_until - time.monotonic())

    def metrics(self) -> dict:
        return {
            "state": self.state,
            "event": self.event_this_frame,
            "smoothed_palm_px": round(self._smoothed_palm or 0.0, 1),
            "baseline_px":     round(self._baseline     or 0.0, 1),
            "ratio":           round(self.ratio_now, 3),
            "cooldown_left_sec": round(self.cooldown_left_sec, 2),
            "depress_count":   self.depress_count,
        }
