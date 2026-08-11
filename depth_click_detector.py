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
  3) COOLDOWN_FRAMES = 15：刚退出 PUSH/HOLD 后有冷却，避免释放回弹造成连击/抖动；
  4) 只有主控手（右手 = MediaPipe "Left"）参与深度点击判定，左手忽略。
"""

# 阈值设计（迟滞：ENTER > EXIT，保证状态退出滞后于进入）
ENTER_RATIO = 1.35   # 进入按下：当前掌宽 > 基线 × 1.35  (向前戳 35% 尺度增幅 = 约 26% 距离减小)
EXIT_RATIO  = 1.15   # 释放回到基线范围内：当前掌宽 < 基线 × 1.15

# 平滑
PALM_SMOOTH_ALPHA = 0.30      # 掌宽 EMA 滤波系数（越大越跟手，越小越稳）
BASELINE_ALPHA_IDLE = 0.03    # IDLE 态下基线缓慢更新速率（越小越慢）

# 防抖/冷却
MIN_HOLD_FRAMES = 2           # 必须连续 N 帧满足 ENTER 才真正触发（抑制瞬间尖峰）
COOLDOWN_FRAMES = 18          # 触发/释放后的冷却帧，冷却中不响应深度事件

class _S:
    """状态枚举"""
    IDLE      = "IDLE"       # 空闲，基线自适应进行中
    ARMED     = "ARMED"      # 已达到 ENTER 阈值，正在等待 MIN_HOLD_FRAMES
    PUSH_DOWN = "PUSH_DOWN"  # 刚触发按下（mouseDown 触发的这一帧）
    HOLD      = "HOLD"       # 保持按住（拖拽中）
    COOLDOWN  = "COOLDOWN"   # 刚释放的冷却期

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
        self._armed_frames = 0       # ARMED 态计帧
        self._cooldown_frames = 0    # 冷却计帧

        # 统计（画面展示用）
        self.depress_count = 0       # 累计按下次数（=点击/拖拽次数）
        self.last_press_time = None  # 上次按下时间
        self.last_release_time = None

    # ---------------------- 外部主接口 ----------------------
    def update(self, palm_pixel_width: float | None) -> int:
        """每帧调用一次，返回本帧事件 E.{NONE,DEPRESS,RELEASE,DRAG_MOVE}。

        Args:
            palm_pixel_width: 主控手 掌宽(像素)，无手时传 None
        Returns:
            int: 本帧事件
        """
        self.event_this_frame = E.NONE

        # ---- 冷却状态优先处理 ----
        if self.state == _S.COOLDOWN:
            self._cooldown_frames -= 1
            if self._cooldown_frames <= 0:
                self.state = _S.IDLE
                self._armed_frames = 0
            # 冷却中忽略任何输入
            self._update_baseline(palm_pixel_width, can_update=False)  # 仅平滑，但不更新基线
            return E.NONE

        # ---- 无手：重置到 IDLE，基线不更新 ----
        if palm_pixel_width is None:
            if self.state in (_S.ARMED, _S.PUSH_DOWN, _S.HOLD):
                # 如果之前是按下态，手突然丢了 → 自动释放
                self.event_this_frame = E.RELEASE
                self.last_release_time = time.time()
                self._enter_cooldown()
            else:
                self.state = _S.IDLE
                self._armed_frames = 0
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
                self._armed_frames = 1

        elif self.state == _S.ARMED:
            # 冻结基线（进入后不再自适应，保证判定稳定）
            if ratio >= ENTER_RATIO:
                self._armed_frames += 1
                if self._armed_frames >= MIN_HOLD_FRAMES:
                    self.state = _S.PUSH_DOWN
                    self.event_this_frame = E.DEPRESS
                    self.depress_count += 1
                    self.last_press_time = time.time()
            else:
                # 还没积累够就退了 → 回IDLE
                self.state = _S.IDLE
                self._armed_frames = 0

        elif self.state == _S.PUSH_DOWN:
            # PUSH_DOWN 是单帧触发态，之后立刻进入 HOLD
            if ratio > EXIT_RATIO:
                self.state = _S.HOLD
                self.event_this_frame = E.DRAG_MOVE
            else:
                # 极短按下立刻松开（< 2帧）→ 也算一次完整 click = depress + release
                self.event_this_frame = E.RELEASE
                self.last_release_time = time.time()
                self._enter_cooldown()

        elif self.state == _S.HOLD:
            if ratio <= EXIT_RATIO:
                self.event_this_frame = E.RELEASE
                self.last_release_time = time.time()
                self._enter_cooldown()
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

    def _enter_cooldown(self):
        self.state = _S.COOLDOWN
        self._cooldown_frames = COOLDOWN_FRAMES
        self._armed_frames = 0

    # ---------------------- 状态查询（画面展示）----------------------
    @property
    def ratio_now(self) -> float:
        """当前掌宽 / 基线 比值（>1 = 比基线近）。"""
        if self._baseline is None or self._smoothed_palm is None:
            return 1.0
        return self._smoothed_palm / max(self._baseline, 1e-6)

    def metrics(self) -> dict:
        return {
            "state": self.state,
            "event": self.event_this_frame,
            "smoothed_palm_px": round(self._smoothed_palm or 0.0, 1),
            "baseline_px":     round(self._baseline     or 0.0, 1),
            "ratio":           round(self.ratio_now, 3),
            "cooldown_left":   max(0, self._cooldown_frames),
            "depress_count":   self.depress_count,
        }
