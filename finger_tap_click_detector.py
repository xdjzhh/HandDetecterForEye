import time
import numpy as np

"""
食指 "Tap-Click" 点击检测器（完全不依赖手掌整体运动！）
========================================================

用户真实点击动作（对应用户口述规则：「点击只是手指的运动，手掌不动」）：
  1) 食指自然伸直 = 待机/移动光标
  2) 食指指尖「勾一下」弯曲（第一/第二关节弯，掌纹/手掌不动）
  3) 食指立刻「弹回」伸直
  → 这 3 步在 300~800ms 内完成 = 1 次单击 tap

  若第 2 步弯曲后「长期保持 > ~500ms 不弹回」= 按住左键 = 支持拖拽
  保持移动 + 之后弹回伸直 = 释放

观测量（相对尺度归一化，对手大小/远近/手掌旋转完全鲁棒，来自经验634118）：
  palm  = dist(INDEX_MCP_5, PINKY_MCP_17)               归一化基准
  bend  = dist(INDEX_TIP_8, INDEX_MCP_5) / palm          食指「伸直度」
    食指完全伸直 → bend ≈ 0.75~0.85
    食指第一关节勾回  → bend ≈ 0.40~0.55
    完全握拳食指    → bend ≈ 0.20~0.35

状态机（严格时序：伸直→弯→伸直 = 一次Tap；弯曲超时=Hold拖拽）：

 STRAIGHT_IDLE (bend≈伸直, 持续学习伸直基线)
   │ (bend < baseline * 0.78   且   相对下降幅度 > 0.12)
   ▼
 BEND_ARMED (检测到弯曲，等 1 帧避免瞬间尖峰)
   │ 下一帧仍满足弯曲
   ▼
 BEND_HOLD  (弯曲持续中 → 计时)
   ├─ (bend 回到 > baseline*0.86  且 hold_frames <= LONG_PRESS)  → TAP_CLICK → 单击
   └─ (hold_frames > LONG_PRESS  且 bend 仍<baseline*0.80)       → LONG_PRESS → 长按拖拽开始
                │
                ▼
          PRESSED_HOLD (保持按住，拖拽跟随)
                │  (bend 回到 > baseline*0.86 → 释放)
                ▼
          COOLDOWN (14帧冷却 + 重置伸直基线慢速学习) → IDLE
"""

# 阈值（文件级默认值；实际运行时可通过构造参数 hardness/speed 被 CLI 参数覆盖）
ENTER_BEND_RATIO_DEFAULT = 0.78       # 进入弯曲判定：bend < baseline * ENTER_BEND_RATIO
EXIT_BEND_RATIO_DEFAULT  = 0.86       # 释放回到"伸直"判定：bend > baseline * EXIT_BEND_RATIO
MIN_BEND_DROP_DEFAULT    = 0.12       # 最低相对下降幅度（从历史高值下降的比例，避免小幅抖动=误触）
# ---------------- 延迟参数（仅保留两处用户指定的改动：COOLDOWN=4 / BEND_SMOOTH_ALPHA=0.75 是默认值，实际会被 speed 覆盖）----------------
# LONG_PRESS_FRAMES：用户要求恢复原设计（22帧≈730ms长按）
LONG_PRESS_FRAMES_DEFAULT = 22
# TAP_MAX_FRAMES：✨ 快速短按优先级。从弯曲进入到伸直回弹的总帧数 ≤ 此值 → 无论发生什么都按 TAP 处理（禁止变长按）
TAP_MAX_FRAMES_DEFAULT    = 18
# BEND_STABLE_THRESH：✨"弯曲保持稳定"阈值（默认值，会被 speed 覆盖）
BEND_STABLE_THRESH_DEFAULT  = 0.010
BEND_STABLE_MIN_HOLD_DEFAULT = 2
# MIN_BEND_HOLD：默认1，实际被 speed 覆盖
MIN_BEND_HOLD_DEFAULT    = 1
# COOLDOWN_FRAMES：用户明确要求保留改动（原14→4，解决连点第二下卡死）
COOLDOWN_FRAMES  = 4

# 平滑 / 基线学习
BEND_SMOOTH_ALPHA_DEFAULT = 0.75
BASELINE_ALPHA_IDLE  = 0.025   # IDLE下 伸直基线 慢速漂移 速率
BASELINE_ALPHA_FAST  = 0.40    # 刚回到伸直 且确定没 tap 时，快速对齐基线

# ---------------- 参数解析：hardness / speed → 8 个内部阈值（用户友好的 0~1 参数） ----------------
def resolve_tap_params(hardness: float | None = None,
                       speed:    float | None = None) -> dict:
    """把 0~1 的用户友好参数解析为具体阈值。

    hardness (0~1): 越大=需要弯得越狠才触发（小幅度弯曲彻底无视）
      ✨ 用户本轮明确：ENTER_BEND_RATIO「整体平移 -0.10」（所有档位同时降 0.10，H0~H1 间距 0.08 不变）
         目的：让点击判定的初始弯曲度整体增大 10%，避免"手指自然放松状态下有一点弯曲就被误判为点击"
      0 = 最松：ENTER=0.62 (旧 0.72 减 0.10), MIN_DROP=0.10  → 必须至少弯 38%
      0.5= 适中：ENTER=0.66 (旧 0.76 减 0.10), MIN_DROP=0.18  → 至少弯 34%
      1 = 最严：ENTER=0.70 (旧 0.80 减 0.10), MIN_DROP=0.26  → 至少弯 30%（间距 0.08 与旧设计一致）
    speed (0~1): 越大=响应越快（识别更灵敏、触发更快、更短的短按时间窗更宽容）
      0 = 最稳：ALPHA=0.55, MIN_HOLD=2, TAP_MAX=14, STABLE=0.020
      0.5= 默认快：ALPHA=0.77, MIN_HOLD=1, TAP_MAX=21, STABLE=0.0125
      1 = 极速：ALPHA=0.95, MIN_HOLD=0, TAP_MAX=28, STABLE=0.005
    """
    if hardness is None: hardness = 0.6
    if speed    is None: speed    = 0.6
    hardness = 0.0 if hardness < 0.0 else (1.0 if hardness > 1.0 else float(hardness))
    speed    = 0.0 if speed    < 0.0 else (1.0 if speed    > 1.0 else float(speed))

    # ✨ ENTER_BEND_RATIO：整体平移 -0.10（0.62 ~ 0.70），间距 0.08 保持不变（= 旧 0.72~0.80 线性下移 0.10）
    enter = 0.62 + hardness * 0.08           # 0.62 ~ 0.70
    drop  = 0.10 + hardness * 0.16           # 0.10 ~ 0.26（门槛B相对降幅，用户未指定改动，保持旧公式不变）
    # EXIT = ENTER + 0.06 形成滞回，ENTER 整体下移 EXIT 同步下移（更早识别回弹）
    exit_r = min(0.92, enter + 0.06)

    alpha       = 0.55 + speed * 0.40        # 0.55 ~ 0.95
    min_hold    = int(round(2.0 - 2.0 * speed))  # speed=0→2; 0.5→1; 1→0
    if min_hold < 0: min_hold = 0
    if min_hold > 3: min_hold = 3
    tap_max     = int(round(14.0 + 14.0 * speed))  # 14 ~ 28 帧
    stable_thr  = 0.020 - 0.015 * speed            # 0.020 ~ 0.005
    stable_min  = 2  # 保持稳定最小帧数不变（这是防抖用的，速度不应该影响它）
    long_press  = 22  # 用户要求恢复22帧长按，保持

    return dict(
        ENTER_BEND_RATIO   = round(enter, 4),
        EXIT_BEND_RATIO    = round(exit_r, 4),
        MIN_BEND_DROP      = round(drop, 4),
        BEND_SMOOTH_ALPHA  = round(alpha, 4),
        MIN_BEND_HOLD      = min_hold,
        TAP_MAX_FRAMES     = tap_max,
        LONG_PRESS_FRAMES  = long_press,
        BEND_STABLE_THRESH = round(stable_thr, 4),
        BEND_STABLE_MIN_HOLD = stable_min,
        COOLDOWN_FRAMES    = COOLDOWN_FRAMES,
        # 保留用户传入的 hardness/speed 原始值（给 UI 展示）
        _hardness          = hardness,
        _speed             = speed,
    )

class _S:
    IDLE        = "IDLE"          # 伸直待机（学习伸直基线）
    BEND_ARMED  = "BEND_ARMED"    # 刚检测到弯曲，等待下一帧确认
    BEND_HOLD   = "BEND_HOLD"     # 弯曲中：区分是 tap 还是 long-press
    PRESSED     = "PRESSED"       # long-press 已经触发 mouseDown，当前按住=拖拽
    TAPPED      = "TAPPED"        # tap已触发（只是1帧态，立刻入冷却）
    COOLDOWN    = "COOLDOWN"

class E:
    NONE       = 0
    TAP_CLICK  = 1   # 一次完整单击：mouseDown -> mouseUp (几乎同时完成)
    LONG_DOWN  = 2   # 长按开始：mouseDown，之后每帧 DRAG_MOVE
    LONG_UP    = 3   # 长按结束：mouseUp
    DRAG_MOVE  = 4   # 长按中持续移动


class FingerTapClickDetector:
    def __init__(self, hardness: float | None = None, speed: float | None = None):
        # 解析参数 → 所有阈值（文件默认值只是 fallback，CLI 参数会覆盖）
        self._p = resolve_tap_params(hardness=hardness, speed=speed)
        self.state = _S.IDLE
        self.event_this_frame = E.NONE

        self._smoothed_bend = None
        self._prev_smoothed_bend = None  # 上一帧 bend，判断"是否还在继续弯曲"
        self._baseline = None        # 伸直状态的 bend "高值"（用户动作前后都以此为参考）
        self._max_recent_bend = None # 近几秒内的最大伸直值（用于 MIN_BEND_DROP 判断）

        self._bend_hold_frames = 0       # 满足"弯曲稳定"条件的帧数
        self._bend_total_frames = 0      # 从 ENTER_BEND 到 EXIT 之间的总帧数（用于 TAP_MAX_FRAMES 快速短按优先级判定）
        self._stable_frames = 0          # 连续"弯曲深度稳定"的帧数
        self._cooldown_frames = 0

        self.tap_count = 0
        self.long_count = 0
        self.last_tap_time = None
        self.last_long_time = None

    # ---------------------- 重置：双手出现时调用，清空所有 bend/状态记忆，防误触 ----------------------
    def reset(self) -> None:
        """把状态机所有内部变量恢复为刚 __init__ 的初始值（双手手势场景避免弯曲积累→回单手就误点）。"""
        self.state = _S.IDLE
        self.event_this_frame = E.NONE
        self._smoothed_bend = None
        self._prev_smoothed_bend = None
        self._baseline = None
        self._max_recent_bend = None
        self._bend_hold_frames = 0
        self._bend_total_frames = 0
        self._stable_frames = 0
        self._cooldown_frames = 0
        # 计数器保留（让 UI 统计不丢），若想清空可自行注释放开：
        # self.tap_count = 0
        # self.long_count = 0
        # self.last_tap_time = None
        # self.last_long_time = None

    # 便捷属性（返回用户友好的 hardness/speed 百分比，供 UI 展示）
    @property
    def hardness_pct(self) -> int:
        return int(round(self._p["_hardness"] * 100))
    @property
    def speed_pct(self) -> int:
        return int(round(self._p["_speed"] * 100))
    @property
    def params(self) -> dict:
        return dict(self._p)

    # ---------------------- 主入口 ----------------------
    def update(self, bend: float | None) -> int:
        """
        Args:
            bend: 食指伸直度 = dist(TIP_8, MCP_5) / dist(MCP_5, MCP_17)；无手传None
        Returns:
            E.{NONE,TAP_CLICK,LONG_DOWN,LONG_UP,DRAG_MOVE} 本帧事件
        """
        self.event_this_frame = E.NONE

        # ---- 冷却 ----
        if self.state == _S.COOLDOWN:
            self._cooldown_frames -= 1
            if self._cooldown_frames <= 0:
                self.state = _S.IDLE
                self._bend_hold_frames = 0
            self._update_baseline(bend, can_update=True)
            return E.NONE

        # ---- 无手：回到IDLE；若之前按住则释放 ----
        if bend is None:
            if self.state in (_S.BEND_HOLD, _S.BEND_ARMED):
                pass  # 只是中间态，不算
            elif self.state == _S.PRESSED:
                self.event_this_frame = E.LONG_UP  # 手丢了 = 释放
                self._enter_cooldown()
            else:
                self.state = _S.IDLE
                self._smoothed_bend = None
                self._bend_hold_frames = 0
            return self.event_this_frame

        # ---- 有手：平滑 bend ----
        prev_b = self._smoothed_bend
        if self._smoothed_bend is None:
            self._smoothed_bend = float(bend)
        else:
            alpha = self._p["BEND_SMOOTH_ALPHA"]
            self._smoothed_bend = alpha * float(bend) + (1.0 - alpha) * self._smoothed_bend
        self._prev_smoothed_bend = prev_b
        # 近期伸直高值（在 IDLE/伸直阶段维护，保证 MIN_BEND_DROP 对新手势初始化友好）
        if self._max_recent_bend is None:
            self._max_recent_bend = self._smoothed_bend
        else:
            if self._smoothed_bend > self._max_recent_bend:
                self._max_recent_bend = self._smoothed_bend
            else:
                self._max_recent_bend = 0.985 * self._max_recent_bend + 0.015 * self._smoothed_bend

        # 初始化基线
        if self._baseline is None:
            self._baseline = self._smoothed_bend

        # 计算 参考阈值（基于 伸直基线）
        enter_t = self._baseline * self._p["ENTER_BEND_RATIO"]
        exit_t  = self._baseline * self._p["EXIT_BEND_RATIO"]
        # 下降幅度 = 相对近期伸直高值的跌幅
        drop = 0.0 if self._max_recent_bend is None else (self._max_recent_bend - self._smoothed_bend) / max(self._max_recent_bend, 1e-6)

        # ================= 状态流转 =================
        B = self._smoothed_bend

        if self.state == _S.IDLE:
            self._update_baseline(B, can_update=True)
            if B < enter_t and drop > self._p["MIN_BEND_DROP"]:
                self.state = _S.BEND_ARMED
                self._bend_hold_frames = 1
                self._bend_total_frames = 1
                self._stable_frames = 0
            elif B > self._baseline * 0.97:
                # 非常接近伸直 → 快速更新基线，避免基线长期滞后用户真实伸直姿态变化
                self._baseline = BASELINE_ALPHA_FAST * B + (1 - BASELINE_ALPHA_FAST) * self._baseline

        elif self.state == _S.BEND_ARMED:
            if B < enter_t:
                self._bend_hold_frames += 1
                self._bend_total_frames   += 1
                if self._bend_hold_frames >= self._p["MIN_BEND_HOLD"]:
                    self.state = _S.BEND_HOLD
            else:
                # 第2帧就不满足了 → 单帧尖峰，回IDLE
                self.state = _S.IDLE
                self._bend_hold_frames = 0
                self._bend_total_frames = 0
                self._stable_frames = 0

        elif self.state == _S.BEND_HOLD:
            # ✨ 总帧数：从 ENTER 开始累计（不管有没有稳定，都要算在 TAP_MAX_FRAMES 时间窗里）
            self._bend_total_frames += 1

            # ---------------- 修复1：先判断 EXIT（伸直回弹） → 优先于 LONG_PRESS ----------------
            if B >= exit_t:
                # -------- 修复3：快速短按优先级。总帧数 <= TAP_MAX_FRAMES 一律按 TAP，绝不能算长按 --------
                if self._bend_total_frames <= self._p["TAP_MAX_FRAMES"]:
                    self.event_this_frame = E.TAP_CLICK
                    self.tap_count += 1
                    self.last_tap_time = time.time()
                    self._enter_cooldown()
                # 常规 TAP：在长按阈值之前就回弹了，也是单击
                elif self._bend_hold_frames <= self._p["LONG_PRESS_FRAMES"]:
                    self.event_this_frame = E.TAP_CLICK
                    self.tap_count += 1
                    self.last_tap_time = time.time()
                    self._enter_cooldown()
                else:
                    # 真·超长按住后才松手 → 作为 LONG_UP 释放
                    self.event_this_frame = E.LONG_UP
                    self.last_long_time = time.time()
                    self._enter_cooldown()
                return self.event_this_frame

            # ---------------- 修复2：仍在弯曲中（exit_t 未到）→ 先计算"弯曲深度是否稳定" ----------------
            is_stable = True
            if self._prev_smoothed_bend is not None:
                delta = self._prev_smoothed_bend - B  # 正值=bend继续变小=更弯了
                if delta > self._p["BEND_STABLE_THRESH"]:
                    is_stable = False
            if not is_stable:
                self._stable_frames = 0
            else:
                self._stable_frames += 1
                # 只有连续 N 帧保持稳定，才把这一帧计入长按帧数
                if self._stable_frames >= self._p["BEND_STABLE_MIN_HOLD"]:
                    self._bend_hold_frames += 1

            # ---------------- 修复1：再判断 LONG_PRESS（最后再触发长按） ----------------
            if self._bend_hold_frames >= self._p["LONG_PRESS_FRAMES"] + 1:
                self.state = _S.PRESSED
                self.event_this_frame = E.LONG_DOWN
                self.long_count += 1
                self.last_long_time = time.time()

        elif self.state == _S.PRESSED:
            if B >= exit_t:
                # 松开 → release
                self.event_this_frame = E.LONG_UP
                self.last_long_time = time.time()
                self._enter_cooldown()
            else:
                # 还按住 → 持续发 DRAG_MOVE
                self.event_this_frame = E.DRAG_MOVE

        return self.event_this_frame

    # ---------------------- 内部 ----------------------
    def _update_baseline(self, b, *, can_update: bool):
        if b is None or self._baseline is None:
            return
        if not can_update:
            return
        # IDLE 下慢速适应；只对 bend 较高（≈伸直）的帧做基线更新，避免弯曲阶段把基线越学越低
        if b > self._baseline * 0.95:
            self._baseline = BASELINE_ALPHA_IDLE * b + (1.0 - BASELINE_ALPHA_IDLE) * self._baseline

    def _enter_cooldown(self):
        self.state = _S.COOLDOWN
        self._cooldown_frames = self._p["COOLDOWN_FRAMES"]
        self._bend_hold_frames = 0
        self._bend_total_frames = 0
        self._stable_frames = 0

    # ---------------------- 查询 / 绘制 ----------------------
    def metrics(self) -> dict:
        return {
            "state":          self.state,
            "event":          self.event_this_frame,
            "bend":           round(self._smoothed_bend or 0.0, 4),
            "baseline":       round(self._baseline       or 0.0, 4),
            "enter_t":        round((self._baseline or 0.0) * self._p["ENTER_BEND_RATIO"], 4),
            "exit_t":         round((self._baseline or 0.0) * self._p["EXIT_BEND_RATIO"],  4),
            "hold_frames":    self._bend_hold_frames,
            "total_frames":   self._bend_total_frames,
            "stable_frames":  self._stable_frames,
            "TAP_MAX":        self._p["TAP_MAX_FRAMES"],
            "LONG_MAX":       self._p["LONG_PRESS_FRAMES"],
            "cooldown_left":  max(0, self._cooldown_frames),
            "tap_count":      self.tap_count,
            "long_count":     self.long_count,
            # 把硬度速度也放进去，方便上层 UI 直接拿
            "hardness_pct":   self.hardness_pct,
            "speed_pct":      self.speed_pct,
        }
