import time
import numpy as np
from collections import deque  # 存最近 N 帧 bend_ratio，算一阶差分

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

⭐ 2026-08-22 用户明确：「点击应该计算的是最近几帧手指弯曲的变化量，而不是以绝对弯曲度判断」
根因：老方案用绝对弯曲度 bend/baseline → 但 baseline 依赖 palm 宽度(MCP_5→MCP_17)，手掌翻转/远近变化时
      palm_n 慢波动 → bend_ratio 也跟着变 → 明明没弯手指，也会在 enter_t / exit_t 上上下下 → 疯狂误判。
新方案（一阶差分 + 绝对阈值兜底，双判据 OR 兼容）：
  主判据 = 最近 K 帧 bend_ratio（=smoothed_bend / baseline）的「一阶差分累积和」
    · 按下 = 近 5 帧 Δbend_ratio 累计 ≤ -TAP_DOWN_DI_SUM（bend_ratio 变小了 = 食指真的主动弯了）
    · 松开 = 近 5 帧 Δbend_ratio 累计 ≥ +TAP_UP_DI_SUM（bend_ratio 变大了 = 食指真的主动回弹伸直）
  兜底判据（保留旧工程约定，高级用户仍能用老参数习惯）：
    · 原来的 enter_t/exit_t + MIN_BEND_DROP 仍作为「绝对弯曲度门槛」兜底，保证：
      - 纯差分的小抖动（比如差分刚好触发但实际根本没弯）= 绝对门槛拦下
      - 差分窗口太窄没接住、但用户确实明显弯了 = 绝对门槛仍能触发（兼容）
  → 手掌翻转/远近变化是低频慢漂（相邻几帧 Δbend_ratio ≈ ±0.01）→ 差分和 = 0 → 绝不误判。
  → 只有真的"食指主动弯下去再弹回来"才会形成大幅差分正负脉冲 → 精准识别。

状态机（严格时序：伸直→弯→伸直 = 一次Tap；弯曲超时=Hold拖拽）：

 STRAIGHT_IDLE (bend≈伸直, 持续学习伸直基线)
   │ (主：近5帧Δ累积 ≥ DOWN_DI  且  B < enter_t*1.02)  OR  (兜底：B < enter_t 且 drop>MIN_DROP)
   ▼
 BEND_ARMED (检测到弯曲，等 1 帧避免瞬间尖峰)
   │ 下一帧仍满足弯曲
   ▼
 BEND_HOLD  (弯曲持续中 → 计时)
   ├─ (主：近5帧Δ累积 ≥ UP_DI  或  兜底：B≥exit_t)  且 hold_frames <= LONG_PRESS  → TAP_CLICK → 单击
   └─ (hold_frames > LONG_PRESS  且 bend 仍<baseline*0.80)                           → LONG_PRESS → 长按拖拽开始
                │
                ▼
          PRESSED_HOLD (保持按住，拖拽跟随)
                │  (主：差分释放  或  兜底：B≥exit_t → 释放)
                ▼
          COOLDOWN (4帧冷却 + 重置伸直基线慢速学习) → IDLE
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
# ⭐ 双击判定（2026-09-10，用户要求：食指勾按有且只有「单击/双击」两种结果，严格互斥）：
#   第一次 Tap 松开后【不立即单击】，进入 WAIT_DBL 窗口：
#     · 窗口内完成第二次 Tap → DOUBLE_TAP（只发一次 doubleClick，挂起的单击作废 → 物理点击恰好 2 次）
#     · 窗口超时无第二次   → 补发 TAP_CLICK（单击延迟确认，杜绝双击同时出两个单击）
DOUBLE_TAP_WINDOW_SEC     = 0.38  # 第一次Tap松开 → 第二次弯曲开始 的最大间隔（秒；人类快速双击≈0.15~0.30s）
DOUBLE_TAP_COOLDOWN_FRAMES = 6    # 双击后的冷却帧数（比单击4帧略长，让伸直基线稳一下）
# ⭐ 实验开关（2026-09-10）：双击判定归属
#   True  = 每次勾按【立即】click()，单击零延迟；快速连勾两次 = 两组 down/up，
#           由 Windows 双击时间（默认500ms/位置容差）自然识别为双击（和物理鼠标行为完全一致；
#           代价：双击时第一击也会送达应用，按钮类元素可能先吃到一次单击）。
#   False = 延迟确认：第一次Tap挂起 0.38s，窗口内第二次Tap只发 doubleClick（单/双击严格互斥，
#           代价：每次单击慢 0.38s）。
OS_DOUBLE_CLICK = False

# ⭐ 诊断开关（2026-09-10 双击排查）：True 时每次勾按打印状态机转移行 [TAP-DIAG]
#   ARM=起弯  TAP=点击发出  LONG=长按转换(双击被吞的头号嫌疑)  SPIKE=尖峰拒绝  NONE帧=掌心不稳被门控
TAP_DIAG = True

# ⭐ 2026-08-22 差分点击新增常量
DI_WINDOW = 5                 # 差分窗口：最近 5 帧 bend_ratio（30fps=167ms，刚好是"按下去"的动作时长）
# ⭐ 2026-08-24 适配 v2 弯曲度度量（关节角度融合版，Δ 比旧 dist_bend 略小 15~20%）：
#   差分门限整体 ×0.70 左右，保证灵敏度回到或略高于纯旧度量的水平
TAP_DOWN_DI_SUM = 0.10        # 按下门限：ΣΔbend_ratio ≤ -0.10 → 食指真的主动弯了（bend_ratio 下降）
#   （v2 关节角度 手指从伸直→90°弯，bend_ratio 变化≈0.85→0.30=-0.55 → 取 0.10 半程即可触发，5帧内必须完成）
TAP_UP_DI_SUM   = 0.07        # 松开门限：ΣΔbend_ratio ≥ +0.07 → 食指真的主动回伸（bend_ratio 上升）
MIN_ABS_BEND_RATIO_FOR_DELTA = 0.75  # 差分触发后仍必须满足 bend_ratio ≤ 这个兜底（防止"差分刚好到了、但其实没弯"）
                                      # ✅ v3 统一弯曲度量（DIP=0.40 主力）后校准：
                                      #   中等点击(DIP 弯至 120°=60°折) → bratio≈0.70 ≤ 0.75 ✅（能过）
                                      #   移动微角度 ±2° → bratio 变化≈0.004 → DI=0.004 远不够，不会误触
                                      #   深点击(DIP 弯至 30°=150°折) → bratio≈0.45 ≤ 0.75 ✅ 肯定过

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

    ⭐ 2026-08-22 差分点击新增：down/up 差分和 随 hardness/speed 联动
      · hardness↑ = DOWN_DI↑（更严=必须更大幅度弯下来才触发），UP_DI 同步
      · speed↑ = DI_WINDOW↓（更灵=窗口更短，更短时间内的变化就能触发）
    """
    if hardness is None: hardness = 0.6
    if speed    is None: speed    = 0.6
    hardness = 0.0 if hardness < 0.0 else (1.0 if hardness > 1.0 else float(hardness))
    speed    = 0.0 if speed    < 0.0 else (1.0 if speed    > 1.0 else float(speed))

    # ✨ ENTER_BEND_RATIO：v3 统一度量校准档 0.65~0.73（旧 0.62~0.70 略提 0.03 = 中等点击能过，移动不误触）
    #    h=0.6(默认) → enter=0.698 → enter_t=baseline*0.7 ≈ 0.85*0.7=0.595（中等点击 B≈0.58 < 0.595 → down_abs 刚好过）
    enter = 0.65 + hardness * 0.08           # 0.65 ~ 0.73
    drop  = 0.10 + hardness * 0.16           # 0.10 ~ 0.26（门槛B相对降幅，保持稳定）
    # EXIT = ENTER + 0.06 形成滞回，ENTER 整体下移 EXIT 同步下移（更早识别回弹）
    exit_r = min(0.92, enter + 0.06)

    alpha       = 0.55 + speed * 0.40        # 0.55 ~ 0.95
    # ⭐ 2026-08-22 修复"点击经常误判"：最小稳定帧数 MIN_BEND_HOLD 全档位 +1（过滤单帧/双帧假尖峰）
    #   原映射（速度越快允许越少帧）：speed=0→2, 0.5→1, 1→0
    #   新映射：speed=0→3, 0.5→2, 0.6→2, 0.8→1, 1→1
    #     · 新手慢速档(0~0.2)：3 帧弯曲稳定 → 必须"连续 3 帧都明显弯"才进 BEND_HOLD（最稳，误触几乎为 0）
    #     · 默认档(0.4~0.7)：2 帧弯曲稳定（原极速档=1 帧，现在默认档=2 帧，响应速度仍够、但过滤单帧噪声）
    #     · 极速档(0.8~1.0)：1 帧稳定（高级用户可接受偶尔误触换速度）
    min_hold_raw = int(round(2.0 - 2.0 * speed)) + 1
    min_hold = 1 if min_hold_raw < 1 else (4 if min_hold_raw > 4 else min_hold_raw)
    tap_max     = int(round(14.0 + 14.0 * speed))  # 14 ~ 28 帧
    stable_thr  = 0.020 - 0.015 * speed            # 0.020 ~ 0.005
    stable_min  = 2  # 保持稳定最小帧数不变（这是防抖用的，速度不应该影响它）
    long_press  = 22  # 用户要求恢复22帧长按，保持

    # ⭐ 2026-08-22 差分点击参数联动
    #   - DOWN_DI_SUM：hardness=0 → 0.08（松），hardness=1 → 0.14（严）；speed 越大略减（更灵）
    #   - UP_DI_SUM  : 同步 = DOWN_DI_SUM * 0.72（松开通常幅度小，按下去必须更明确）
    #   - DI_WINDOW  : speed=0 → 6帧，speed=0.5 → 5帧，speed=1 → 4帧
    #   ⭐ 2026-08-24 适配 v2 关节角度弯曲度：基准整体 ×0.72（补偿 v2 Δ 比旧 dist_bend 略小的问题）
    down_di = 0.08 + 0.06 * hardness - 0.022 * speed
    down_di = 0.06 if down_di < 0.06 else (0.16 if down_di > 0.16 else float(round(down_di, 4)))
    up_di   = float(round(down_di * 0.72, 4))
    di_win  = int(round(6.0 - 2.0 * speed))
    di_win  = 4 if di_win < 4 else (7 if di_win > 7 else di_win)

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
        # ⭐ 差分点击新增参数
        DI_WINDOW          = di_win,
        TAP_DOWN_DI_SUM    = down_di,
        TAP_UP_DI_SUM      = up_di,
        MIN_ABS_BEND_RATIO_FOR_DELTA = MIN_ABS_BEND_RATIO_FOR_DELTA,
        # 保留用户传入的 hardness/speed 原始值（给 UI 展示）
        _hardness          = hardness,
        _speed             = speed,
    )

class _S:
    IDLE        = 0  # 伸直待机（学习伸直基线） → 匹配 hands_tasks.py _STATE_NAMES[0]="IDLE"
    BEND_ARMED  = 1  # 刚检测到弯曲，等待下一帧确认 → 映射 "ARM"（_STATE_NAMES[1]）
    BEND_HOLD   = 2  # 弯曲中：区分是 tap 还是 long-press → 映射 "HOLD"（_STATE_NAMES[2]）
    PRESSED     = 3  # long-press 已经触发 mouseDown，当前按住=拖拽 → 映射 "PRS"（_STATE_NAMES[3]）
    COOLDOWN    = 4  # 冷却 → 映射 "COOL"（_STATE_NAMES[4]）
    TAPPED      = 5  # tap已触发（1帧瞬态，立刻入COOLDOWN；_STATE_NAMES无映射=不会出现在日志，安全）
    WAIT_DBL    = 6  # 第一次Tap完成、单击挂起，等待第二次Tap构成双击（双击窗口）

class E:
    NONE       = 0
    TAP_CLICK  = 1   # 一次完整单击：mouseDown -> mouseUp (双击窗口超时后补发)
    LONG_DOWN  = 2   # 长按开始：mouseDown，之后每帧 DRAG_MOVE
    LONG_UP    = 3   # 长按结束：mouseUp
    DRAG_MOVE  = 4   # 长按中持续移动
    DOUBLE_TAP = 5   # 双击：双击窗口内完成第二次 Tap（mouse doubleClick，单击不补发）


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

        # ⭐ 2026-08-22 差分点击新增状态
        #   _bend_ratio_hist：连续最近 N 帧 bend_ratio（=smoothed_bend / baseline），maxlen=8 够覆盖窗口波动
        self._bend_ratio_hist: deque = deque(maxlen=8)
        #   标记当前 ARMED→HOLD→PRESSED 这段是「差分进入」还是「绝对兜底进入」，影响 EXIT 判据优先级
        self._armed_by_delta = False
        #   本帧计算出的差分和（给 metrics/UI 展示用）
        self._di_sum_neg_now = 0.0  # ΣΔ<0 的总和 = 按下去的累积（负值；触发时 ≤ -DOWN_DI）
        self._di_sum_pos_now = 0.0  # ΣΔ>0 的总和 = 抬起来的累积（正值；触发时 ≥ +UP_DI）

        # ⭐ 双击判定状态
        self._double_candidate = False  # 当前这段弯曲是否始于双击窗口内（=第二次Tap候选）
        self._tap_pending = False       # WAIT_DBL 中是否挂着一个待确认的单击
        self._wait_deadline = 0.0       # WAIT_DBL 超时时刻（time.time()）
        # ⭐ TAP-DIAG 诊断
        self._dbg_arm_t = None          # 最近一次 ARMED 时刻
        self._dbg_none_in_tap = 0       # 当前 ARM/HOLD 段内 bend=None（掌心不稳门控）帧数

        self.tap_count = 0
        self.long_count = 0
        self.double_count = 0
        self.last_tap_time = None
        self.last_long_time = None
        self.last_double_time = None

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
        # 差分点击状态也同步清空
        self._bend_ratio_hist.clear()
        self._armed_by_delta = False
        self._di_sum_neg_now = 0.0
        self._di_sum_pos_now = 0.0
        # 双击判定状态同步清空（挂起的单击作废）
        self._double_candidate = False
        self._tap_pending = False
        self._wait_deadline = 0.0
        # 计数器保留（让 UI 统计不丢），若想清空可自行注释放开：
        # self.tap_count = 0
        # self.long_count = 0
        # self.double_count = 0
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

    # ---------------------- 内部：一阶差分计算 ----------------------
    def _compute_di_sums(self, bend_ratio_now: float):
        """把当前 bend_ratio 入队，然后计算：
          - di_sum_neg = 最近 DI_WINDOW 帧 一阶差分中 负值的总和（向下按：bend_ratio↓ → Δ<0，累加其绝对值）
            注意：bend_ratio 越大=越伸直；所以"食指弯下去" = bend_ratio 变小 → 每帧 Δ=br_now - br_prev < 0。
            为了 UI 展示和阈值判断直观，di_sum_neg 存「负值累积的绝对值」，即一个正数。
          - di_sum_pos = 最近 DI_WINDOW 帧 一阶差分中 正值的总和（向上抬：bend_ratio↑ → Δ>0）
        返回：(di_sum_neg, di_sum_pos)，都是非负实数。
        """
        self._bend_ratio_hist.append(float(bend_ratio_now))
        win = self._p["DI_WINDOW"]
        if len(self._bend_ratio_hist) < 2:
            self._di_sum_neg_now = 0.0
            self._di_sum_pos_now = 0.0
            return 0.0, 0.0
        # 取最近 win 个（最多 deque 里有的）算差分
        hist = list(self._bend_ratio_hist)
        if len(hist) > win + 1:
            hist = hist[-(win + 1):]
        neg = 0.0
        pos = 0.0
        for i in range(1, len(hist)):
            d = hist[i] - hist[i - 1]  # bend_ratio 差分（正=伸直，负=弯曲）
            if d < 0:
                neg += (-d)  # 负值→按下去的幅度累积（正数）
            elif d > 0:
                pos += d     # 正值→抬起来的幅度累积（正数）
        self._di_sum_neg_now = float(round(neg, 5))
        self._di_sum_pos_now = float(round(pos, 5))
        return self._di_sum_neg_now, self._di_sum_pos_now

    # ---------------------- TAP-DIAG 诊断 ----------------------
    def _diag(self, msg: str):
        if TAP_DIAG:
            try:
                print(f"[TAP-DIAG] t={time.time():.3f} {msg}", flush=True)
            except Exception:
                pass

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
                self._armed_by_delta = False
            self._update_baseline(bend, can_update=True)
            return E.NONE

        # ---- 无手：回到IDLE；若之前按住则释放 ----
        if bend is None:
            # ⭐ 差分队列清空（下一次有手是全新场景，不能带上旧差分）
            self._bend_ratio_hist.clear()
            self._di_sum_neg_now = 0.0
            self._di_sum_pos_now = 0.0
            if self.state in (_S.BEND_HOLD, _S.BEND_ARMED):
                self._dbg_none_in_tap += 1
                if self._dbg_none_in_tap <= 3 or self._dbg_none_in_tap % 10 == 0:
                    self._diag(f"⚠ bend=None(掌心不稳门控) 状态={self.state} 累计{self._dbg_none_in_tap}帧")
                # 只是中间态，不算
            elif self.state == _S.WAIT_DBL:
                # 手丢失 = 双击窗口无意义（也没有可点击的有效位置）→ 挂起的单击作废
                self.state = _S.IDLE
                self._tap_pending = False
                self._double_candidate = False
                self._wait_deadline = 0.0
                self._bend_hold_frames = 0
                self._bend_total_frames = 0
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

        # ---- bend_ratio（归一化弯曲度 = 当前伸直度 / 伸直基线）+ 一阶差分 ----
        #   含义：1.00 = 完全伸直（和基线一致）；0.50 = 弯到一半；0.25 = 弯得很深
        B = self._smoothed_bend
        bend_ratio = B / max(self._baseline, 1e-6)
        # ⭐ 计算差分（push 到历史队列 + 返回 ΣΔ负 / ΣΔ正）
        di_sum_down, di_sum_up = self._compute_di_sums(bend_ratio)

        # 计算 参考阈值（基于 伸直基线）
        enter_t = self._baseline * self._p["ENTER_BEND_RATIO"]
        exit_t  = self._baseline * self._p["EXIT_BEND_RATIO"]
        # 下降幅度 = 相对近期伸直高值的跌幅
        drop = 0.0 if self._max_recent_bend is None else (self._max_recent_bend - self._smoothed_bend) / max(self._max_recent_bend, 1e-6)

        # ⭐ 差分/绝对双判据的"按下判定"和"松开判定"封装
        #   down_trigger = 进入弯曲（IDLE→ARMED）
        down_delta = (di_sum_down >= self._p["TAP_DOWN_DI_SUM"]) and (
            bend_ratio <= min(self._p["MIN_ABS_BEND_RATIO_FOR_DELTA"],
                              self._p["ENTER_BEND_RATIO"] + 0.06)
        )
        down_abs   = (B < enter_t) and (drop > self._p["MIN_BEND_DROP"])
        down_trigger = down_delta or down_abs

        #   up_trigger = 从弯曲回到伸直（BEND_HOLD→TAP / PRESSED→LONG_UP）
        #   【优先级】：如果当前这段是「差分进入」(armed_by_delta=True)，先看差分松手；否则看绝对松手；
        #   再用另一种判据兜底 OR，确保不会漏。
        if self._armed_by_delta:
            up_delta_first = di_sum_up >= self._p["TAP_UP_DI_SUM"]
            up_abs_fallback = B >= exit_t
            up_trigger = up_delta_first or up_abs_fallback
        else:
            up_abs_first = B >= exit_t
            up_delta_fallback = di_sum_up >= self._p["TAP_UP_DI_SUM"]
            up_trigger = up_abs_first or up_delta_fallback

        # ================= 状态流转 =================

        if self.state == _S.IDLE:
            self._update_baseline(B, can_update=True)
            if down_trigger:
                self.state = _S.BEND_ARMED
                self._bend_hold_frames = 1
                self._bend_total_frames = 1
                self._stable_frames = 0
                self._double_candidate = False
                self._dbg_arm_t = time.time()
                self._dbg_none_in_tap = 0
                self._diag(f"① ARM 起弯 ratio={bend_ratio:.3f} Σ下={di_sum_down:.3f}/Σ上={di_sum_up:.3f} "
                           f"入口={'差分' if (down_delta and not down_abs) else '绝对'}")
                # ⭐ 标记这次进入方式（决定 EXIT 判据的优先级）
                self._armed_by_delta = bool(down_delta and not down_abs)  # 纯差分进入 = True；兜底或同时 = False 归绝对
            elif B > self._baseline * 0.97:
                # 非常接近伸直 → 快速更新基线，避免基线长期滞后用户真实伸直姿态变化
                self._baseline = BASELINE_ALPHA_FAST * B + (1 - BASELINE_ALPHA_FAST) * self._baseline

        elif self.state == _S.WAIT_DBL:
            # 第一次 Tap 已完成（单击挂起、不立即触发），等第二次 Tap 构成双击
            self._update_baseline(B, can_update=True)
            if B > self._baseline * 0.97:
                self._baseline = BASELINE_ALPHA_FAST * B + (1 - BASELINE_ALPHA_FAST) * self._baseline
            if down_trigger:
                # 第二次弯曲在窗口内开始 → 走常规 ARMED/HOLD，标记为双击候选
                self.state = _S.BEND_ARMED
                self._bend_hold_frames = 1
                self._bend_total_frames = 1
                self._stable_frames = 0
                self._double_candidate = True
                self._armed_by_delta = bool(down_delta and not down_abs)
            elif time.time() > self._wait_deadline:
                # 窗口超时仍无第二次弯曲 → 确认单击，补发 TAP_CLICK
                self._tap_pending = False
                self._double_candidate = False
                self.event_this_frame = E.TAP_CLICK
                self.tap_count += 1
                self.last_tap_time = time.time()
                self._enter_cooldown()
                return self.event_this_frame

        elif self.state == _S.BEND_ARMED:
            # ARMED 的下一帧：仍需满足"还没回到伸直"（不能刚一帧就直接弹回去=尖峰）
            still_down = (di_sum_down >= max(0.0, self._p["TAP_DOWN_DI_SUM"] * 0.35)) or (B < enter_t + self._baseline * 0.03)
            if still_down:
                self._bend_hold_frames += 1
                self._bend_total_frames += 1
                if self._bend_hold_frames >= self._p["MIN_BEND_HOLD"]:
                    self.state = _S.BEND_HOLD
            else:
                # 第2帧就不满足了 → 单帧尖峰
                self._diag(f"✗ 尖峰拒绝(第2帧未保持弯曲) ratio={bend_ratio:.3f} Σ下={di_sum_down:.3f} "
                           f"{'双击候选→回WAIT' if self._double_candidate else '→IDLE'}")
                if self._double_candidate:
                    # 双击窗口内的第二次弯曲只是尖峰 → 回 WAIT_DBL 继续等（窗口到期自动补单击）
                    self.state = _S.WAIT_DBL
                    self._double_candidate = False
                else:
                    self.state = _S.IDLE
                self._bend_hold_frames = 0
                self._bend_total_frames = 0
                self._stable_frames = 0
                self._armed_by_delta = False

        elif self.state == _S.BEND_HOLD:
            # ✨ 总帧数：从 ENTER 开始累计（不管有没有稳定，都要算在 TAP_MAX_FRAMES 时间窗里）
            self._bend_total_frames += 1

            # ---------------- 修复1：先判断 EXIT（伸直回弹） → 优先于 LONG_PRESS ----------------
            if up_trigger:
                # -------- 修复3：快速短按优先级。总帧数 <= TAP_MAX_FRAMES 一律按 TAP，绝不能算长按 --------
                # 判定本次是 Tap（快速短按 或 长按阈值前回弹）还是 真·长按释放
                is_tap = (self._bend_total_frames <= self._p["TAP_MAX_FRAMES"]
                          or self._bend_hold_frames <= self._p["LONG_PRESS_FRAMES"])
                _seg_ms = (time.time() - self._dbg_arm_t) * 1000.0 if self._dbg_arm_t else 0.0
                _none_n = self._dbg_none_in_tap
                if OS_DOUBLE_CLICK and is_tap:
                    # ★ 即时模式：每次勾按回弹立刻单击（双击交给 Windows 双击时间自然识别）
                    self._double_candidate = False
                    self._tap_pending = False
                    self.event_this_frame = E.TAP_CLICK
                    self.tap_count += 1
                    self.last_tap_time = time.time()
                    self._diag(f"② TAP点击发出(即时) 段长{self._bend_total_frames}帧≈{_seg_ms:.0f}ms "
                               f"None门控{_none_n}帧 ratio={bend_ratio:.3f}")
                    self._enter_cooldown()
                elif is_tap and self._double_candidate:
                    # 延迟确认模式：双击窗口内完成第二次 Tap → DOUBLE_TAP（挂起的单击作废，
                    #   只发一次 doubleClick = 物理点击恰好 2 次，严格互斥）
                    self._double_candidate = False
                    self._tap_pending = False
                    self.event_this_frame = E.DOUBLE_TAP
                    self.double_count += 1
                    self.last_double_time = time.time()
                    self._enter_cooldown(DOUBLE_TAP_COOLDOWN_FRAMES)
                elif is_tap:
                    # 第一次 Tap：不立即单击 → 进双击窗口等第二次（超时在 WAIT_DBL 补发单击）
                    self.state = _S.WAIT_DBL
                    self._tap_pending = True
                    self._double_candidate = False
                    self._wait_deadline = time.time() + DOUBLE_TAP_WINDOW_SEC
                    self._bend_hold_frames = 0
                    self._bend_total_frames = 0
                    self._stable_frames = 0
                    self._armed_by_delta = False
                    # 同 _enter_cooldown 的差分清零手法：推两帧伸直值，让窗口内 ΣΔ 立刻回 0
                    if self._baseline is not None and self._smoothed_bend is not None:
                        for _ in range(2):
                            self._bend_ratio_hist.append(
                                self._smoothed_bend / max(self._baseline, 1e-6))
                else:
                    # 真·超长按住后才松手 → 作为 LONG_UP 释放；若此前挂着单击/双击候选，
                    #   拖拽语义优先，挂起的单击作废（避免 click+mouseDown 连在同一点）
                    self._tap_pending = False
                    self._double_candidate = False
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
                self._diag(f"⚠⚠ 转长按LONG_DOWN(没检测到回弹!) 总{self._bend_total_frames}帧 "
                           f"≈{(time.time()-self._dbg_arm_t)*1000.0:.0f}ms None门控{self._dbg_none_in_tap}帧 "
                           f"ratio={bend_ratio:.3f} Σ上={di_sum_up:.3f}(需{self._p['TAP_UP_DI_SUM']}) "
                           f"→ 双击被吞：两次弯曲被合成一段按住")
                # 第二次弯曲演变为长按拖拽 → 挂起的单击作废（拖拽语义优先）
                self._tap_pending = False
                self._double_candidate = False

        elif self.state == _S.PRESSED:
            if up_trigger:
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

    def _enter_cooldown(self, frames: int | None = None):
        self.state = _S.COOLDOWN
        self._cooldown_frames = int(frames) if frames is not None else self._p["COOLDOWN_FRAMES"]
        self._bend_hold_frames = 0
        self._bend_total_frames = 0
        self._stable_frames = 0
        self._armed_by_delta = False
        self._tap_pending = False
        self._double_candidate = False
        self._wait_deadline = 0.0
        # ⭐ 冷却时不清空 bend_ratio_hist（保留弯曲尾部的差分，让"连点"的第二次 tap 也能被差分准确识别）
        #   但要手动 push 一个"当前伸直值"，让下一帧 IDLE 阶段的 ΣΔ 不会被上一个 tap 的尾巴误触发
        if self._baseline is not None and self._smoothed_bend is not None:
            # 推两帧伸直基线 → 差分立刻回 0，防止连点残留
            for _ in range(2):
                self._bend_ratio_hist.append(self._smoothed_bend / max(self._baseline, 1e-6))

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
            "double_count":   self.double_count,
            # 双击窗口剩余秒数（WAIT_DBL 中 >0；其余状态=0）
            "dbl_window_left": round(max(0.0, self._wait_deadline - time.time()), 3)
                                if self.state == _S.WAIT_DBL else 0.0,
            # ⭐ 差分点击新增字段（给 UI 条/调试）
            "bend_ratio":     round(
                (self._smoothed_bend / max(self._baseline, 1e-6)) if (self._baseline is not None and self._smoothed_bend is not None) else 0.0,
                4),
            "di_sum_down":    round(self._di_sum_neg_now, 4),   # 向下按=正值（累积 Σ|Δ负|）
            "di_sum_up":      round(self._di_sum_pos_now, 4),   # 向上抬=正值（累积 ΣΔ正）
            "DI_DOWN_THR":    round(self._p["TAP_DOWN_DI_SUM"], 4),
            "DI_UP_THR":      round(self._p["TAP_UP_DI_SUM"],   4),
            "DI_WINDOW":      self._p["DI_WINDOW"],
            "armed_by_delta": bool(self._armed_by_delta),
            # 把硬度速度也放进去，方便上层 UI 直接拿
            "hardness_pct":   self.hardness_pct,
            "speed_pct":      self.speed_pct,
        }
