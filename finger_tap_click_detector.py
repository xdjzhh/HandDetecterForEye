import time
import math
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
新方案（一阶差分变化量是【唯一】起弯判据，2026-09-21 用户拍板废除绝对弯曲度入口）：
  按下 = 最近 DI_WINDOW_SEC 秒 bend_ratio 的负差分累积 ≥ TAP_DOWN_DI_SUM，且已离开伸直位
    · 必须真的发生「下弯变化」：手指本来就弯着 / 黑帧恢复时仍弯着 / 长按回直途中还弯着，
      只要没有新的下弯变化量 → 一律不起弯（绝对弯曲度 B<enter_t 兜底入口已删除，它正是
      "长按回直被识别成双击"的根因：LONG_UP 后手指没动却被当成新一次起弯）
    · bend_ratio ≤ MIN_ABS_BEND_RATIO_FOR_DELTA 只是"确实弯离伸直位"的确认门，单独不触发
  松开 = 必须【真的回直】（2026-09-21 用户拍板，长按/双击隔离；同日真机日志补第三门）：
    ① 回直变化量：从本段弯曲谷底 bend_ratio 回升 ≥ RECOVER_RISE_MIN
    ② EMA 伸直位：平滑 bend_ratio ≥ RECOVER_RATIO_MIN（基本伸直）
    ③ raw 伸直位：原始测量 bend ≥ 基线*0.985（手指物理位置真到位；②会在回弹运动途中
       被 EMA 惯性提前满足——真机误触 raw=0.77 未直、EMA 0.881 险过②，③拦之）
    三个门 AND，再与快速回弹线索（正差分累积 ≥ TAP_UP_DI_SUM 或 B≥exit_t）AND。
    长按中手指不回直（微松到 ratio 0.55~0.70 仍明显弯着，谷底回升量小；或回弹途中 raw 未到位）
    → 不松手、不产生第一次 Tap、不进 WAIT_DBL → 长按中在状态机层面不可能凑出双击。
  点击质量门：Tap/双击任一击回弹时，整段谷底必须 ≤ TAP_MIN_BEND_RATIO(0.66)，
    半弯抖动（谷底 0.69、raw 才到 0.52）静默取消，首击不发点击、无效第二击回 WAIT 不占名额。
    ARMED 第二帧"尖峰拒绝"对深弯（起弯帧谷底 ≤0.55）豁免：1 帧深弯后立刻弹直=最短合法点击，
    不得整条吞掉（真机日志：真第二击被吞→后面没松直的抖动顶包成双击）。
  双击第二击 = 第一次回直变化【之后】又出现一段新的下弯变化（WAIT_DBL down_trigger / 趋势探测）；
    长按与双击互斥：①长按中途的微松不构成第一次 Tap（上面的真回直门）；
    ②LONG_DOWN 即清掉全部挂起的单击/双击状态，LONG_UP 之后没有新下弯变化就
    不会产生任何可配对的点击（不需要额外的"伸直观测"等复杂状态）。
  → 手掌翻转/远近变化是低频慢漂（相邻几帧 Δbend_ratio ≈ ±0.01）→ 差分和 = 0 → 绝不误判。
  → 只有真的"食指主动弯下去再弹回来"才会形成大幅差分正负脉冲 → 精准识别。

状态机（严格时序：伸直→弯→伸直 = 一次Tap；弯曲超时=Hold拖拽）：

 STRAIGHT_IDLE (bend≈伸直, 持续学习伸直基线)
   │ 唯一入口：近窗 Δ负累积 ≥ DOWN_DI（真的发生下弯变化）且 bend_ratio 已离开伸直位
   ▼
 BEND_ARMED (检测到弯曲，等 1 帧避免瞬间尖峰)
   │ 下一帧仍满足弯曲
   ▼
 BEND_HOLD  (弯曲持续中 → 计时)
   ├─ 回弹线索(Δ正累积≥UP_DI 或 B≥exit_t) 且【真回直】(谷底回升≥0.30 且 ratio≥0.88) 且短按窗 → TAP_CLICK
   └─ hold 超过 LONG_PRESS（仍弯着；中途微松不满足真回直 → 不松手、不计Tap）              → LONG_PRESS → 长按拖拽开始
                │
                ▼
          PRESSED_HOLD (保持按住，拖拽跟随)
                │  释放同样要求【真回直】(谷底回升≥0.30 且 ratio≥0.88)；黑帧(bend=None)立即释放（既有行为）
                ▼
          COOLDOWN (4帧冷却 + 重置伸直基线慢速学习) → IDLE
"""

# ⭐ 统一时钟源（2026-09-18 墙钟化）：全文件一律 time.monotonic()，不受系统校时回拨影响。
#   写成模块级可替换绑定，供等价比对 harness 注入虚拟时钟（生产环境勿改）。
_CLOCK = time.monotonic

# 阈值（文件级默认值；实际运行时可通过构造参数 hardness/speed 被 CLI 参数覆盖）
ENTER_BEND_RATIO_DEFAULT = 0.78       # 进入弯曲判定：bend < baseline * ENTER_BEND_RATIO
EXIT_BEND_RATIO_DEFAULT  = 0.86       # 释放回到"伸直"判定：bend > baseline * EXIT_BEND_RATIO
MIN_BEND_DROP_DEFAULT    = 0.12       # 最低相对下降幅度（仅双指右键检测器的绝对兜底仍在用；左键起弯已废除绝对入口）
# ---------------- 墙钟时间归一化（2026-09-18，帧计数→time.monotonic 秒）----------------
# 所有"物理时长"阈值一律以秒表示，按 NOMINAL_FPS=30 由旧帧阈值【精确】换算：
#   状态机首帧计 1（ARM 当帧 hold/total=1），"N 帧成立"= 经过 N-1 个帧间隔 → 阈值=(N-1)*FRAME_SEC；
#   冷却类"进入后数 N 次 update 结束"= N 个帧间隔 → 阈值=N*FRAME_SEC。
# 30fps 下事件序列与旧帧计数版逐帧一致（_sim_clock.py 等价比对）；20/60fps 下物理时长恒定。
NOMINAL_FPS = 30.0
FRAME_SEC   = 1.0 / NOMINAL_FPS
MAX_DT_SEC  = 2.0 * FRAME_SEC        # 单帧 dt 钳幅：卡顿/恢复喂帧不得一次跳秒（不影响 30fps 正常路径）

# 墙钟阈值比较容差：逐帧累加的 elapsed 与单次乘法得到的阈值常量在帧边界可能相差 1 个浮点 ulp
# （如 22 次 +1/30 与 22/30），1µs 容差只吃掉舍入误差、不改变物理语义（真机帧抖动远大于此）。
_EPS_SEC = 1e-6
def _t_ge(elapsed: float, thr: float) -> bool:   # elapsed >= thr（边界帧成立）
    return elapsed + _EPS_SEC >= thr
def _t_le(elapsed: float, thr: float) -> bool:   # elapsed <= thr（边界帧成立）
    return elapsed <= thr + _EPS_SEC
def _t_lt(elapsed: float, thr: float) -> bool:   # elapsed <  thr（边界帧不成立）
    return elapsed + _EPS_SEC < thr
# LONG_PRESS_SEC：旧 hold_frames>=23（LONG_PRESS_FRAMES+1）在第24帧触发=t23=23/30；
#   hold_sec 从 ARM 起累计（f2 得1个间隔，稳定帧从连续第2个稳定样本起累计），阈值 22/30 → 同在第24帧触发
LONG_PRESS_FRAMES_DEFAULT = 22
LONG_PRESS_SEC_DEFAULT    = LONG_PRESS_FRAMES_DEFAULT * FRAME_SEC   # 0.7333s（触发墙钟时刻=0.7667s）
# TAP_MAX_SEC：✨ 快速短按优先级。总时长 ≤ 此值 → 无论发生什么都按 TAP 处理（禁止变长按）。
#   旧 total_frames<=18：第18帧（=17个间隔）仍为 TAP → (18-1)*FRAME_SEC
TAP_MAX_FRAMES_DEFAULT    = 18
TAP_MAX_SEC_DEFAULT       = (TAP_MAX_FRAMES_DEFAULT - 1) * FRAME_SEC
# BEND_STABLE_THRESH：✨"弯曲保持稳定"阈值（默认值，会被 speed 覆盖）
BEND_STABLE_THRESH_DEFAULT  = 0.010
BEND_STABLE_MIN_HOLD_DEFAULT = 2
# MIN_BEND_HOLD 是【连续样本数】毛刺门：旧 ARM 当帧=1、再 1 帧（=1个间隔）>=2 进 HOLD，秒阈值=(N-1)*FRAME_SEC
MIN_BEND_HOLD_DEFAULT    = 1
MIN_HOLD_SEC_DEFAULT     = (MIN_BEND_HOLD_DEFAULT - 1) * FRAME_SEC
# COOLDOWN_SEC：用户明确要求保留改动（原14帧→4帧，解决连点第二下卡死）
COOLDOWN_FRAMES  = 4
COOLDOWN_SEC     = COOLDOWN_FRAMES * FRAME_SEC
# ⭐⭐⭐ 2026-09-21 18:12 真机日志铁证：MediaPipe 整手丢失(非掌心门控)可连续 0.5s+，
#   而手回来时手指始终弯着(raw .21/.22/.55)——"丢手超时=松手"假设物理上不成立。
#   最终语义：手势进行中(ARMED/HOLD/PRESSED)黑帧【无限期冻结】，松手与否只由手回来当帧的
#   手指物理状态裁决（回直三门：弯着→继续按；直了→LONG_UP/点击成立）。
#   仅保留超长硬兜底：连续黑帧超过此时长（手真离开再也不回来）才强制结束，防鼠标永久按住。
NONE_GESTURE_ABANDON_SEC = 10.0   # 硬兜底：连续黑帧超过此时长才强制释放/丢弃
# ⭐⭐⭐ 2026-09-21 18:57 真机日志铁证：误触链的真正起点是【丢手恢复期的过渡垃圾帧】——
#   手重新被 MediaPipe 捕获的头几帧会吐中间态关键点（日志 L43-46：黑帧2帧后 raw .535→.726
#   →.785→.788→.801 连"直"4帧，下一帧 L47 立刻又测到 .20 深弯）。旧逻辑当帧即释放 →
#   真实未松的按压被斩断，随后真实弯指动作被当成新点击 → 双击。
#   修复：黑帧结束后的 POST_NONE_CONFIRM_WINDOW 秒内，"松手"必须连续 POST_NONE_CONFIRM_FRAMES
#   帧都稳定过伸直门（任一帧重新弯下即清零）。真松手手指会一直直着，代价仅 ~0.1s 确认延迟；
#   重新捕获的瞬时假伸直会被紧接着的弯帧打断，不释放。
POST_NONE_CONFIRM_WINDOW = 0.35
POST_NONE_CONFIRM_FRAMES = 3
# ⭐ 双击判定（2026-09-10，用户要求：食指勾按有且只有「单击/双击」两种结果，严格互斥）：
#   第一次 Tap 松开后【不立即单击】，进入 WAIT_DBL 窗口：
#     · 窗口内完成第二次 Tap → DOUBLE_TAP（只发一次 doubleClick，挂起的单击作废 → 物理点击恰好 2 次）
#     · 窗口超时无第二次   → 补发 TAP_CLICK（单击延迟确认，杜绝双击同时出两个单击）
DOUBLE_TAP_WINDOW_SEC     = 0.38  # 趋势确认后，留给第二次完整 Tap（弯+回弹）的时间窗（秒）
DOUBLE_TAP_COOLDOWN_FRAMES = 6    # 双击后的冷却帧数（比单击4帧略长，让伸直基线稳一下）
DOUBLE_TAP_COOLDOWN_SEC    = DOUBLE_TAP_COOLDOWN_FRAMES * FRAME_SEC
# ⭐ 单击快速反应（2026-09-15 用户方案：别盲等，先探测"二次下弯趋势"）：
#   第一次 Tap 回弹后不再固定等 0.38s，而是先用 TREND_WAIT_SEC 短窗探测第二击【下弯趋势】——
#   低门限差分（完整按下门限的 TREND_DI_FRAC）+ bend_ratio 已离开伸直位 + 连续确认，
#   不要求完整点击成立：
#     · 短窗内出现趋势 → 判定双击意图，等待窗延长 DOUBLE_TAP_WINDOW_SEC 等第二击完成 → DOUBLE_TAP
#     · 短窗内无趋势   → 立即补发 TAP_CLICK（单击延迟从 0.38s 压到 ~0.10s），随后照常进 RESCUE_DBL
#   趋势偶尔漏判也无害：第二击仍落在 2.5s 补救窗内 → 原子 doubleClick 兜底。
TREND_WAIT_SEC       = 0.10   # 首次回弹 → 探测到二次下弯趋势 的最长时间（秒；30fps≈3帧观察窗）
TREND_DI_FRAC        = 0.40   # 趋势差分门限 = TAP_DOWN_DI_SUM 的比例（四成累计降幅+离开伸直位即算有趋势）
TREND_MAX_RATIO      = 0.95   # 且 bend_ratio 低于此值（确实离开伸直位，排除回弹尾段 EMA 锯齿）
TREND_CONFIRM_FRAMES = 2      # 连续满足的帧数（防单帧噪声误延长；0.10s窗内须在头2帧就出现趋势）
# ⭐ 双击补救窗口（2026-09-10 修复"慢节奏双击文件变成重命名"）：
#   真机实测体感勾指的"双击"两次间隔长达 1.5~3.3s（手要复位再勾），远超 Windows 双击窗 500ms。
#   关键认知：doubleClick() 是后台原子指令，内部两次点击间隔几十毫秒由代码控制，
#   与用户手指实际间隔无关——OS 必然识别为双击。所以只要识别出"双击意图"就直接发 doubleClick，
#   绝不能发两次独立 click（资源管理器对"选中后慢速再点文件名"的语义=重命名）。
#   流程：WAIT_DBL 0.38s 超时 → 立即补发 click#1（单击响应不受影响）→ 进入 RESCUE_DBL 补救窗：
#     窗内第二击 Tap 回弹 → 发 RESCUE_DBL_TAP 事件，mouse_controller 按【位置门控】处理：
#       · 落点与 click#1 同图标(≤RESCUE_DBL_SNAP_PX) → 原子 doubleClick()（OS 必判双击打开）
#       · 落点已在别的图标上(>门控) → 降级为普通 click（点别的东西不受影响）
#   窗口取 2.5s（用户指定 2026-09-10）：覆盖大多数自然节奏的慢双击；注意真机曾测到 3.3s 的
#   更慢节奏，超过 2.5s 的第二击会被当成两次独立单击（窗口期间光标移动/其他手势均不受影响）。
#   超时无第二击 = 真·单击，回 IDLE。
RESCUE_DBL_WINDOW_SEC     = 2.5   # 进入 RESCUE 状态 → 第二击弯曲开始 的最大间隔（秒）
# ⭐ 等待双击期间的 bend=None（丢帧/掌心不稳门控）处理：
#   2026-09-17 用户拍板【取消丢手宽限】（曾有 NONE_GRACE_SEC=1.0s 保留窗口的方案，已删除）：
#   WAIT_DBL/RESCUE_DBL 中一旦 bend=None → 立即结束序列（WAIT 立即补发挂起的 click#1 回 IDLE，
#   不再开 RESCUE；RESCUE 直接回 IDLE）。手完全消失时 screen_pt=None，事件在 mouse_controller 层丢弃。
# ⭐ 实验开关（2026-09-10）：双击判定归属
#   True  = 每次勾按【立即】click()，单击零延迟；快速连勾两次 = 两组 down/up，
#           由 Windows 双击时间（默认500ms/位置容差）自然识别为双击（和物理鼠标行为完全一致；
#           代价：双击时第一击也会送达应用，按钮类元素可能先吃到一次单击）。
#   False = 延迟确认：第一次Tap挂起 0.38s，窗口内第二次Tap只发 doubleClick（单/双击严格互斥，
#           代价：每次单击慢 0.38s）。
OS_DOUBLE_CLICK = False

# ⭐ 诊断开关（2026-09-10 双击排查）：True 时每次勾按打印状态机转移行 [TAP-DIAG]
#   ARM=起弯  TAP=点击发出  LONG=长按转换(双击被吞的头号嫌疑)  SPIKE=尖峰拒绝  NONE帧=掌心不稳被门控
#   2026-09-15：日常使用默认关闭（控制台刷屏会拖慢主循环、影响鼠标流畅度）；排查点击问题时临时改 True
TAP_DIAG = False

# ⭐ 真机测量（2026-09-17）：纯内存计数 + 周期摘要，【不参与任何判定逻辑】（零风险）。
#   每 TAP_STATS_PRINT_SEC 秒打印一行 [TAP-STAT]：帧率/None占比/ARM入口占比/尖峰拒绝/
#   点击段内None/首击/单击送达/双击/趋势确认/补救/长按/丢手杀等待次数/回弹与送达延迟分布。
#   用途：拿真机分布数据决定后续"墙钟化/丢手宽限/自适应平滑"的参数，不盲调。
TAP_STATS = False  # 用户要求减少刷屏，禁用 [TAP-STAT] 周期性统计打印（2026-10-17）
TAP_STATS_PRINT_SEC = 20.0   # 摘要打印周期（秒）；仅时间判断+一行 print，不拖主循环

# ⭐ 双指并拢右键手势（2026-09-16，食+中并拢同时勾按=鼠标右键）：
#   2026-09-17 v7（用户拍板：分阶段阈值 + 起势顺序门）：
#     真机深弯实测：起势帧 TIP≈0.5；勾到 HOLD 帧 DIP≈0.95、TIP≈1.33（深弯遮挡越靠指尖失真越大）。
#     → 两排阈值分阶段使用：
#       · 起势（IDLE，手指尚高、无遮挡）：TIP ≤ TIP_ENTRY_R(0.80) —— 指尖先并拢才准起势；
#       · 手势中（ARMED/HOLD/PRESSED，深弯遮挡）：改看 DIP ≤ DIP_HOLD_R(1.30)，
#         TIP 再炸也不解散（用户真机值 DIP 0.947 放行）。
#     → 起势顺序门（修"单击灵敏度下降"）：
#       单指点击时中指蜷在掌心，食指勾【到底】时指尖才扫近中指尖（TIP 瞬间 ≤0.8），
#       若此时即起弯会 freeze_left + cancel_in_progress 掐断正在识别的左键 → 单击被吞。
#       真右键 = 先举着并拢手（高 bend）持续 ≥3 帧，再勾；单指点击 = 先弯到底才碰巧并拢。
#       故要求：bend ≥ ENTRY_BEND(0.55，不要求伸直) 的 TIP 并拢帧连续 ≥ ENTRY_FRAMES(3) 帧，
#       才允许进 ARMED。资格取得后下弯帧不再要求 bend 线；指尖张开立即清零重来。
#     ⭐ 左键参数从未修改：单击变迟钝的唯一通道是右指检测器误 ARM 后的 freeze/cancel，本门修复。
#     ⭐ 2026-10-04【上面的 freeze/cancel 机制已废除】：四份真机日志实锤，用户自然悬停两尖本就
#       并拢（TIP 0.27~0.49，单会话 27 次左键被"外部抢占取消"、双击第二击被掐死），
#       现改为右检测器 ARM/HOLD 全程不碰左键，仅右键送达帧互斥，见 hands_tasks 仲裁块。
# ⭐⭐ 2026-10-03 真机日志（[2F-ENTRY]/[2F-SEG] 埋点）实锤误右键链：
#   单指勾按时食指尖扫向掌心、物理扫过中尖附近距离带，TIP 瞬间 0.15~0.43；
#   另有自然悬停姿势两尖本就相距 0.63~0.78（=5~6cm，根本不是"并拢"）。
#   旧门 0.80（≤6.4~7.2cm）两类全部白拿资格，9 次资格 / 3 次误右键，全是"入口=绝对"、
#   中指摆幅 0.16~0.28（没动）vs 食指摆幅 1.37~1.80（勾按）= 单指点击误判。
#   真"并拢"=双指尖相触/相邻，3D 距/掌宽约 0.10~0.30；门收紧到 0.50（≈4cm@掌宽8cm）。
TWO_FINGER_TIP_ENTRY_R = 0.50   # 起势：TIP排/米制掌宽 ≤ 此值（旧 0.80 真机实锤过宽）
TWO_FINGER_DIP_HOLD_R = 1.30    # 手势中：DIP排/米制掌宽 ≤ 此值（深弯 TIP 失真，实测 DIP≤0.95）
TWO_FINGER_ENTRY_BEND = 0.55    # 起势资格帧的最低合成 bend（0.55=明显尚未深弯，微弯悬停可过）
TWO_FINGER_ENTRY_FRAMES = 3     # 旧帧阈值（仅留作换算来源）：高 bend 指尖并拢连续 3 帧才允许 ARMED
# 墙钟：旧第3个连续并拢帧（=2个间隔）当帧资格满并可起 ARM → (3-1)*FRAME_SEC
TWO_FINGER_ENTRY_SEC = (TWO_FINGER_ENTRY_FRAMES - 1) * FRAME_SEC
#   双指右键不做双击语义：弯曲/回弹判据【逐条照搬左键 FingerTapClickDetector 模板】（2026-09-17 重构）：
#     · 唯一输入是"双指单元合成 bend"（四关节中点代入与左键相同的关节角公式）→ 单条 r 序列；
#       下游 down/up 差分、尖峰确认、短按窗、长按帧全部直接跑左键同一套 resolve_tap_params 公式；
#     · 松开 = 回弹变化量 Σ↑达标 OR r 回伸直线（与左键同为"变化量优先"）；
#     · 右键额外门控：起势 TIP 并拢 + 先并拢顺序门；手势中 DIP 连续失效 2 帧才作废（去抖）。
#       同步性由四排物理间距直接保证，不需要 min(ri,rm) 软地板 / 落后指等待。
#   ARMED/HOLD 中 bend=None（丢手/掌心不稳门控）的连续帧上限：30fps≈0.4s，超过作废。
#   墙钟：None 帧也占墙钟（旧第13个 None 帧作废=13/30），阈值 12*FRAME_SEC=0.40s
TWO_FINGER_NONE_GRACE_FRAMES = 12
TWO_FINGER_NONE_GRACE_SEC = TWO_FINGER_NONE_GRACE_FRAMES * FRAME_SEC
# 手势进行中 together 连续失效多少帧才判定为"张开/不同步作废"（1 帧抖动容忍）
TWO_FINGER_GAP_BREAK_FRAMES = 2

# ⭐⭐ 2026-10-04 资格确认后的"开弯过渡宽限"（真机 18 次资格仅 1 次起 ARM 的根因修复）：
#   资格帧用食/中【原始独立弯度】直门（都 ≥0.80），而勾按启动时两指开弯常差 1~2 个相机帧：
#   先弯的一指率先 <0.80，但四关节【中点】合成 bend 几何上滞后，仍 ≥0.55，旧 L1776 分支当场
#   按"并拢时手指已弯"作废资格 → 等中点 bend 跌破 0.55 时已无资格，整次右键被当单指点击拒绝。
#   仅食/中严格同帧开弯（罕见）才能一次通过。
#   规则：资格【已确认】后，连续"仍 TIP 并拢但任一指已弯"的高 bend 帧头 TWO_FINGER_ENTRY_GRACE_FRAMES
#   个 → 保留资格（bend 仍不喂状态机），等中点 bend 跌破 0.55 的勾弯帧正常消费；用帧计数而非
#   墙钟窗——用户常并拢停顿 0.3~1s 再勾，墙钟窗会把正常起势也杀掉；连续第 3 个弯指帧中点 bend
#   还没跌破 0.55 = 一直保持弯指姿势而非勾按动作，照旧作废。指尖【张开】帧任何时候立即作废。
TWO_FINGER_ENTRY_GRACE_FRAMES = 2

# ⭐⭐ 2026-10-03 双指独立弯度门（根治"单指勾按扫过中尖 → 合成 bend 骤降 → 误右键"）：
#   hands_tasks 用与左键完全同公式（关节角 0.20弯~0.88直）分别计算食/中指独立弯度
#   idx_bend/mid_bend，随 update 喂入：
#   ① 资格帧双指必须都近直（≥RF_ENTRY_FINGER_R）：并拢只可能来自"伸直双指主动并拢"，
#      勾指扫过带发生时食指已经弯着（<0.80）→ 不累计资格；
#   ② 【已废弃 2026-10-04】ARM 后"双指都必须弯"的同弯门：10-04 真机反证——
#      用户刻意真右键中弯摆幅仅 0.02~0.07（放松中指天生弯 0.31~0.45，双指勾按时
#      中指几何弯度本就几乎不动），该门放过所有误右键、反杀真右键；改走指尖距持续门
#      TWO_FINGER_TIP_HOLD_R（见下）。
RF_ENTRY_FINGER_R = 0.80    # 资格帧：食/中弯度都 ≥ 此值（近伸直）才许累计（保留，仅挡勾指途中才并拢）

# ⭐⭐ 2026-10-04 手势中指尖距持续门（替代 10-03 被真机数据证伪的双指同弯门）：
#   旧 v7 架构 ARM 之后只看 DIP 关节排（≤1.30），但 DIP 在手指中部，单指勾按时它也不动
#   （真机单指/双指 DIP 都在 0.23~0.33，无区分力），等于 ARM 后指尖距信号彻底丢失。
#   现 ARM/手势中持续校验两【指尖】距离：连续 TWO_FINGER_GAP_BREAK_FRAMES 帧
#   TIP排/掌宽 > TWO_FINGER_TIP_HOLD_R 才解散（沿用 2 帧去抖）。
#   阈值依据：10-04 全部右键样本（含刻意）段内 TIP max ≤ 0.42，取 0.60 留 0.18 余量；
#   观测模式（RF_OBSERVE_ONLY）下先量"真右键段内 TIP max"与"左键段 TIP"分布再决定启用。
TWO_FINGER_TIP_HOLD_R = 0.60   # 手势中：两指尖距/米制掌宽须持续 ≤ 此值

# ⭐⭐ 2026-10-04 观测模式总开关（真机校准，不改判定架构）：
#   True 时右检测器状态机照常跑（[2F-ENTRY]/[2F-SEG] 埋点照常），但
#     ① 不冻结/抢占左键（左键不再被右键 ARM 杀死 → 恢复"点得了"）；
#     ② 右键事件只打印 [2F-OBS] 后吞掉，绝不送系统（→ 恢复"不会误点"）。
#   目的：采集"正常单指点击 vs 刻意双指右键"两群的指尖距/中弯摆幅分布，
#   用用户真机数据决定最终门限后再改 False 启用。
# ⭐ 2026-10-04 22:32 真机校准会话结论（日志 L2172-2338）：刻意双指右键 7/7 全部
#   GRACE→ARM→END 送达，段内 TIP 0.14~0.35、食/中弯深勾 0.20~0.45；同期左键 11 次
#   ARM/单击5/双击2 组零误判为右键。校准通过，正式启用（False）。
RF_OBSERVE_ONLY = False


# ⭐ 双指右键手势诊断开关（2026-09-16 识别困难排查）：True 时打印
#   · [2F-DIAG-HB] 每 30 有效帧一行心跳：gap 并拢度/together/ri/rm/r/阈值/状态
#   · [2F-DIAG]    关键事件：起弯被拦原因、ARMED 确认/拒绝、HOLD 释放/作废、长按静默、丢帧宽限、触发
#   识别手感调好后可改 False（与 TAP_DIAG 同样的临时排查用途）。
# 2026-09-21：手感已确认，默认关闭（心跳每30帧刷屏）；排查右键问题时临时改 True。
TWO_FINGER_DIAG = False

# ⭐ 2026-08-22 差分点击新增常量
# 差分窗口【墙钟化】：hist 存 (t, bend_ratio)，取最近 DI_WINDOW_SEC 秒（旧"5帧"=5个帧间隔=167ms，
# 刚好是"按下去"的动作时长）。30fps 取到的样本数与旧版逐点一致；60fps=10点/20fps=3点，物理时长恒定。
# deque 容量按 60fps 上限放宽；None 帧仍整窗清空（旧行为）。
DI_WINDOW = 5                 # 旧帧窗口（speed=0.6 档；换算来源），实际秒值在 resolve_tap_params 按档给出
# ⭐ 2026-09-19 真机调参（只动墙钟参数，不动状态机）：
#   当日日志 TAP-STAT 实测主循环仅 7~9 更新/秒（每 20s 窗口帧计数 126/169/179/139/145），
#   有效帧间隔 0.11~0.16s；旧窗 speed0.8=4/30=0.133s 恰好≈1 个慢间隔，快速勾按在【回弹帧】
#   窗内只剩"下弯点→回弹点"一对上升差分，Σ下=0.000 → ARMED 尖峰拒绝（日志 8/14 次），
#   用户被迫慢速保持 ~1s 弯直到后续帧 r 仍低才确认。
#   加宽 5 个 30fps 间隔(0.167s)：慢环下窗口 0.30s 可跨 2 个间隔，把"待机点+下弯点+回弹点"
#   三点同时纳入（Σ下≈0.4，轻松过 35% 保持门限）；30fps 由 5 点增至 10 点，60fps 19 点。
#   down/up 门限与绝对门控不变，松开判定为 OR（更大 Σ上 只会更易识别回弹，不延迟）。
DI_WINDOW_SLOW_PAD_SEC = 5 * FRAME_SEC
DI_HIST_MAXLEN = 32           # 加宽后 60fps 窗内最多 ~25 点（speed0 档 0.40s），32 留余量
# ⭐ 2026-08-24 适配 v2 弯曲度度量（关节角度融合版，Δ 比旧 dist_bend 略小 15~20%）：
#   差分门限整体 ×0.70 左右，保证灵敏度回到或略高于纯旧度量的水平
TAP_DOWN_DI_SUM = 0.10        # 按下门限：ΣΔbend_ratio ≤ -0.10 → 食指真的主动弯了（bend_ratio 下降）
#   （v2 关节角度 手指从伸直→90°弯，bend_ratio 变化≈0.85→0.30=-0.55 → 取 0.10 半程即可触发，5帧内必须完成）
TAP_UP_DI_SUM   = 0.07        # 松开门限：ΣΔbend_ratio ≥ +0.07 → 食指真的主动回伸（bend_ratio 上升）
# ⭐⭐ 2026-09-21 长按/双击隔离（用户拍板）：松开不能只看过没过一条低绝对线 / 窗内差分，
#   必须验证「真的回直」——否则按住中途手指微松（仍明显弯着）会被当成第一次 Tap 回弹，
#   再压弯、再微松即在短按窗内凑成 DOUBLE_TAP（长按中直接触发双击的根因）。
#   两个门 AND，缺一不可：
#     ① 回直变化量门：从本段弯曲【谷底】算起的 bend_ratio 回升幅度 ≥ RECOVER_RISE_MIN
#        （真机标定：真实点击 谷底≈0.25~0.43 → 回直 0.95，回升 0.52~0.70；
#          长按中微松 谷底0.25 → 0.55~0.70，回升仅 0.30~0.45 且手指仍弯着）
#     ② 伸直位确认门：当前 bend_ratio ≥ RECOVER_RATIO_MIN（基本回到伸直；0.70/0.76 旧线太低已废）
#   长按中手指没回直 → 两个门必不过 → 不产生 Tap、不进 WAIT_DBL → 状态机层面不可能发双击。
RECOVER_RISE_MIN  = 0.30      # 相对本段谷底的最小回升幅度（ratio 尺度）
RECOVER_RATIO_MIN = 0.88      # 当前 bend_ratio 必须到此线（接近伸直）才承认回弹/松手
# ⭐⭐ 2026-09-21 真机日志(tap_gate.log 17:05)三连证：只靠 EMA 平滑值判回直会在【回弹运动途中】松手——
#   误触帧 raw=0.7705（伸直≈0.81，手指挥弹中未到位）EMA 后 ratio=0.881 险过 0.88 门(仅多0.001)。
#   补第三门：原始测量 raw bend（不经 EMA）也必须回到基线 98% 以上，三个门 AND。
#   标定：真回直 raw=0.806~0.816；途中误触 0.7704/0.7705。基线随伸直帧快速学习，长期稳定。
RAW_RECOVER_FRAC  = 0.98
# 点击段（Tap/双击任一击）谷底必须真的弯到过这条线以下，否则算半弯抖动、静默取消不发点击。
#   标定：正常点击谷底 ratio=0.25~0.45；日志误触段谷底 0.69（raw 仅到 0.52，几乎没弯）。
#   仅用于 HOLD→点击分支；长按(PRESSED)释放不受此限（能进长按说明已弯得够久）。
TAP_MIN_BEND_RATIO = 0.66
# ⭐⭐ 2026-10-15 长按 3D 正向证据门（架构变更：单手握拳不再是全局取消手势）
#   旧"握拳全局 guard"在遮挡场景致命（world 缺失宁拦、TIP 推断点乱跳→点击被整段杀光、
#   光标冻结），已整体拆除。防"握拳保持误转长按"的职责收敛到【长按转换这一个动作边界】：
#     · 只门 BEND_HOLD→PRESSED(LONG_DOWN)；ARM/单击/双击链路完全不读此门，遮挡绝不杀点击；
#     · 正向证据 = 食指近节骨(MCP5→PIP6) 与掌轴(wrist0→MCP9) 的 3D world 夹角：
#       勾按/长按食指作业近节必须沿掌轴前伸，握拳（松/紧）近节必折向掌面。
#   真机标定 2026-10-14（22:15 会话，点击/勾按段 n=26+ 食角 27~51°，握拳段 56~80°，
#   间隙 51/56）→ 阈值 53° 取中；只用 MCP/PIP 两个遮挡下仍稳定的点，与 TIP 可信度无关。
#   阈值为单轮小样本标定，[TAP-SEG]/[LONG3D] 心跳持续观测，有新分布再调。
#   world/角度缺失：长按 fail-closed（不转长按、松手静默结束）；实测该路径计数≈0，
#   且单击不受影响，风险远小于旧全局 guard 的宁拦勿漏。
LONG_GATE_MCP_IDX_DEG = 53.0
# ARMED 第二帧"尖峰拒绝"的深弯豁免：起弯帧谷底已到此线（实打实弯到过半），第二帧即便弹直
#   也是【最短合法点击】而不是噪声（真机日志：raw 0.20 弯1帧→0.81 真直，旧逻辑整条吞掉，
#   反让后面没松直的抖动顶替成第二击）。
DEEP_ARM_KEEP_RATIO = 0.55
MIN_ABS_BEND_RATIO_FOR_DELTA = 0.75  # 差分触发后仍必须满足 bend_ratio ≤ 这个兜底（防止"差分刚好到了、但其实没弯"）
                                      # ✅ v3 统一弯曲度量（DIP=0.40 主力）后校准：
                                      #   中等点击(DIP 弯至 120°=60°折) → bratio≈0.70 ≤ 0.75 ✅（能过）
                                      #   移动微角度 ±2° → bratio 变化≈0.004 → DI=0.004 远不够，不会误触
                                      #   深点击(DIP 弯至 30°=150°折) → bratio≈0.45 ≤ 0.75 ✅ 肯定过

# 平滑 / 基线学习
# ⭐ 2026-10-02 bend EMA 墙钟化：α = 1 - exp(-dt/τ)，物理时间常数 τ = 50ms。
#   旧固定 α=0.79(默认档) 只在标称 30fps 等效（≈τ21ms）；真机主循环曾测到 7~9fps，
#   一帧混入 79% 原始值，帧率越低抖动进平滑越多。墙钟化后 30/10/120fps 物理行为一致：
#     30fps dt=0.0333 → α≈0.487；10fps dt=0.100 → α≈0.865；120fps dt=0.0083 → α≈0.154
#   τ(50ms) 远小于 DI 窗(默认0.33s)与点击动作时长，阶跃在窗内可完成约 6.7τ ≈ 99.9% 跟随，
#   真点击的窗内 Σ负差分几乎不缩水，只压帧间高频抖动。固定物理常数，不再随 speed 变化。
#   ⚠ EMA 必须吃【真实帧间隔】：不能用计时路径的 MAX_DT_SEC(2帧) 钳幅值，否则稳定 10fps
#   （0.1s/帧是正常帧率不是卡顿）每帧只按 0.0667s 衰减 → 低帧率下被过度平滑、帧率不变性失效。
#   EMA 仅对两类异常间隔回退：dt<=0（同刻帧）→ 标称 FRAME_SEC；dt>BEND_EMA_MAX_DT_SEC(0.5s)
#   （真卡顿/跳帧）→ 钳 0.5s（此时 α≈0.99995，已近似全取新值，钳与不钳几乎无差，只为守界）。
BEND_EMA_TAU_SEC = 0.05
BEND_EMA_MAX_DT_SEC = 0.5
BASELINE_ALPHA_IDLE  = 0.025   # IDLE下 伸直基线 慢速漂移 速率
BASELINE_ALPHA_FAST  = 0.40    # 刚回到伸直 且确定没 tap 时，快速对齐基线


def _bend_ema_alpha(dt: float, tau: float = BEND_EMA_TAU_SEC) -> float:
    """bend EMA 墙钟化系数：α = 1 - exp(-dt/τ)。dt 必须是【真实帧间隔】（非计时钳幅值）。
    异常回退：dt<=0（同刻帧/非正数/NaN）按标称 FRAME_SEC；dt>BEND_EMA_MAX_DT_SEC(0.5s)
    （真卡顿/跳帧）钳到 0.5s。稳定低帧率（如 10fps 的 0.1s）原样进指数，保持帧率不变性。
    """
    if not (dt > 0.0):
        dt = FRAME_SEC
    elif dt > BEND_EMA_MAX_DT_SEC:
        dt = BEND_EMA_MAX_DT_SEC
    return 1.0 - math.exp(-float(dt) / float(tau))


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
      0 = 最稳：MIN_HOLD=3, TAP_MAX=14, STABLE=0.020
      0.5= 默认快：MIN_HOLD=2, TAP_MAX=21, STABLE=0.0125
      1 = 极速：MIN_HOLD=1, TAP_MAX=28, STABLE=0.005
      （bend EMA 已墙钟化为固定 τ=BEND_EMA_TAU_SEC(50ms)，不再随 speed 变化）

    ⭐ 2026-08-22 差分点击新增：down/up 差分和 随 hardness/speed 联动
      · hardness↑ = DOWN_DI↑（更严=必须更大幅度弯下来才触发），UP_DI 同步
      · speed↑ = DI_WINDOW↓（更灵=窗口更短，更短时间内的变化就能触发）
    """
    if hardness is None: hardness = 0.6
    if speed    is None: speed    = 0.6
    hardness = 0.0 if hardness < 0.0 else (1.0 if hardness > 1.0 else float(hardness))
    speed    = 0.0 if speed    < 0.0 else (1.0 if speed    > 1.0 else float(speed))

    # ✨ ENTER_BEND_RATIO：v3 统一度量校准档 0.65~0.73（旧 0.62~0.70 略提 0.03 = 中等点击能过，移动不误触）
    #    h=0.6(默认) → enter=0.698 → enter_t=baseline*0.7 ≈ 0.85*0.7=0.595
    #    （左键起弯只看变化量，enter 线仅用于松开兜底/双指右键；MIN_BEND_DROP 仅右键在用）
    enter = 0.65 + hardness * 0.08           # 0.65 ~ 0.73
    drop  = 0.10 + hardness * 0.16           # 0.10 ~ 0.26（门槛B相对降幅，保持稳定）
    # EXIT = ENTER + 0.06 形成滞回，ENTER 整体下移 EXIT 同步下移（更早识别回弹）
    exit_r = min(0.92, enter + 0.06)

    # bend EMA 平滑系数已墙钟化（τ=50ms 固定，见 _bend_ema_alpha），不再由 speed 映射固定 α
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
        BEND_EMA_TAU_SEC   = BEND_EMA_TAU_SEC,
        # ⭐ 墙钟化（2026-09-18）：物理时长阈值全部秒；首帧计1的帧阈值 N → (N-1)*FRAME_SEC
        MIN_HOLD_SEC       = (min_hold - 1) * FRAME_SEC,   # 旧 MIN_BEND_HOLD 帧（ARM 后再过 N-1 间隔进 HOLD）
        TAP_MAX_SEC        = (tap_max - 1) * FRAME_SEC,    # 旧 TAP_MAX_FRAMES（第 N 帧 total=N 占 N-1 间隔）
        LONG_PRESS_SEC     = long_press * FRAME_SEC,       # 旧 LONG_PRESS_FRAMES+1 触发，hold_sec 阈值=22/30
        BEND_STABLE_THRESH = round(stable_thr, 4),
        BEND_STABLE_MIN_HOLD = stable_min,   # 【保留帧计数】连续"稳定样本"数（毛刺去抖，非物理时长）
        COOLDOWN_SEC       = COOLDOWN_SEC,
        # ⭐ 差分点击新增参数
        # 旧 DI_WINDOW 帧窗口（win+1 点跨 win 个间隔）+ 慢环加宽（见 DI_WINDOW_SLOW_PAD_SEC 注释，
        # 真机主循环 7~9fps 时 0.133s 窗在回弹帧把下弯点老化→Σ下=0→尖峰拒，2026-09-19）
        DI_WINDOW_SEC      = di_win * FRAME_SEC + DI_WINDOW_SLOW_PAD_SEC,
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
    RESCUE_DBL  = 7  # 单击已补发(click#1已送达)，补救窗内等第二击：窗内Tap回弹立即发单击(click#2追加双击)

class E:
    NONE       = 0
    TAP_CLICK  = 1   # 一次完整单击：mouseDown -> mouseUp (双击窗口超时后补发)
    LONG_DOWN  = 2   # 长按开始：mouseDown，之后每帧 DRAG_MOVE
    LONG_UP    = 3   # 长按结束：mouseUp
    DRAG_MOVE  = 4   # 长按中持续移动
    DOUBLE_TAP = 5   # 双击窗口内完成第二次 Tap（原子 doubleClick，单击不补发）
    RESCUE_DBL_TAP = 6  # 补救窗口内第二击 Tap（mouse_controller 按位置门控：同图标→原子doubleClick打开，异处→普通click）
    RIGHT_TAP    = 7   # 双指并拢（食+中）同时弯曲回弹 → 鼠标右键单击（无双击语义，每次手势恰好1次右键）


class FingerTapClickDetector:
    def __init__(self, hardness: float | None = None, speed: float | None = None):
        # 解析参数 → 所有阈值（文件默认值只是 fallback，CLI 参数会覆盖）
        self._p = resolve_tap_params(hardness=hardness, speed=speed)
        self.state = _S.IDLE
        self.event_this_frame = E.NONE

        self._smoothed_bend = None
        self._prev_smoothed_bend = None  # 上一帧 bend，判断"是否还在继续弯曲"
        self._baseline = None        # 伸直状态的 bend "高值"（用户动作前后都以此为参考）

        # ⭐ 墙钟化（2026-09-18）：物理累计一律秒（monotonic）；_stable_frames 仍是连续样本计数
        self._hold_sec = 0.0             # ARM 起累计的"有效弯曲保持时长"（ARMED 每有效帧+dt；HOLD 仅稳定帧+dt）
        self._total_sec = 0.0            # ARM 起有效帧总时长（不管稳不稳定，用于 TAP_MAX_SEC 快速短按判定）
        self._stable_frames = 0          # 连续"弯曲深度稳定"的【样本帧数】（毛刺门，按帧不按时长）
        self._cooldown_until = 0.0       # COOLDOWN 结束时刻（monotonic）；每次 update（含 None）都查
        self._last_valid_t = None        # 上一有效帧时刻；None 帧不推进，恢复首帧 dt=FRAME_SEC（旧 None 透明语义）
        self._none_run_start = None      # 连续黑帧首帧时刻（墙钟）；有效帧清零。短黑帧冻结、超时才释放
        self._post_none_deadline = 0.0   # 黑帧恢复后的"松手确认窗"截止时刻；窗内松手须连续N帧伸直
        self._straight_confirm = 0       # 恢复窗内连续过伸直门的帧数；重新弯下即清零

        # ⭐ 2026-08-22 差分点击新增状态
        #   _bend_ratio_hist：连续最近窗内的 (时刻, bend_ratio)，maxlen 按 60fps 窗放宽到 16
        self._bend_ratio_hist: deque = deque(maxlen=DI_HIST_MAXLEN)
        #   标记当前 ARMED→HOLD→PRESSED 这段是「差分进入」还是「绝对兜底进入」，影响 EXIT 判据优先级
        self._armed_by_delta = False
        #   本帧计算出的差分和（给 metrics/UI 展示用）
        self._di_sum_neg_now = 0.0  # ΣΔ<0 的总和 = 按下去的累积（负值；触发时 ≤ -DOWN_DI）
        self._di_sum_pos_now = 0.0  # ΣΔ>0 的总和 = 抬起来的累积（正值；触发时 ≥ +UP_DI）
        # ⭐⭐ 2026-09-21 段内弯曲谷底（bend_ratio）：ARMED 当帧起记录，HOLD/PRESSED 每帧取 min；
        #   松开必须从此谷底回升 RECOVER_RISE_MIN 且当前 ≥ RECOVER_RATIO_MIN（真回直才松手）。
        #   None = 当前不在弯曲段（IDLE/WAIT_DBL/RESCUE_DBL/COOLDOWN）。
        self._seg_min_ratio: float | None = None

        # ⭐ 双击判定状态
        self._double_candidate = False  # 当前这段弯曲是否始于双击窗口内（=第二次Tap候选）
        self._tap_pending = False       # WAIT_DBL 中是否挂着一个待确认的单击
        self._wait_deadline = 0.0       # WAIT_DBL 超时时刻（monotonic；趋势确认后=完整第二击截止）
        # ⭐ 二次下弯趋势探测（首次 Tap 回弹后的短窗，无趋势立即补发单击）
        self._wait_trend_seen = False   # 本等待窗内是否已确认二次下弯趋势（确认后延长等完整第二击）
        self._wait_trend_frames = 0     # 连续满足趋势条件的【样本帧数】（毛刺门，保留帧计数）
        self._wait_trend_deadline = 0.0 # 趋势探测短窗截止时刻（monotonic）
        # ⭐ 双击补救状态（单击补发后的第二击追击）
        self._rescue_candidate = False  # 当前这段弯曲是否始于补救窗口内（=双击第二击，回弹立即单击）
        self._rescue_deadline = 0.0     # RESCUE_DBL 超时时刻（monotonic）
        self._cooldown_next = _S.IDLE   # COOLDOWN 结束后转入的状态（补发单击后转 RESCUE_DBL）
        # ⭐ TAP-DIAG 诊断
        self._dbg_arm_t = None          # 最近一次 ARMED 时刻
        self._dbg_none_in_tap = 0       # 当前 ARM/HOLD 段内 bend=None（掌心不稳门控）帧数
        # ⭐ 2026-10-15 长按 3D 正向门运行态（每个弯曲段在 ARM 时复位）
        self._long_3d_blocked = False   # 本段长按转换是否【曾】被 3D 门拦截（松手→静默结束；一旦成功 LONG_DOWN 即清）
        self._long_3d_diag_t = 0.0      # [LONG3D] 拦截心跳节流（墙钟）
        # ⭐ 2026-10-03 段生命周期埋点（纯观测）：每个弯曲段无论成败都留一条记录，
        #   由 hands_tasks 每帧 take_seg_logs() 取走附坐标/分区打印 [TAP-SEG]。
        #   解决"标题栏重复尝试、慢送达但失败过程无日志"的观测盲区。
        self.armed_this_frame = False   # 本帧是否进入 ARMED（hands_tasks 据此记 ARM 时刻坐标）
        self._seg_logs: deque = deque(maxlen=30)

        # ⭐ 真机测量（2026-09-17）：纯观测计数，reset() 不清（保留整段会话的统计），绝不参与判定
        self._st = {
            "frames": 0, "none": 0,             # update 调用总数 / bend=None 帧数
            "arm_delta": 0, "arm_abs": 0,       # 起弯次数（差分入口 / 绝对入口）
            "spike": 0,                         # ARMED 第2帧尖峰拒绝次数
            "shallow": 0,                       # 回弹时发现整段弯得太浅（谷底>0.66）→静默取消次数
            "none_ah_frames": 0, "none_ah_seg": 0,  # 点击段(ARM/HOLD)内 None 总帧 / 含 None 的段数
            "first_tap": 0,                     # 首击回弹进 WAIT_DBL 次数
            "click": 0,                         # TAP_CLICK 实际发出次数（含超时补发/丢手立即补发）
            "dbl": 0,                           # DOUBLE_TAP（窗口内第二击原子双击）
            "trend": 0,                         # WAIT_DBL 中二次下弯趋势确认次数
            "rescue_tap": 0,                    # RESCUE 补救窗内第二击次数
            "long": 0,                          # LONG_DOWN 次数
            "long_shallow": 0,                  # 保持够长按时长但谷底>0.66（半弯）→ 不转长按次数
            "long_3d_block_seg": 0,             # 被 3D 正向门拦截的弯曲段数（一段计一次）
            "long_3d_block_frames": 0,          # 3D 门拦截的 ripe 帧数（含同段多帧）
            "wait_none_cancel": 0,              # WAIT_DBL 遇 None 立即结束次数（原宽限方案已删）
            "rescue_none_cancel": 0,            # RESCUE_DBL 遇 None 立即结束次数
        }
        self._gest_ms: deque = deque(maxlen=200)   # ARM→回弹识别耗时（检测链响应）
        self._click_ms: deque = deque(maxlen=200)  # ARM→TAP_CLICK 实际送达（含双击等待，用户感知延迟）
        self._stats_last_t = 0.0

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
        self._hold_sec = 0.0
        self._total_sec = 0.0
        self._stable_frames = 0
        self._cooldown_until = 0.0
        self._last_valid_t = None
        self._none_run_start = None
        self._post_none_deadline = 0.0
        self._straight_confirm = 0
        # 差分点击状态也同步清空
        self._bend_ratio_hist.clear()
        self._armed_by_delta = False
        self._seg_min_ratio = None
        self._di_sum_neg_now = 0.0
        self._di_sum_pos_now = 0.0
        # 长按 3D 正向门运行态同步清空
        self._long_3d_blocked = False
        self._long_3d_diag_t = 0.0
        # 双击判定状态同步清空（挂起的单击作废）
        self._double_candidate = False
        self._tap_pending = False
        self._wait_deadline = 0.0
        self._wait_trend_seen = False
        self._wait_trend_frames = 0
        self._wait_trend_deadline = 0.0
        # 补救状态同步清空
        self._rescue_candidate = False
        self._rescue_deadline = 0.0
        self._cooldown_next = _S.IDLE
        # 计数器保留（让 UI 统计不丢），若想清空可自行注释放开：
        # 注：_st/_gest_ms/_click_ms 真机测量计数同样不在 reset() 清空（整段会话累计观测）。
        # self.tap_count = 0
        # self.long_count = 0
        # self.double_count = 0
        # self.last_tap_time = None
        # self.last_long_time = None

    # ---------------------- 双指右键抢占：静默取消"正在进行的弯曲段"，不发任何事件 ----------------------
    def cancel_in_progress(self) -> None:
        """双指并拢右键手势确认占用本帧手势时调用（hands_tasks 仲裁）：
        若食指已先 1~2 帧起弯进入 ARMED/HOLD，直接丢弃这段、回 IDLE，绝不补发单击，
        保证"双指右键"和"食指左键"物理互斥（不会同一个动作左右键一起响）。
        注意：只清进行中的弯曲段；WAIT_DBL 里挂起的"上一次已完成 Tap"不受影响（那是早先独立动作）。
        """
        if self.state in (_S.BEND_ARMED, _S.BEND_HOLD):
            self._emit_seg("外部抢占取消", _CLOCK())
            self.state = _S.IDLE
            self.event_this_frame = E.NONE
            self._hold_sec = 0.0
            self._total_sec = 0.0
            self._stable_frames = 0
            self._last_valid_t = None
            self._armed_by_delta = False
            self._seg_min_ratio = None
            self._di_sum_neg_now = 0.0
            self._di_sum_pos_now = 0.0
            self._long_3d_blocked = False
            # 差分队列清掉这段弯曲残留，避免手势结束恢复喂帧时被旧尾巴误触发
            self._bend_ratio_hist.clear()

    def force_long_up(self) -> int:
        """外部仲裁【明确作废当前长按】时调用：看见姿态已变、物理上不可能仍在按住，
        立即抬键（避免外部模块需要废弃长按时长按卡键 10s）。2026-10-15 起单手握拳
        全局 guard 已拆除，握拳不再调用本方法；保留给未来双手和弦等外部仲裁复用。
        与 bend=None 的丢手语义不同——None=没看见手→冻结等待；本方法=看见手且判定
        姿态已变→立即抬键。
        Returns: E.LONG_UP（调用方把它作为本帧事件）；非 PRESSED 态返回 E.NONE（无动作）。
        """
        if self.state != _S.PRESSED:
            return E.NONE
        self.event_this_frame = E.LONG_UP
        self._emit_seg("握拳强抬", _CLOCK())
        self._bend_ratio_hist.clear()
        self._di_sum_neg_now = 0.0
        self._di_sum_pos_now = 0.0
        self._none_run_start = None
        self._st["long"] += 1
        self._enter_cooldown()
        return E.LONG_UP

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

    # ---------------------- 内部：一阶差分计算（时间窗） ----------------------
    def _compute_di_sums(self, bend_ratio_now: float, now: float):
        """把当前 (时刻, bend_ratio) 入队，然后计算最近 DI_WINDOW_SEC 秒：
          - di_sum_neg = 窗内一阶差分中 负值的总和（向下按：bend_ratio↓ → Δ<0，累加其绝对值）
            注意：bend_ratio 越大=越伸直；所以"食指弯下去" = bend_ratio 变小 → Δ=br_now - br_prev < 0。
            为了 UI 展示和阈值判断直观，di_sum_neg 存「负值累积的绝对值」，即一个正数。
          - di_sum_pos = 窗内一阶差分中 正值的总和（向上抬：bend_ratio↑ → Δ>0）
        旧帧版取最近 win+1 个样本（跨 win 个帧间隔）；30fps 下 t>=now-win*FRAME_SEC 恰好取到同样 6 个点。
        返回：(di_sum_neg, di_sum_pos)，都是非负实数。
        """
        self._bend_ratio_hist.append((now, float(bend_ratio_now)))
        win_sec = self._p["DI_WINDOW_SEC"]
        if len(self._bend_ratio_hist) < 2:
            self._di_sum_neg_now = 0.0
            self._di_sum_pos_now = 0.0
            return 0.0, 0.0
        # 取窗内全部 (t, r)（含左边界 now-win_sec 那个点，与旧"win+1 点"对齐）
        hist = [(t, r) for (t, r) in self._bend_ratio_hist if _t_ge(t, now - win_sec)]
        neg = 0.0
        pos = 0.0
        for i in range(1, len(hist)):
            d = hist[i][1] - hist[i - 1][1]  # bend_ratio 差分（正=伸直，负=弯曲）
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
                print(f"[TAP-DIAG] t={_CLOCK():.3f} {msg}", flush=True)
            except Exception:
                pass

    # ---------------------- 真机测量（纯观测，不参与判定）----------------------
    @staticmethod
    def _pctle(q: deque, frac: float) -> float | None:
        if not q:
            return None
        s = sorted(q)
        return s[min(len(s) - 1, int(round(frac * (len(s) - 1))))]

    def _arm_stats(self, by_delta: bool) -> None:
        self._st["arm_delta" if by_delta else "arm_abs"] += 1

    def _record_gest_ms(self) -> None:
        """ARM → 本帧回弹识别 的耗时（检测链响应速度）。"""
        if self._dbg_arm_t:
            self._gest_ms.append((_CLOCK() - self._dbg_arm_t) * 1000.0)

    def _record_click_ms(self) -> None:
        """ARM → TAP_CLICK 实际发出 的耗时（延迟双击模式下含等待，= 用户感知单击延迟）。"""
        if self._dbg_arm_t:
            self._click_ms.append((_CLOCK() - self._dbg_arm_t) * 1000.0)

    # ---------------------- 2026-10-03 段生命周期埋点（纯观测，绝不参与判定）----------------------
    def _emit_seg(self, result: str, now: float,
                  ratio: float | None = None, raw_frac: float | None = None,
                  rise: float | None = None, ctx: str | None = None,
                  freeze_ms: int | None = None) -> None:
        """段结束时落一条记录。result=结局标签；其余为送达帧的回弹门快照。"""
        _wall = round((now - self._dbg_arm_t) * 1000.0) if self._dbg_arm_t else -1
        _active = round((self._total_sec or 0.0) * 1000.0)
        if freeze_ms is None:
            freeze_ms = max(0, _wall - _active)  # 墙钟 - 有效帧计时 ≈ None 冻结时长
        rec = {
            "result": result,
            "ctx": ctx or ("dbl2" if self._double_candidate
                           else ("rescue2" if self._rescue_candidate else "1")),
            "wall_ms": _wall,                        # ARM→段末墙钟（含冻结/等待，=用户感知）
            "freeze_ms": int(freeze_ms),             # None 冻结时长（补发类段显式给 0）
            "none_n": self._dbg_none_in_tap,         # 段内 bend=None 帧数
            "seg_min": (round(float(self._seg_min_ratio), 3)
                        if self._seg_min_ratio is not None else None),
            "ratio": round(float(ratio), 3) if ratio is not None else None,
            "raw_frac": round(float(raw_frac), 3) if raw_frac is not None else None,
            "rise": round(float(rise), 3) if rise is not None else None,
        }
        self._seg_logs.append(rec)

    def take_seg_logs(self) -> list:
        """取走并清空自上次以来结束的段记录（hands_tasks 每帧调用）。"""
        if not self._seg_logs:
            return []
        out = list(self._seg_logs)
        self._seg_logs.clear()
        return out

    def _stats_maybe_print(self) -> None:
        if not TAP_STATS:
            return
        now = _CLOCK()
        if self._stats_last_t == 0.0:
            self._stats_last_t = now
            return
        if now - self._stats_last_t < TAP_STATS_PRINT_SEC:
            return
        self._stats_last_t = now
        s = self._st
        arms = s["arm_delta"] + s["arm_abs"]
        none_pct = 100.0 * s["none"] / max(1, s["frames"])

        def _dist(q: deque) -> str:
            if not q:
                return "avg- p90-"
            avg = sum(q) / len(q)
            return f"avg{avg:.0f} p90{self._pctle(q, 0.9):.0f}ms"

        print(
            f"[TAP-STAT] 帧{s['frames']} None{s['none']}({none_pct:.1f}%) "
            f"ARM{arms}(差分{s['arm_delta']}/绝对{s['arm_abs']}) 尖峰拒{s['spike']} "
            f"点击段None{s['none_ah_frames']}帧/{s['none_ah_seg']}段 | "
            f"首击{s['first_tap']} 单击发出{s['click']} 双击{s['dbl']} "
            f"趋势确认{s['trend']} 补救击{s['rescue_tap']} 长按{s['long']}"
            f"(浅弯拦{s['long_shallow']}/3D门拦{s['long_3d_block_seg']}段{s['long_3d_block_frames']}帧) | "
            f"丢手杀等待{s['wait_none_cancel']}/补救{s['rescue_none_cancel']} | "
            f"回弹 {_dist(self._gest_ms)} | 送达 {_dist(self._click_ms)}",
            flush=True)

    # ---------------------- 长按 3D 正向证据门 ----------------------
    @staticmethod
    def _long_press_3d_gate(long_ev: dict | None) -> tuple[bool, str]:
        """BEND_HOLD→PRESSED 转换前的 3D 正向证据校验（单击/ARM 不调用）。

        long_ev: {"mcp_idx_deg": 食指近节骨-掌轴 3D 夹角(度)|None,
                  "bend3d": 3D 弯度(纯观测), "dip3d": 3D DIP 角(纯观测)}
        Returns: (ok, reason)。证据缺失/超阈 fail-closed：只阻止转长按，不影响点击。
        """
        if not isinstance(long_ev, dict):
            return False, "无3D证据"
        ang = long_ev.get("mcp_idx_deg")
        if ang is None:
            return False, "world缺失"
        try:
            ang = float(ang)
        except (TypeError, ValueError):
            return False, "角度异常"
        if ang > LONG_GATE_MCP_IDX_DEG:
            return False, f"食近节角{ang:.0f}>{LONG_GATE_MCP_IDX_DEG:.0f}"
        return True, f"食近节角{ang:.0f}"

    def _long3d_obs(self, long_ev: dict | None) -> str:
        """[LONG3D] 心跳用：3D 弯度/DIP 角观测串（阈值未标定，本版本只记录不门控）。"""
        if isinstance(long_ev, dict):
            b3 = long_ev.get("bend3d")
            d3 = long_ev.get("dip3d")
            return (f" 3D弯{float(b3):.2f}" if b3 is not None else " 3D弯n/a") + (
                f" DIP3D{float(d3):.0f}°" if d3 is not None else " DIP3Dn/a")
        return " 3D弯n/a DIP3Dn/a"

    # ---------------------- 主入口 ----------------------
    def update(self, bend: float | None, long_ev: dict | None = None) -> int:
        """
        Args:
            bend: 食指伸直度 = dist(TIP_8, MCP_5) / dist(MCP_5, MCP_17)；无手传None
            long_ev: 长按转换专用 3D 正向证据（mcp_idx_deg/bend3d/dip3d），仅门 LONG_DOWN；
                     None=无 3D 证据，长按 fail-closed，点击链路不受影响
        Returns:
            E.{NONE,TAP_CLICK,LONG_DOWN,LONG_UP,DRAG_MOVE} 本帧事件
        """
        self.event_this_frame = E.NONE
        self.armed_this_frame = False
        now = _CLOCK()

        # ⭐ 真机测量：帧计数（纯观测）+ 周期摘要（到点才打印一行，不影响任何判定分支）
        self._st["frames"] += 1
        if bend is None:
            self._st["none"] += 1
        self._stats_maybe_print()

        # ---- 冷却（墙钟：每次 update 都查，None 帧也消耗冷却——与旧"每次 update 减1帧"一致）----
        if self.state == _S.COOLDOWN:
            if _t_ge(now, self._cooldown_until):
                # 补发单击后需转入补救窗口 RESCUE_DBL（_enter_cooldown 时指定）；其余回 IDLE
                self.state = self._cooldown_next
                self._cooldown_next = _S.IDLE
                self._hold_sec = 0.0
                self._armed_by_delta = False
                if self.state == _S.RESCUE_DBL:
                    # ⭐ 补救窗口从【真正进入 RESCUE 状态】起算（cooldown 墙钟不占补救时间，
                    #   否则冷却会吃掉大半个窗口，第二击弯下时已超时回 IDLE）
                    self._rescue_deadline = now + RESCUE_DBL_WINDOW_SEC
                    self._rescue_candidate = False
            if bend is not None:
                self._last_valid_t = now   # 冷却中的有效帧仍推进有效帧时刻（不影响任何进行中计数）
            self._update_baseline(bend, can_update=True)
            return E.NONE

        # ---- 无手(黑帧)：手势进行中无限期冻结，松手由手回来时的物理状态裁决；仅10s硬兜底 ----
        if bend is None:
            if self._none_run_start is None:
                self._none_run_start = now
            _none_dur = now - self._none_run_start
            _none_dead = _none_dur >= NONE_GESTURE_ABANDON_SEC
            if self.state in (_S.BEND_HOLD, _S.BEND_ARMED, _S.PRESSED):
                if not _none_dead:
                    # ⭐⭐⭐ 冻结：不推进计时、不释放、不清差分历史、不动 _smoothed_bend。
                    #   手回来当帧走正常分支：手指仍弯→继续手势；已回直→回直三门成立才释放。
                    #   时间不裁决松手（实测 MediaPipe 丢手 0.5s+ 手指仍弯着，超时释放=凭空双击之源）。
                    if self.state in (_S.BEND_HOLD, _S.BEND_ARMED):
                        if self._dbg_none_in_tap == 0:
                            self._st["none_ah_seg"] += 1      # 本段第一次 None = 又一个受门控影响的点击段
                        self._dbg_none_in_tap += 1
                        self._st["none_ah_frames"] += 1
                        if self._dbg_none_in_tap <= 3 or self._dbg_none_in_tap % 30 == 0:
                            self._diag(f"⚠ bend=None 冻结 状态={self.state} "
                                       f"累计{self._dbg_none_in_tap}帧 dur={_none_dur:.2f}s")
                    # PRESSED 同样冻结：按住保持，不抬键不计时，等手回来按物理状态裁决
                    # （None="没看见手" ≠ "手指上弹"）；真永不回来由下方 10s 硬兜底释放。
                    return E.NONE
                # ---- 连续黑帧超 10s 硬兜底：手真的再也没回来，强制结束防鼠标永久按住 ----
                if self.state == _S.PRESSED:
                    self._diag(f"⚠ PRESSED 连续黑帧{_none_dur:.1f}s≥{NONE_GESTURE_ABANDON_SEC:.0f}"
                               f" → 硬兜底 LONG_UP")
                    self._emit_seg("黑帧强抬", now)
                    self.event_this_frame = E.LONG_UP
                    self._bend_ratio_hist.clear()
                    self._di_sum_neg_now = 0.0
                    self._di_sum_pos_now = 0.0
                    self._enter_cooldown()
                    return self.event_this_frame
                else:
                    # ARMED/HOLD 段超 10s：未发出过任何点击，整段静默丢弃
                    self._diag(f"⚠ bend=None 连续{_none_dur:.1f}s 状态={self.state} → 硬兜底段丢弃")
                    self._emit_seg("黑帧段丢弃", now)
                    self._bend_ratio_hist.clear()
                    self._di_sum_neg_now = 0.0
                    self._di_sum_pos_now = 0.0
                    self._seg_min_ratio = None
                    self._long_3d_blocked = False  # 段已丢弃：3D 门拦截标记同步清（下次 ARM 也会复位）
                    self._double_candidate = False
                    self._rescue_candidate = False
                    self._armed_by_delta = False
                    self.state = _S.IDLE
                    self._smoothed_bend = None
                    self._hold_sec = 0.0
                    self._total_sec = 0.0
                    self._stable_frames = 0
                    return E.NONE
            elif self.state == _S.WAIT_DBL:
                # ⭐ 2026-09-17 取消丢手宽限（用户拍板保留）：挂着 click#1 时遇 None →
                #   立即补发单击结束序列，cooldown 后直接回 IDLE（不开 RESCUE：补救窗同样遇 None 即结束）。
                #   能进 WAIT 必已过回直三门（手指物理上刚伸直），补发不涉及"弯着被伪造"问题。
                self._st["wait_none_cancel"] += 1
                self._tap_pending = False
                self._double_candidate = False
                self._rescue_candidate = False
                self._wait_deadline = 0.0
                self._wait_trend_deadline = 0.0
                self._wait_trend_seen = False
                self._wait_trend_frames = 0
                self.event_this_frame = E.TAP_CLICK
                self.tap_count += 1
                self.last_tap_time = _CLOCK()
                self._st["click"] += 1
                self._record_click_ms()
                self._emit_seg("丢手补发单击", _CLOCK(), ctx="wait_none", freeze_ms=0)
                self._bend_ratio_hist.clear()
                self._di_sum_neg_now = 0.0
                self._di_sum_pos_now = 0.0
                self._diag("③ 单击补发(等待双击中遇None→无宽限立即补发) → cooldown回IDLE")
                self._enter_cooldown()   # next_state 默认 IDLE
                return self.event_this_frame
            elif self.state == _S.RESCUE_DBL:
                # click#1 已送达；遇 None 立即结束补救序列（真单击），回 IDLE
                self._st["rescue_none_cancel"] += 1
                self.state = _S.IDLE
                self._rescue_candidate = False
                self._rescue_deadline = 0.0
                self._bend_ratio_hist.clear()
                self._di_sum_neg_now = 0.0
                self._di_sum_pos_now = 0.0
                return E.NONE
            else:
                # IDLE 等非手势态：差分队列清空（全新场景，不带旧差分）
                self._bend_ratio_hist.clear()
                self._di_sum_neg_now = 0.0
                self._di_sum_pos_now = 0.0
                self.state = _S.IDLE
                self._smoothed_bend = None
                self._hold_sec = 0.0
                self._total_sec = 0.0
                self._stable_frames = 0
            self._last_valid_t = None   # None 段不产生 dt；恢复首帧按标称 FRAME_SEC
            return self.event_this_frame

        # ---- 有手：墙钟 dt（仅有效帧）+ 平滑 bend ----
        #   dt      ：计时用（hold/total/稳定门），钳 MAX_DT_SEC=2帧 防卡顿跳秒
        #   ema_dt  ：bend EMA 用【真实间隔】不钳 2 帧（10fps 的 0.1s 是正常帧率），
        #             异常回退（dt<=0 / >0.5s）在 _bend_ema_alpha 内处理
        if self._last_valid_t is None:
            dt = FRAME_SEC
            ema_dt = FRAME_SEC
        else:
            ema_dt = max(0.0, now - self._last_valid_t)
            dt = min(MAX_DT_SEC, ema_dt)
        self._last_valid_t = now
        # 有效帧打断黑帧连续计数：若此前确有黑帧段 → 开启"松手确认窗"，
        # 窗内 up_trigger 需连续多帧伸直（防重新捕获过渡垃圾帧假释放）
        if self._none_run_start is not None:
            self._post_none_deadline = now + POST_NONE_CONFIRM_WINDOW
            self._straight_confirm = 0
        self._none_run_start = None
        prev_b = self._smoothed_bend
        if self._smoothed_bend is None:
            self._smoothed_bend = float(bend)
        else:
            # ⭐ 2026-10-02 墙钟化 EMA：α=1-exp(-dt/τ) 由本帧真实间隔 ema_dt 决定（τ=50ms），
            #   不使用随 speed 固定的 α、也不吃计时 2 帧钳幅 —— 30/10/120fps 物理平滑量一致
            alpha = _bend_ema_alpha(ema_dt, self._p["BEND_EMA_TAU_SEC"])
            self._smoothed_bend = alpha * float(bend) + (1.0 - alpha) * self._smoothed_bend
        self._prev_smoothed_bend = prev_b

        # 初始化基线
        if self._baseline is None:
            self._baseline = self._smoothed_bend

        # ---- bend_ratio（归一化弯曲度 = 当前伸直度 / 伸直基线）+ 一阶差分 ----
        #   含义：1.00 = 完全伸直（和基线一致）；0.50 = 弯到一半；0.25 = 弯得很深
        B = self._smoothed_bend
        bend_ratio = B / max(self._baseline, 1e-6)
        # ⭐ 计算差分（push (时刻,比值) 到历史队列 + 返回 ΣΔ负 / ΣΔ正）
        di_sum_down, di_sum_up = self._compute_di_sums(bend_ratio, now)

        # 计算 参考阈值（基于 伸直基线）
        enter_t = self._baseline * self._p["ENTER_BEND_RATIO"]
        exit_t  = self._baseline * self._p["EXIT_BEND_RATIO"]

        # ⭐⭐ 2026-09-21 用户拍板：「起弯」只认【下弯变化量】，彻底废除绝对弯曲度入口。
        #   down_trigger = 进入弯曲（IDLE/WAIT_DBL/RESCUE_DBL → ARMED），三处共用，一处修复全路径生效。
        #   真机日志(tap_gate.log L58-60)实证旧逻辑故障：
        #     长按中黑帧 → LONG_UP → 冷却后手指"根本没动、只是还弯着"(ratio=0.252, Σ下=0.000)，
        #     旧 down_abs 兜底(B<enter_t 且 drop>MIN_BEND_DROP)把它当成【新一次起弯】；
        #     手指真正回直时这段幻影手势补发 click，与长按自身的 mouseDown/Up 同点且间隔落在
        #     OS 双击时间(500ms)内 → 长按回直后被系统执行成双击。
        #   手势识别是【动态问题】（变化了多少、如何变化），不是弯下去多少的静态问题：
        #     · 只有近窗内真的发生下弯变化量（ΣΔ负 ≥ 门限）且已离开伸直位，才允许起弯；
        #     · "本来就弯着 / 黑帧恢复时仍弯着 / 长按回直途中还弯着" → 没有新变化量，一概不起弯；
        #     · 双击第二击同理：必须先出现回直变化、之后再出现一段【新的下弯变化】才算第二击。
        #   长按因此与双击天然互斥：LONG_UP 之后若没有新的下弯变化，不会产生任何可配对的点击。
        #   （bend_ratio 上限只是"确实弯离伸直位"的确认门，单独不触发；松开另见下方【真回直双门】。）
        down_delta = (di_sum_down >= self._p["TAP_DOWN_DI_SUM"]) and (
            bend_ratio <= min(self._p["MIN_ABS_BEND_RATIO_FOR_DELTA"],
                              self._p["ENTER_BEND_RATIO"] + 0.06)
        )
        down_trigger = down_delta

        #   up_trigger = 从弯曲回到伸直（BEND_HOLD→TAP / PRESSED→LONG_UP）：
        #   ⭐⭐ 2026-09-21 长按/双击隔离：必须【真的回直】才承认松手。
        #      段内谷底 _seg_min_ratio 在每个弯曲段有效帧更新；松开双门 AND：
        #        ① 回直变化量：bend_ratio - 谷底 ≥ RECOVER_RISE_MIN（从最深弯处弹回得够多）
        #        ② 伸直位确认：bend_ratio ≥ RECOVER_RATIO_MIN（当前已基本伸直）
        #      按住中途手指微松（谷底0.25→0.55~0.70，手指仍明显弯着）两个门都不过 →
        #      不会结束弯曲段、不会产生第一次 Tap、不进 WAIT_DBL → 长按中物理上发不出双击。
        #      旧的 Σ上差分/enter_t/exit_t 低线只作"回弹发生过"的快速线索，不再能单独决定松手。
        if self.state in (_S.BEND_ARMED, _S.BEND_HOLD, _S.PRESSED) \
                and self._seg_min_ratio is not None:
            if bend_ratio < self._seg_min_ratio:
                self._seg_min_ratio = bend_ratio
        _recover_rise = (bend_ratio - self._seg_min_ratio) if self._seg_min_ratio is not None else 0.0
        # 三门 AND：①谷底回升量 ②EMA ratio 到伸直位 ③原始 raw 也真到伸直位（防回弹途中 EMA 先过线）
        _raw_recover_ok = bend >= (self._baseline * RAW_RECOVER_FRAC)
        _recover_ok = ((_recover_rise >= RECOVER_RISE_MIN)
                       and (bend_ratio >= RECOVER_RATIO_MIN)
                       and _raw_recover_ok)
        _up_thr = self._p["TAP_UP_DI_SUM"]
        # 快速回弹线索（窗内回直差分 或 越过旧回伸直线）AND 真回直双门
        up_delta_release = (di_sum_up >= _up_thr) and (B >= enter_t)
        up_abs_fallback   = B >= exit_t
        up_trigger = (up_delta_release or up_abs_fallback) and _recover_ok
        # ⭐⭐⭐ 丢手恢复期确认窗：窗内"松手"必须【连续 N 帧】当前位置都在伸直位
        #   （raw 门 + ratio 门；任一帧重新弯下即清零）。专杀重新捕获过渡垃圾帧造成的假释放。
        if self.state in (_S.BEND_ARMED, _S.BEND_HOLD, _S.PRESSED) \
                and _t_lt(now, self._post_none_deadline):
            _pos_straight = (bend_ratio >= RECOVER_RATIO_MIN) and _raw_recover_ok
            if _pos_straight:
                self._straight_confirm += 1
            else:
                self._straight_confirm = 0
            if up_trigger and self._straight_confirm < POST_NONE_CONFIRM_FRAMES:
                up_trigger = False

        # ================= 状态流转 =================

        if self.state == _S.IDLE:
            self._update_baseline(B, can_update=True)
            if down_trigger:
                self.state = _S.BEND_ARMED
                self.armed_this_frame = True
                self._hold_sec = 0.0        # ARM 当帧为 t0：0 秒（旧帧版首帧计 1，阈值已按 N-1 间隔换算）
                self._total_sec = 0.0
                self._stable_frames = 0
                self._double_candidate = False
                self._dbg_arm_t = _CLOCK()
                self._dbg_none_in_tap = 0
                self._armed_by_delta = True   # 起弯只可能由下弯变化量触发（绝对入口已废除）
                self._seg_min_ratio = bend_ratio  # 段内弯曲谷底从 ARM 当帧起记
                self._long_3d_blocked = False  # 新弯曲段：3D 长按门拦截标记复位
                self._arm_stats(self._armed_by_delta)
                self._diag(f"① ARM 起弯 ratio={bend_ratio:.3f} Σ下={di_sum_down:.3f}/Σ上={di_sum_up:.3f}")
            elif B > self._baseline * 0.97:
                # 非常接近伸直 → 快速更新基线，避免基线长期滞后用户真实伸直姿态变化
                self._baseline = BASELINE_ALPHA_FAST * B + (1 - BASELINE_ALPHA_FAST) * self._baseline

        elif self.state == _S.WAIT_DBL:
            # 第一次 Tap 已完成（单击挂起、不立即触发）：先用短窗探测"二次下弯趋势"，
            #   有趋势→延长等完整第二击(DOUBLE_TAP)；无趋势→立即补发单击（单击快速反应）。
            self._update_baseline(B, can_update=True)
            if B > self._baseline * 0.97:
                self._baseline = BASELINE_ALPHA_FAST * B + (1 - BASELINE_ALPHA_FAST) * self._baseline
            _now = _CLOCK()
            _fire_reason = None  # 本帧需要补发单击的原因（None=不补发）
            if down_trigger:
                # 完整按下判据直接成立（趋势自然算已确认）→ 走常规 ARMED/HOLD，标记双击候选
                self._wait_trend_seen = True
                self.state = _S.BEND_ARMED
                self.armed_this_frame = True
                self._hold_sec = 0.0        # ARM 当帧为 t0：0 秒（旧帧版首帧计 1，阈值已按 N-1 间隔换算）
                self._total_sec = 0.0
                self._stable_frames = 0
                self._double_candidate = True
                self._armed_by_delta = True   # 第二击同样必须是新的下弯变化量
                self._seg_min_ratio = bend_ratio  # 第二击段谷底从本帧起记
                self._long_3d_blocked = False  # 新弯曲段：3D 长按门拦截标记复位
                self._arm_stats(self._armed_by_delta)
                self._dbg_arm_t = _CLOCK()   # 第二击段耗时从第二击起弯算（click#1 延迟已在补发前记完）
                self._dbg_none_in_tap = 0
            elif not self._wait_trend_seen:
                # ---- 趋势探测短窗：低门限差分 + 已离开伸直位 + 连续确认（不要求完整点击）----
                _trend_thr = TREND_DI_FRAC * self._p["TAP_DOWN_DI_SUM"]
                trend_down = (di_sum_down >= _trend_thr) and (bend_ratio <= TREND_MAX_RATIO)
                if trend_down:
                    self._wait_trend_frames += 1
                    if self._wait_trend_frames >= TREND_CONFIRM_FRAMES:
                        # 二次下弯趋势确认 → 双击意图，延长窗口等完整第二击
                        self._wait_trend_seen = True
                        self._st["trend"] += 1
                        self._wait_deadline = _now + DOUBLE_TAP_WINDOW_SEC
                        self._diag(f"⓪ 探测到二次下弯趋势 ratio={bend_ratio:.3f} "
                                   f"Σ下={di_sum_down:.3f}≥趋势阈{_trend_thr:.3f} "
                                   f"→ 延长{DOUBLE_TAP_WINDOW_SEC:.2f}s等完整第二击")
                else:
                    self._wait_trend_frames = 0
                if not self._wait_trend_seen and _now > self._wait_trend_deadline:
                    # 短窗内无二次下弯趋势 → 立即补发单击（0.38s→~0.16s 的快速反应）
                    _fire_reason = f"趋势短窗{TREND_WAIT_SEC:.2f}s内无二次下弯→立即补发"
            elif _now > self._wait_deadline:
                # 已确认趋势但第二击未在完整窗内完成 → 补发单击，后续 RESCUE_DBL 补救窗兜底
                _fire_reason = f"趋势后{DOUBLE_TAP_WINDOW_SEC:.2f}s第二击未完成→补发"
            if _fire_reason is not None:
                # 确认单击，补发 TAP_CLICK（click#1 送达），cooldown 后进入 RESCUE_DBL 补救窗
                self._tap_pending = False
                self._double_candidate = False
                self._rescue_candidate = False
                self.event_this_frame = E.TAP_CLICK
                self.tap_count += 1
                self.last_tap_time = _now
                self._st["click"] += 1
                self._record_click_ms()
                self._emit_seg("超时补发单击", _now, ctx="wait", freeze_ms=0)
                self._diag(f"③ 单击补发({_fire_reason}) → "
                           f"cooldown后开补救窗口{RESCUE_DBL_WINDOW_SEC:.1f}s，窗内第二击将按位置门控判双击/单击")
                self._enter_cooldown(next_state=_S.RESCUE_DBL)
                return self.event_this_frame

        elif self.state == _S.RESCUE_DBL:
            # click#1 已补发送达；补救窗内等第二击弯曲：窗内开始的 Tap 回弹时立即发 click#2
            self._update_baseline(B, can_update=True)
            if B > self._baseline * 0.97:
                self._baseline = BASELINE_ALPHA_FAST * B + (1 - BASELINE_ALPHA_FAST) * self._baseline
            if down_trigger:
                # 第二击弯曲在补救窗内开始 → 走常规 ARMED/HOLD，标记补救候选（回弹立即单击）
                self.state = _S.BEND_ARMED
                self.armed_this_frame = True
                self._hold_sec = 0.0        # ARM 当帧为 t0：0 秒（旧帧版首帧计 1，阈值已按 N-1 间隔换算）
                self._total_sec = 0.0
                self._stable_frames = 0
                self._rescue_candidate = True
                self._double_candidate = False
                self._armed_by_delta = True   # 补救第二击同样必须是新的下弯变化量
                self._seg_min_ratio = bend_ratio  # 补救第二击段谷底从本帧起记
                self._long_3d_blocked = False  # 新弯曲段：3D 长按门拦截标记复位
                self._arm_stats(self._armed_by_delta)
                self._dbg_arm_t = _CLOCK()   # 补救第二击段耗时从本击起弯算
                self._dbg_none_in_tap = 0
                self._diag(f"④ 补救窗口内捕获第二击弯曲 → 回弹将按位置门控发双击(同图标)/单击(异处)")
            elif _CLOCK() > self._rescue_deadline:
                # 补救窗超时仍无第二击 = 真·单击
                self.state = _S.IDLE
                self._rescue_candidate = False
                self._rescue_deadline = 0.0

        elif self.state == _S.BEND_ARMED:
            # ARMED 的下一帧：仍需满足"还没回到伸直"（不能刚一帧就直接弹回去=尖峰）
            # ⭐⭐ 2026-09-21 深弯豁免：起弯帧已实打实弯到 DEEP_ARM_KEEP_RATIO 以下（噪声不可能
            #   瞬时压出 >0.45 的 ratio 跌幅），第二帧即便立刻弹直也是【最短合法点击】，
            #   进 HOLD 后由回直三门诊放/继续等；不能整条吞掉让后续抖动顶替成双击第二击。
            _deep_arm = (self._seg_min_ratio is not None
                         and self._seg_min_ratio <= DEEP_ARM_KEEP_RATIO)
            still_down = _deep_arm or (
                (di_sum_down >= max(0.0, self._p["TAP_DOWN_DI_SUM"] * 0.35))
                or (B < enter_t + self._baseline * 0.03))
            if still_down:
                # ARMED 中每个有效帧都累计（无稳定门）；旧第2帧 hold=2>=MIN_BEND_HOLD
                self._hold_sec += dt
                self._total_sec += dt
                if _t_ge(self._hold_sec, self._p["MIN_HOLD_SEC"]):
                    self.state = _S.BEND_HOLD
            else:
                # 第2帧就不满足了 → 单帧尖峰
                self._st["spike"] += 1
                self._emit_seg("尖峰取消", now)
                self._diag(f"✗ 尖峰拒绝(第2帧未保持弯曲) ratio={bend_ratio:.3f} Σ下={di_sum_down:.3f} "
                           f"{'双击候选→回WAIT' if self._double_candidate else ('补救候选→回RESCUE' if self._rescue_candidate else '→IDLE')}")
                if self._double_candidate:
                    # 双击窗口内的第二次弯曲只是尖峰 → 回 WAIT_DBL 继续等（窗口到期自动补单击）
                    self.state = _S.WAIT_DBL
                    self._double_candidate = False
                elif self._rescue_candidate:
                    # 补救窗口内的第二击只是尖峰 → 回 RESCUE_DBL 继续等（窗口到期回 IDLE）
                    self.state = _S.RESCUE_DBL
                    self._rescue_candidate = False
                else:
                    self.state = _S.IDLE
                self._hold_sec = 0.0
                self._total_sec = 0.0
                self._stable_frames = 0
                self._armed_by_delta = False
                self._seg_min_ratio = None

        elif self.state == _S.BEND_HOLD:
            # ✨ 总时长：从 ARM 开始累计（不管有没有稳定，都要算在 TAP_MAX_SEC 时间窗里）
            self._total_sec += dt

            # ---------------- 修复1：先判断 EXIT（伸直回弹） → 优先于 LONG_PRESS ----------------
            if up_trigger:
                # -------- 修复3：快速短按优先级。总时长 <= TAP_MAX_SEC 一律按 TAP，绝不能算长按 --------
                # 判定本次是 Tap（快速短按 或 长按阈值前回弹）还是 真·长按释放
                #   旧帧版：total<=18 或 hold<=22；墙钟阈值 (N-1)/30 与 22/30（hold 用严格<，f24=阈值不算 tap）
                is_tap = (_t_le(self._total_sec, self._p["TAP_MAX_SEC"])
                          or _t_lt(self._hold_sec, self._p["LONG_PRESS_SEC"]))
                _seg_ms = (now - self._dbg_arm_t) * 1000.0 if self._dbg_arm_t else 0.0
                _none_n = self._dbg_none_in_tap
                # ⭐⭐ 2026-09-21 浅弯门：整段谷底都没弯过 TAP_MIN_BEND_RATIO（半弯抖动/手指挥弹），
                #   即使回弹三门诊全过也不发任何点击：首击→静默回 IDLE；无效第二击→回 WAIT_DBL
                #   让窗口到期补单击（不能占掉第二击名额，否则后面的真点击会被错配/顶包）。
                _shallow_tap = is_tap and (
                    self._seg_min_ratio is None or self._seg_min_ratio > TAP_MIN_BEND_RATIO)
                # ⭐ 2026-10-15 3D 正向门拦截过的段松手：整段从未发过 LONG_DOWN，
                #   也不补发 click（握拳保持不是点击动作），静默结束；第二击身份按浅弯同口径
                #   回 WAIT/RESCUE，避免占掉真第二击名额。
                if self._long_3d_blocked:
                    self._emit_seg("3D门拦截松手", now, ratio=bend_ratio,
                                   raw_frac=(bend / self._baseline) if self._baseline else None,
                                   rise=_recover_rise)
                    _vl = (f"{self._seg_min_ratio:.3f}"
                           if self._seg_min_ratio is not None else "n/a")
                    self._diag(f"⑥ 3D门拦截段松手，静默不发 click/LONG_UP "
                               f"段{self._total_sec:.3f}s 谷底{_vl}"
                               f"{' 第二击→回WAIT' if self._double_candidate else (' 补救→回RESCUE' if self._rescue_candidate else ' →IDLE')}")
                    for _ in range(2):
                        self._bend_ratio_hist.append(
                            (now, self._smoothed_bend / max(self._baseline, 1e-6)))
                    if self._double_candidate:
                        self.state = _S.WAIT_DBL
                    elif self._rescue_candidate:
                        self.state = _S.RESCUE_DBL
                    else:
                        self.state = _S.IDLE
                    self._double_candidate = False
                    self._rescue_candidate = False
                    self._hold_sec = 0.0
                    self._total_sec = 0.0
                    self._stable_frames = 0
                    self._armed_by_delta = False
                    self._seg_min_ratio = None
                    self._long_3d_blocked = False
                    return E.NONE
                if _shallow_tap:
                    self._st["shallow"] += 1
                    self._emit_seg("浅弯取消", now, ratio=bend_ratio,
                                   raw_frac=(bend / self._baseline) if self._baseline else None,
                                   rise=_recover_rise)
                    self._diag(f"⑤ 浅弯取消(谷底{self._seg_min_ratio:.3f}>{TAP_MIN_BEND_RATIO}) "
                               f"不发点击 {'第二击→回WAIT等真第二击' if self._double_candidate else ('补救→回RESCUE' if self._rescue_candidate else '→IDLE')}")
                    for _ in range(2):
                        self._bend_ratio_hist.append(
                            (now, self._smoothed_bend / max(self._baseline, 1e-6)))
                    if self._double_candidate:
                        self.state = _S.WAIT_DBL
                    elif self._rescue_candidate:
                        self.state = _S.RESCUE_DBL
                    else:
                        self.state = _S.IDLE
                    self._double_candidate = False
                    self._rescue_candidate = False
                    self._hold_sec = 0.0
                    self._total_sec = 0.0
                    self._stable_frames = 0
                    self._armed_by_delta = False
                    self._seg_min_ratio = None
                elif OS_DOUBLE_CLICK and is_tap:
                    # ★ 即时模式：每次勾按回弹立刻单击（双击交给 Windows 双击时间自然识别）
                    self._double_candidate = False
                    self._tap_pending = False
                    self.event_this_frame = E.TAP_CLICK
                    self.tap_count += 1
                    self.last_tap_time = _CLOCK()
                    self._st["click"] += 1
                    self._record_gest_ms()
                    self._record_click_ms()
                    self._emit_seg("单击即时", now, ratio=bend_ratio,
                                  raw_frac=(bend / self._baseline) if self._baseline else None,
                                  rise=_recover_rise)
                    self._diag(f"② TAP点击发出(即时) 段长{self._total_sec:.3f}s≈{_seg_ms:.0f}ms "
                               f"None门控{_none_n}帧 ratio={bend_ratio:.3f}")
                    self._enter_cooldown()
                elif is_tap and self._double_candidate:
                    # 延迟确认模式：双击窗口内完成第二次 Tap → DOUBLE_TAP（挂起的单击作废，
                    #   只发一次 doubleClick = 物理点击恰好 2 次，严格互斥）
                    self._double_candidate = False
                    self._tap_pending = False
                    self.event_this_frame = E.DOUBLE_TAP
                    self.double_count += 1
                    self.last_double_time = _CLOCK()
                    self._st["dbl"] += 1
                    self._record_gest_ms()
                    self._emit_seg("原子双击", now, ctx="dbl2", ratio=bend_ratio,
                                  raw_frac=(bend / self._baseline) if self._baseline else None,
                                  rise=_recover_rise)
                    self._enter_cooldown(DOUBLE_TAP_COOLDOWN_SEC)
                elif is_tap and self._rescue_candidate:
                    # ⭐ 补救窗口内的第二击 Tap：发 RESCUE_DBL_TAP，由 mouse_controller 按位置门控：
                    #   落点仍在 click#1 同一图标上 → 原子 doubleClick（几十ms 内两次点击，OS 必判双击打开，
                    #   彻底规避"选中后慢节奏再点=重命名"）；落点已移到别处 → 降级普通 click。
                    self._rescue_candidate = False
                    self._rescue_deadline = 0.0
                    self.event_this_frame = E.RESCUE_DBL_TAP
                    self._st["rescue_tap"] += 1
                    self._record_gest_ms()
                    self._emit_seg("补救第二击", now, ctx="rescue2", ratio=bend_ratio,
                                  raw_frac=(bend / self._baseline) if self._baseline else None,
                                  rise=_recover_rise)
                    self._diag(f"⑤ 补救第二击捕获(段长{self._total_sec:.3f}s≈{_seg_ms:.0f}ms) "
                               f"→ 交mouse_controller按位置门控:同图标=doubleClick打开/异处=普通click")
                    self._enter_cooldown()
                elif is_tap:
                    # 第一次 Tap：不立即单击 → 进双击窗口等第二次（超时在 WAIT_DBL 补发单击）
                    self._st["first_tap"] += 1
                    self._record_gest_ms()
                    self._emit_seg("首击待双击", now, ratio=bend_ratio,
                                  raw_frac=(bend / self._baseline) if self._baseline else None,
                                  rise=_recover_rise)
                    self.state = _S.WAIT_DBL
                    self._tap_pending = True
                    self._double_candidate = False
                    self._wait_deadline = now + DOUBLE_TAP_WINDOW_SEC
                    # ⭐ 趋势探测短窗：TREND_WAIT_SEC 内看不到二次下弯趋势 → 立即补发单击
                    self._wait_trend_seen = False
                    self._wait_trend_frames = 0
                    self._wait_trend_deadline = now + TREND_WAIT_SEC
                    self._hold_sec = 0.0
                    self._total_sec = 0.0
                    self._stable_frames = 0
                    self._armed_by_delta = False
                    self._seg_min_ratio = None   # 本段已随第一次 Tap 结束（第二击 ARM 时重新记谷底）
                    # 同 _enter_cooldown 的差分清零手法：推两个"当前时刻"伸直值，让窗内 ΣΔ 立刻回 0
                    if self._baseline is not None and self._smoothed_bend is not None:
                        for _ in range(2):
                            self._bend_ratio_hist.append(
                                (now, self._smoothed_bend / max(self._baseline, 1e-6)))
                else:
                    # 真·超长按住后才松手 → 作为 LONG_UP 释放；若此前挂着单击/双击/补救候选，
                    #   拖拽语义优先，挂起的单击作废（避免 click+mouseDown 连在同一点）
                    self._tap_pending = False
                    self._double_candidate = False
                    self._rescue_candidate = False
                    self._rescue_deadline = 0.0
                    self.event_this_frame = E.LONG_UP
                    self.last_long_time = _CLOCK()
                    self._emit_seg("长按释放", now, ratio=bend_ratio,
                                  raw_frac=(bend / self._baseline) if self._baseline else None,
                                  rise=_recover_rise)
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
                # 连续第 2+ 个稳定【样本帧】才把本帧 dt 计入保持时长（第1个稳定样本=去抖，不计时）
                if self._stable_frames >= self._p["BEND_STABLE_MIN_HOLD"]:
                    self._hold_sec += dt

            # ---------------- 修复1：再判断 LONG_PRESS（最后再触发长按） ----------------
            #   旧 hold_frames>=LONG_PRESS_FRAMES+1(23)；墙钟 hold_sec>=22/30（第24帧触发=t0.767s）
            # ⭐ 2026-10-03 长按浅弯门：整段谷底 > TAP_MIN_BEND_RATIO(0.66) 的半弯保持不转长按
            #   （同样深度的松指会被判"浅弯取消"，长按却能 mouseDown，口径不一致：
            #    内容区=误选中/误拖，标题栏上=按住标题栏误拖窗口，见 [TAP-SEG] 转长按 谷底0.729）。
            #   不取消、留在 BEND_HOLD：后续弯深(≤0.66)下帧即转；回直则走浅弯取消静默结束。
            _long_ripe = _t_ge(self._hold_sec, self._p["LONG_PRESS_SEC"])
            _long_ripe = False  # ⚠️ 食指长按逻辑已禁用（用户要求 2026-10-17）：不再触发 LONG_DOWN
            _long_shallow = (self._seg_min_ratio is not None
                             and self._seg_min_ratio > TAP_MIN_BEND_RATIO)
            # ⭐ 2026-10-15 3D 正向证据：只在长按转换点校验（点击/ARM 不读）。
            #   证据达标才 LONG_DOWN；不达标留 HOLD 逐帧重判（遮挡恢复/姿态转正即放行）。
            _long_3d_ok, _long_3d_why = self._long_press_3d_gate(long_ev)
            if _long_ripe and not _long_shallow and _long_3d_ok:
                self.state = _S.PRESSED
                self.event_this_frame = E.LONG_DOWN
                self.long_count += 1
                self.last_long_time = _CLOCK()
                self._st["long"] += 1
                self._long_3d_blocked = False  # 证据恢复成功转换：此前的拦截历史作废
                self._emit_seg("转长按", now, ratio=bend_ratio)
                self._diag(f"⚠⚠ 转长按LONG_DOWN(没检测到回弹!) 总{self._total_sec:.3f}s "
                           f"≈{(now-self._dbg_arm_t)*1000.0:.0f}ms None门控{self._dbg_none_in_tap}帧 "
                           f"ratio={bend_ratio:.3f} Σ上={di_sum_up:.3f}(需{self._p['TAP_UP_DI_SUM']}) "
                           f"3D证据[{_long_3d_why}{self._long3d_obs(long_ev)}]"
                           f" → 双击被吞：两次弯曲被合成一段按住")
                # 第二次弯曲演变为长按拖拽 → 挂起的单击/补救候选作废（拖拽语义优先）
                self._tap_pending = False
                self._double_candidate = False
                self._rescue_candidate = False
                self._rescue_deadline = 0.0
            elif _long_ripe and _long_shallow:
                self._st["long_shallow"] += 1
                self._diag(f"⏸ 浅弯保持不转长按 谷底{self._seg_min_ratio:.3f}"
                           f">{TAP_MIN_BEND_RATIO} 总{self._total_sec:.3f}s → 留 HOLD 等弯深/回直")
            elif _long_ripe and not _long_3d_ok:
                # 3D 正向证据不达标（握拳构型/world 缺失）：不转长按，留 HOLD 逐帧重判。
                # 标记本段"被拦截"→ 松手时静默结束（不发 click/LONG_UP）。
                if not self._long_3d_blocked:
                    self._long_3d_blocked = True
                    self._st["long_3d_block_seg"] += 1
                self._st["long_3d_block_frames"] += 1
                if now - self._long_3d_diag_t >= 0.30:
                    self._long_3d_diag_t = now
                    _vl = (f"{self._seg_min_ratio:.3f}"
                           if self._seg_min_ratio is not None else "n/a")
                    print(f"[LONG3D] 长按转换被3D门拦截（{_long_3d_why}）留HOLD逐帧重判 "
                          f"总{self._total_sec:.3f}s 2D谷底{_vl}"
                          f"{self._long3d_obs(long_ev)}"
                          f"（bend3d/DIP 本版本仅观测不门控）", flush=True)

        elif self.state == _S.PRESSED:
            if up_trigger:
                # 松开 → release
                self.event_this_frame = E.LONG_UP
                self.last_long_time = _CLOCK()
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

    def _enter_cooldown(self, sec: float | None = None, next_state: int | None = None):
        now = _CLOCK()
        self.state = _S.COOLDOWN
        # 旧"进入后数 N 次 update"= N 个帧间隔 → deadline=now+N*FRAME_SEC
        self._cooldown_until = now + (self._p["COOLDOWN_SEC"] if sec is None else float(sec))
        # COOLDOWN 结束后转入的状态：默认 IDLE；补发单击后传 RESCUE_DBL 进入双击补救窗口
        self._cooldown_next = next_state if next_state is not None else _S.IDLE
        self._hold_sec = 0.0
        self._total_sec = 0.0
        self._stable_frames = 0
        self._armed_by_delta = False
        self._seg_min_ratio = None
        self._tap_pending = False
        self._double_candidate = False
        self._rescue_candidate = False
        # 注意：_rescue_deadline 不在此清——补发单击后靠它在 RESCUE_DBL 状态判超时
        self._wait_deadline = 0.0
        # ⭐ 冷却时不清空 bend_ratio_hist（保留弯曲尾部的差分，让"连点"的第二次 tap 也能被差分准确识别）
        #   但要手动 push 两个"当前时刻伸直值"，让下一帧 IDLE 阶段的 ΣΔ 不会被上一个 tap 的尾巴误触发
        if self._baseline is not None and self._smoothed_bend is not None:
            # 推两个当前点（彼此零差分）→ 防止连点残留
            for _ in range(2):
                self._bend_ratio_hist.append(
                    (now, self._smoothed_bend / max(self._baseline, 1e-6)))

    # ---------------------- 查询 / 绘制 ----------------------
    def metrics(self) -> dict:
        return {
            "state":          self.state,
            "event":          self.event_this_frame,
            "bend":           round(self._smoothed_bend or 0.0, 4),
            "baseline":       round(self._baseline       or 0.0, 4),
            "enter_t":        round((self._baseline or 0.0) * self._p["ENTER_BEND_RATIO"], 4),
            "exit_t":         round((self._baseline or 0.0) * self._p["EXIT_BEND_RATIO"],  4),
            "hold_sec":       round(self._hold_sec, 3),
            "total_sec":      round(self._total_sec, 3),
            "stable_frames":  self._stable_frames,
            "TAP_MAX_SEC":    round(self._p["TAP_MAX_SEC"], 3),
            "LONG_PRESS_SEC": round(self._p["LONG_PRESS_SEC"], 3),
            "cooldown_left":  round(max(0.0, self._cooldown_until - _CLOCK()), 3),
            "tap_count":      self.tap_count,
            "long_count":     self.long_count,
            "double_count":   self.double_count,
            # 双击窗口剩余秒数（WAIT_DBL 中 >0；其余状态=0）：趋势确认前=短窗剩余，确认后=完整窗剩余
            "dbl_window_left": round(max(0.0, (self._wait_deadline if self._wait_trend_seen
                                               else self._wait_trend_deadline) - _CLOCK()), 3)
                                if self.state == _S.WAIT_DBL else 0.0,
            "dbl_trend_seen": self._wait_trend_seen,
            # 补救窗口剩余秒数（RESCUE_DBL 中 >0；其余状态=0）
            "rescue_window_left": round(max(0.0, self._rescue_deadline - _CLOCK()), 3)
                                if self.state == _S.RESCUE_DBL else 0.0,
            # ⭐ 差分点击新增字段（给 UI 条/调试）
            "bend_ratio":     round(
                (self._smoothed_bend / max(self._baseline, 1e-6)) if (self._baseline is not None and self._smoothed_bend is not None) else 0.0,
                4),
            "di_sum_down":    round(self._di_sum_neg_now, 4),   # 向下按=正值（累积 Σ|Δ负|）
            "di_sum_up":      round(self._di_sum_pos_now, 4),   # 向上抬=正值（累积 ΣΔ正）
            "DI_DOWN_THR":    round(self._p["TAP_DOWN_DI_SUM"], 4),
            "DI_UP_THR":      round(self._p["TAP_UP_DI_SUM"],   4),
            "DI_WINDOW_SEC":  round(self._p["DI_WINDOW_SEC"], 3),
            "armed_by_delta": bool(self._armed_by_delta),
            # 把硬度速度也放进去，方便上层 UI 直接拿
            "hardness_pct":   self.hardness_pct,
            "speed_pct":      self.speed_pct,
        }


# ======================================================================
# 双指并拢右键检测器（2026-09-16）：食指+中指并拢，同时勾按回弹 → 鼠标右键
# ======================================================================
class TwoFingerRightClickDetector:
    """食/中双指并拢 + 双指勾按回弹 → RIGHT_TAP（每次手势恰好 1 次右键，无双击语义）。

    2026-09-17 v7（用户拍板：分阶段阈值 + 起势顺序门）：
      · 起势（IDLE）：TIP 3D 排 ≤ 0.80（指尖先并拢）且须在 bend≥0.55 时连续并拢 3 帧取得资格，
        才允许起弯 —— 区分"先并拢后勾"的右键与"弯到底才碰巧并拢"的单指点击（左键不再被抢杀）；
      · 手势中（ARMED/HOLD/PRESSED）：深弯遮挡 TIP 失真（实测 1.33），改看 DIP 3D 排 ≤ 1.30；
      · MCP/PIP/spread 不参与判定，仅日志；任意微弯姿态可起势，不要求伸直；
      · 弯曲度只算一个"双指单元合成 bend"（四关节中点代入与左键相同的关节角公式）。
    弯曲/回弹状态机逐条照搬左键 FingerTapClickDetector 模板；
    右键门控 = 起势顺序门 + 手势中 DIP 连续 2 帧失效去抖。
    """

    def __init__(self, hardness: float | None = None, speed: float | None = None):
        self._p = resolve_tap_params(hardness=hardness, speed=speed)
        self.state = _S.IDLE
        self.event_this_frame = E.NONE
        # 双指单元合成 bend 的平滑值 / 伸直基线 / 近期伸直高值（单条序列，同左键）
        self._sm:   float | None = None
        self._base: float | None = None
        self._maxr: float | None = None
        self._hist: deque = deque(maxlen=DI_HIST_MAXLEN)   # (时刻, r) 滑窗（差分，同左键）
        self._di_down = 0.0
        self._di_up = 0.0
        self._prev_r: float | None = None     # 合成 r 上一帧（同左键 _prev_smoothed_bend，稳定性判定）
        self._max_recent_r: float | None = None  # 合成 r 近期伸直高值（右键绝对兜底 down_abs 的 drop 用；左键已无此机制）
        # ⭐ 墙钟化（2026-09-18）
        self._hold_sec = 0.0                  # ARM 起有效保持时长（同左键口径）
        self._total_sec = 0.0                 # ARM 起有效总时长（TAP_MAX_SEC 判定）
        self._stable_frames = 0               # 连续稳定【样本帧】（毛刺门，保留帧计数）
        self._cooldown_until = 0.0            # COOLDOWN 结束时刻（monotonic）
        self._last_valid_t: float | None = None   # 上一有效帧时刻（None 不推进，恢复首帧=FRAME_SEC）
        self._armed_by_delta = False
        self._none_since: float | None = None     # 手势中连续无效段起点（>NONE_GRACE_SEC 作废）
        self._gap_break_frames = 0            # 手势中 together 连续失效【样本帧】计数（去抖，保留帧计数）
        # ⭐ 2026-10-03 双指独立弯度门：食/中各自关节角弯度（0.20弯~0.88直），None=本帧不可算
        self._last_idx_bend: float | None = None
        self._last_mid_bend: float | None = None
        self._tip_break_frames = 0          # 手势中两【指尖】距连续超门帧计数（2026-10-04 TIP 持续门，去抖 2 帧）
        self._entry_t0: float | None = None   # 高 bend 指尖并拢连续段起点；None=无资格（一次性门票）
        self._entry_ok = False                # 资格已在【并拢帧】上确认满 TWO_FINGER_ENTRY_SEC
        self._entry_ok_at: float | None = None  # 资格确认墙钟（仅 [2F-GRACE] 诊断显示确认后停顿多久）
        self._entry_grace_n = 0     # 资格确认后连续"弯指并拢高bend帧"数（≤GRACE_FRAMES 保留资格）
        self._arm_t = None
        self._diag_n = 0              # 有效帧计数（心跳节拍，reset 不清零）
        self._diag_geom_n = 0         # 未并拢帧计数（几何心跳节拍，reset 不清零）
        self._last_gap: float | None = None
        self._last_parts: tuple | None = None  # (MCP,PIP,DIP,TIP,spread) 指距比值（仅日志，不参与判定）
        self.armed_this_frame = False        # 本帧是否 IDLE→ARMED（hands_tasks 据此取消左指检测器）
        self.right_count = 0
        # ⭐ 2026-10-03 [2F-SEG] 右键段生命周期埋点（纯观测，不参与任何判定）：
        #   起势资格取得即推一条；ARM 起记录 r 谷底/None 帧；送达/作废各终点推一条。
        self._rf_logs: deque = deque(maxlen=30)
        self._rf_entry_tip = None            # 起势资格取得当帧 TIP 比值
        self._rf_entry_bend = None           # 起势资格取得当帧合成 bend
        self._rf_arm_r = None                # ARMED 当帧合成 r
        self._rf_min_r = None                # 段内合成 r 谷底
        self._rf_none_n = 0                  # 段内无效帧计数
        self._rf_kind = ""                   # 起弯入口："差分"/"绝对"
        # ⭐ 2026-10-04 检测器本体直接埋点（不依赖 hands_tasks 打印块，裸 except 也吞不掉）：
        #   ARM 当帧 TIP + 段内 TIP 谷底；食/中独立弯度 ARM 值与段内 min/max（摆幅）
        self._rf_tip_arm = None
        self._rf_tip_min = None
        self._rf_idx_arm = None
        self._rf_mid_arm = None
        self._rf_idx_min = self._rf_idx_max = None
        self._rf_mid_min = self._rf_mid_max = None

    # ------------------------------------------------------------
    @staticmethod
    def _diag(msg: str) -> None:
        if TWO_FINGER_DIAG:
            print(f"[2F-DIAG] {msg}", flush=True)

    # ------------------------------------------------------------
    # ⭐ 2026-10-03 [2F-SEG] 段生命周期埋点（纯观测）
    def _emit_rf(self, result: str, r=None, **extra) -> None:
        _wall = int((_CLOCK() - self._arm_t) * 1000.0) if self._arm_t is not None else 0
        _d = {"result": result, "wall_ms": _wall, "none_n": self._rf_none_n,
              "entry_tip": self._rf_entry_tip, "entry_bend": self._rf_entry_bend,
              "arm_r": self._rf_arm_r, "min_r": self._rf_min_r, "r": r,
              "kind": self._rf_kind,
              "parts": self._last_parts, "gap": self._last_gap}
        _d.update(extra)
        self._rf_logs.append(_d)
        # ⭐ 2026-10-04 检测器本体直接打印（绕过外层队列/裸 except，任何结局都落日志）：
        #   校准"单指点击 vs 双指右键"的指尖距/弯度分布，无需任何专门操作
        def _f(_v):
            return f"{_v:.2f}" if isinstance(_v, (int, float)) else "n/a"
        _sw_i = (self._rf_idx_max - self._rf_idx_min
                 if self._rf_idx_min is not None and self._rf_idx_max is not None else None)
        _sw_m = (self._rf_mid_max - self._rf_mid_min
                 if self._rf_mid_min is not None and self._rf_mid_max is not None else None)
        _ptip = self._last_parts[3] if self._last_parts is not None else None
        print(f"[2F-END!] {result} 段{_wall}ms None{self._rf_none_n}帧 入口={self._rf_kind or '?'} "
              f"| 资格TIP{_f(self._rf_entry_tip)} ARM_TIP{_f(self._rf_tip_arm)} "
              f"段内TIP_min{_f(self._rf_tip_min)} 末帧TIP{_f(_ptip)} "
              f"| 合成r ARM{_f(self._rf_arm_r)} 谷底{_f(self._rf_min_r)} 末{_f(r)} "
              f"| 食弯 ARM{_f(self._rf_idx_arm)} 段内{_f(self._rf_idx_min)}~{_f(self._rf_idx_max)}"
              f"摆{_f(_sw_i)} "
              f"| 中弯 ARM{_f(self._rf_mid_arm)} 段内{_f(self._rf_mid_min)}~{_f(self._rf_mid_max)}"
              f"摆{_f(_sw_m)}",
              flush=True)

    def take_rf_logs(self) -> list:
        """取走并清空右键段/资格日志（hands_tasks 每帧取，纯观测）。"""
        _out = list(self._rf_logs)
        self._rf_logs.clear()
        return _out

    def _reset_rf_seg(self) -> None:
        self._rf_arm_r = None
        self._rf_min_r = None
        self._rf_none_n = 0
        self._rf_entry_tip = None
        self._rf_entry_bend = None
        self._rf_kind = ""
        self._rf_tip_arm = None
        self._rf_tip_min = None
        self._rf_idx_arm = None
        self._rf_mid_arm = None
        self._rf_idx_min = self._rf_idx_max = None
        self._rf_mid_min = self._rf_mid_max = None

    # ------------------------------------------------------------
    def reset(self) -> None:
        """双手同时在镜等全局禁用场景调用：清空全部进行中状态。"""
        self.state = _S.IDLE
        self.event_this_frame = E.NONE
        self._sm = None
        self._base = None
        self._maxr = None
        self._hist.clear()
        self._di_down = 0.0
        self._di_up = 0.0
        self._prev_r = None
        self._max_recent_r = None
        self._hold_sec = 0.0
        self._total_sec = 0.0
        self._stable_frames = 0
        self._cooldown_until = 0.0
        self._last_valid_t = None
        self._armed_by_delta = False
        self._none_since = None
        self._gap_break_frames = 0
        self._last_idx_bend = None
        self._last_mid_bend = None
        self._tip_break_frames = 0
        self._entry_t0 = None
        self._entry_ok = False
        self._entry_ok_at = None
        self._entry_grace_n = 0
        self._arm_t = None
        self.armed_this_frame = False
        self._reset_rf_seg()
        # right_count 保留（UI 统计不丢）

    @property
    def is_active(self) -> bool:
        """手势进行中（ARMED/HOLD/PRESSED）：hands_tasks 在此期间冻结左指检测器。"""
        return self.state in (_S.BEND_ARMED, _S.BEND_HOLD, _S.PRESSED)

    @property
    def freeze_left(self) -> bool:
        """左指冻结范围 = 手势进行中 + 冷却尾帧。
        ⭐ 冷却尾帧必须冻结：右键触发帧状态转 COOLDOWN，当帧/随后 2~3 帧的双指回弹 bend 值
        若漏喂给左指检测器，会被它当成一次独立食指勾按（虚构 Tap 挂起，之后与真左键配成误双击）。"""
        return self.state in (_S.BEND_ARMED, _S.BEND_HOLD, _S.PRESSED, _S.COOLDOWN)

    # ------------------------------------------------------------
    def _to_idle(self) -> None:
        self.state = _S.IDLE
        self._hold_sec = 0.0
        self._total_sec = 0.0
        self._stable_frames = 0
        self._armed_by_delta = False
        self._none_since = None
        self._gap_break_frames = 0
        self._tip_break_frames = 0
        self._entry_t0 = None
        self._entry_ok = False
        self._entry_ok_at = None
        self._entry_grace_n = 0
        self._hist.clear()
        self._di_down = 0.0
        self._di_up = 0.0
        self._reset_rf_seg()

    def _enter_cooldown(self) -> None:
        now = _CLOCK()
        self.state = _S.COOLDOWN
        # 旧"进入后数 4 次 update"= 4 个帧间隔 → deadline=now+COOLDOWN_SEC（None 帧也消耗）
        self._cooldown_until = now + self._p["COOLDOWN_SEC"]
        self._hold_sec = 0.0
        self._total_sec = 0.0
        self._stable_frames = 0
        self._armed_by_delta = False
        self._none_since = None
        self._gap_break_frames = 0
        self._tip_break_frames = 0
        # ⭐ 2026-09-18 真机修复：手势结束（右键触发/长按静默/PRESSED结束）→ 起势资格必须失效。
        #   旧实现此处不清 _entry_frames，一次右键后资格永久停在 3/3；之后单指勾按（TIP 全程张开
        #   1.3~1.7、DIP=1.0~1.17<1.30）凭旧资格走"低 bend 有资格放行"分支，全部误判成右键连发，
        #   且 freeze_left 把左键掐断。资格是一次性门票（与 _to_idle 对称清零，ARM 转换处也会消费）。
        self._entry_t0 = None
        self._entry_ok = False
        self._entry_ok_at = None
        self._entry_grace_n = 0
        # 推两个当前时刻近伸直值，让冷却后 IDLE 的差分立刻回 0（防连按残留，同左键手法）
        if self._base is not None and self._sm is not None:
            for _ in range(2):
                self._hist.append((now, self._sm / max(self._base, 1e-6)))
        self._reset_rf_seg()

    def _frame_dt(self, now: float) -> tuple[float, float]:
        """有效帧墙钟 dt 并推进时刻，返回 (dt_计时, dt_真实)：
          · dt_计时：钳 MAX_DT_SEC=2帧，供 hold/total/稳定门；
          · dt_真实：原始间隔，仅 _bend_ema_alpha 内部对 dt<=0/>0.5s 异常回退，供 EMA。
        首帧/黑帧恢复两者均=FRAME_SEC。COOLDOWN 与主路径共用同一口径。"""
        if self._last_valid_t is None:
            raw = FRAME_SEC
        else:
            raw = max(0.0, now - self._last_valid_t)
        self._last_valid_t = now
        return min(MAX_DT_SEC, raw), raw

    def _smooth(self, b: float, dt: float) -> float:
        """单条合成 bend 序列的 EMA 平滑 + 近期伸直高值 + 基线初始化（同左键）。
        ⭐ 2026-10-02 EMA 墙钟化（同左键 τ=50ms）：dt 为有效帧墙钟间隔。"""
        if self._sm is None:
            self._sm = float(b)
        else:
            alpha = _bend_ema_alpha(dt, self._p["BEND_EMA_TAU_SEC"])
            self._sm = alpha * float(b) + (1.0 - alpha) * self._sm
        if self._maxr is None:
            self._maxr = self._sm
        elif self._sm > self._maxr:
            self._maxr = self._sm
        else:
            self._maxr = 0.985 * self._maxr + 0.015 * self._sm
        if self._base is None:
            self._base = self._sm
        return float(self._sm)

    def _learn_baseline_idle(self) -> None:
        """IDLE 下双指近伸直时慢速学习基线（同左键规则）。"""
        b, base = self._sm, self._base
        if b is None or base is None:
            return
        if b > base * 0.95:
            self._base = BASELINE_ALPHA_IDLE * b + (1.0 - BASELINE_ALPHA_IDLE) * base
        if b > base * 0.97:
            self._base = BASELINE_ALPHA_FAST * b + (1.0 - BASELINE_ALPHA_FAST) * self._base

    def _compute_di(self, r: float, now: float) -> tuple[float, float]:
        # ⭐ 墙钟窗（含左边界点，对齐旧 win+1 点跨 win 个间隔）
        self._hist.append((now, float(r)))
        if len(self._hist) < 2:
            self._di_down, self._di_up = 0.0, 0.0
            return 0.0, 0.0
        win_sec = self._p["DI_WINDOW_SEC"]
        hist = [rv for t, rv in self._hist if _t_ge(t, now - win_sec)]
        if not hist:
            hist = [float(r)]
        neg = pos = 0.0
        for k in range(1, len(hist)):
            d = hist[k] - hist[k - 1]
            if d < 0:
                neg += -d
            elif d > 0:
                pos += d
        self._di_down = float(round(neg, 5))
        self._di_up = float(round(pos, 5))
        return self._di_down, self._di_up

    # ------------------------------------------------------------
    def update(self, two_finger_bend: float | None,
               together: bool | None, gap_r: float | None = None,
               gap_parts: tuple | None = None,
               idx_bend: float | None = None,
               mid_bend: float | None = None) -> int:
        """
        Args:
            two_finger_bend: 双指单元合成伸直度（四关节中点代入同公式，0.20弯~0.88直）；无效帧 None
            together:    【起势门控】TIP 3D 排是否并拢（TIP/米制掌宽≤TIP_ENTRY_R）；无法判定传 None。
                         手势中（ARMED/HOLD/PRESSED）不再使用本参数，改用 gap_parts 的 DIP 排门控
            gap_r:       指尖比值（日志用）
            gap_parts:   (MCP,PIP,DIP,TIP,spread) 3D 比值五值；MCP/PIP/spread 仅诊断，DIP 供保持门控
            idx_bend:    食指独立关节角弯度（0.20弯~0.88直）；仅资格帧近直门使用，None=不可算
            mid_bend:    中指独立关节角弯度（口径同上）；仅记录/埋点（10-03 手势同弯门已删）
        Returns:
            E.NONE / E.RIGHT_TAP
        """
        self.event_this_frame = E.NONE
        self.armed_this_frame = False
        now = _CLOCK()
        if gap_r is not None:
            self._last_gap = float(gap_r)
        if gap_parts is not None:
            self._last_parts = gap_parts
        # 独立弯度只在有效帧更新；None 帧保留上一值还是清空？清空——无效帧门控不允许用旧值裁决，
        #   手势中"任一指弯度缺失"帧对解散计数不贡献（下方判定显式跳过 None）。
        self._last_idx_bend = None if idx_bend is None else float(idx_bend)
        self._last_mid_bend = None if mid_bend is None else float(mid_bend)

        # ---- 冷却（墙钟：到点转 IDLE，None 帧也消耗；近伸直帧继续学基线，恢复无跳变）----
        if self.state == _S.COOLDOWN:
            # ⭐ 只有指尖并拢时合成 bend 才有物理意义；中指蜷着时中点是"半弯虚拟指"，不允许学基线
            if two_finger_bend is not None and together:
                _, _ema_dt = self._frame_dt(now)   # 冷却中无计时需求，只用真实间隔喂 EMA
                B = self._smooth(two_finger_bend, _ema_dt)
                if self._base is not None and B > self._base * 0.97:
                    self._learn_baseline_idle()
            if _t_ge(now, self._cooldown_until):
                self.state = _S.IDLE
            return E.NONE

        # ---- 无效帧（丢手/掌心不稳门控/并拢度不可算）：同左键清差分 + 宽限 ----
        if two_finger_bend is None or together is None:
            self._hist.clear()
            self._di_down = self._di_up = 0.0
            self._prev_r = None
            self._last_valid_t = None       # 冻结手势计时；恢复首帧 dt=FRAME_SEC
            if self.state == _S.IDLE:
                self._entry_t0 = None       # 等待中丢手：先并拢资格作废，须重新取得
                self._entry_ok = False
                self._entry_ok_at = None
                self._entry_grace_n = 0
            if self.state in (_S.BEND_ARMED, _S.BEND_HOLD):
                self._rf_none_n += 1
                if self._none_since is None:
                    self._none_since = now
                _none_elapsed = now - self._none_since
                # 旧第 1/5/9/13 个 None 帧打印（=0/4/8/12 间隔）
                _none_n = int(round(_none_elapsed / FRAME_SEC)) + 1
                if _none_n % 4 == 1 or _none_n == TWO_FINGER_NONE_GRACE_FRAMES + 1:
                    self._diag(f"手势中无效帧 {_none_n}/{TWO_FINGER_NONE_GRACE_FRAMES} "
                               f"({_none_elapsed:.2f}s/{TWO_FINGER_NONE_GRACE_SEC:.2f}s) 宽限"
                               f"（state={self._state_name()}）")
                # 旧 _none_frames>12：第 13 个 None 帧作废 = elapsed 12/30 当帧（>= 对齐）
                if _t_ge(_none_elapsed, TWO_FINGER_NONE_GRACE_SEC):
                    self._emit_rf("丢手超宽限作废")
                    self._to_idle()      # 手势中丢手超宽限 → 作废
                    self._diag("✘ 手势中丢手/门控超宽限 → 作废回 IDLE")
            elif self.state == _S.PRESSED:
                self._emit_rf("长按丢手结束")
                self._enter_cooldown()   # 长按时丢手=释放（v1 无事件）
                self._diag("双指长按中丢手 → 静默结束")
            return E.NONE

        # ---- 手势中的门控：DIP 排（v7 深弯兜底）+ TIP 排（2026-10-04 指尖距持续门）----
        #   旧架构 ARM 后只看 DIP（手指中部，单指勾按也不动，无区分力），指尖距信号丢失；
        #   现两个量独立去抖计数：任一连续 2 帧超门即解散。parts 缺失（None）按放行处理，
        #   不与张开混计（与旧 DIP 语义一致；无效帧另有 None 宽限路径）。
        _dip_now = (self._last_parts[2]
                    if (self._last_parts and len(self._last_parts) >= 4) else None)
        _tip_now = (self._last_parts[3]
                    if (self._last_parts and len(self._last_parts) >= 4) else None)
        _dip_ok = bool(_dip_now is None or _dip_now <= TWO_FINGER_DIP_HOLD_R)
        _tip_ok = bool(_tip_now is None or _tip_now <= TWO_FINGER_TIP_HOLD_R)
        _hold_ok = _dip_ok and _tip_ok

        # ---- IDLE 起势门控（v7）：先在高 bend 指尖并拢 ≥N 帧（顺序门），之后才准下弯 ----
        if self.state == _S.IDLE:
            _high_bend = two_finger_bend >= TWO_FINGER_ENTRY_BEND
            # ⭐ 2026-10-03 双指独立近直门：并拢必须发生在食/中两指都近直时。
            #   单指勾按的指尖扫过中尖距离带发生在食指【已经弯着】之后 → 此帧不累计资格。
            #   弯度缺失帧按失败处理（合成 bend 有效时独立弯度同源必有效，仅防御性）。
            _fingers_straight = (self._last_idx_bend is not None
                                 and self._last_mid_bend is not None
                                 and self._last_idx_bend >= RF_ENTRY_FINGER_R
                                 and self._last_mid_bend >= RF_ENTRY_FINGER_R)
            if _high_bend and together and _fingers_straight:
                # 高 bend + TIP 并拢 + 双指近直：连续段第1帧起计时；资格只在【并拢帧】上确认满 2 间隔
                #   （旧第3个并拢帧 entry=3；若第3帧已是低 bend 勾弯帧，旧版拒绝——确认必须落在并拢帧）
                if self._entry_t0 is None:
                    self._entry_t0 = now
                if not self._entry_ok and _t_ge(now - self._entry_t0, TWO_FINGER_ENTRY_SEC):
                    self._entry_ok = True
                    self._entry_ok_at = now
                    self._entry_grace_n = 0
                    # ⭐ [2F-ENTRY] 资格取得即记录（纯观测）：看单指点击时是否"自然并拢"误得资格
                    self._rf_entry_tip = (self._last_parts[3]
                                          if self._last_parts and len(self._last_parts) >= 4
                                          else self._last_gap)
                    self._rf_entry_bend = float(two_finger_bend)
                    self._rf_logs.append({
                        "result": "起势资格", "wall_ms": 0, "none_n": 0,
                        "entry_tip": self._rf_entry_tip,
                        "entry_bend": self._rf_entry_bend,
                        "arm_r": None, "min_r": None, "r": None, "kind": "",
                        "parts": self._last_parts, "gap": self._last_gap})
                # 直指并拢帧：宽限按【连续弯指帧】计数，回到直指即归零
                self._entry_grace_n = 0
                _has_entry = self._entry_ok
            elif _high_bend and not (together and _fingers_straight):
                # 高 bend 但不满足起势几何，分两类：
                # ① 指尖张开（单指待机/张开手）→ 任何时候资格立即作废，bend 不喂状态机；
                # ② 仍 TIP 并拢但任一指已弯：资格【已确认】后的头 GRACE_FRAMES 个这种帧 =
                #    两指开弯差 1~2 帧的正常过渡（中点合成 bend 几何滞后仍 ≥0.55），保留资格
                #    （bend 仍不喂），等中点 bend 跌破 0.55 的勾弯帧正常消费；
                #    累计中未确认、或连续第 3 个弯指帧中点 bend 还未跌破 0.55（保持弯指姿势而非
                #    勾按动作）→ 照旧作废。
                if (together and self._entry_ok
                        and self._entry_grace_n < TWO_FINGER_ENTRY_GRACE_FRAMES):
                    self._entry_grace_n += 1
                    _g_age = (0.0 if self._entry_ok_at is None
                              else now - self._entry_ok_at)
                    _iv = (f"{self._last_idx_bend:.2f}"
                           if self._last_idx_bend is not None else "n/a")
                    _mv = (f"{self._last_mid_bend:.2f}"
                           if self._last_mid_bend is not None else "n/a")
                    print(f"[2F-GRACE] 开弯过渡宽限 {self._entry_grace_n}/"
                          f"{TWO_FINGER_ENTRY_GRACE_FRAMES}：资格确认后{_g_age * 1000:.0f}ms "
                          f"遇高bend弯指帧（合成bend{float(two_finger_bend):.2f}≥"
                          f"{TWO_FINGER_ENTRY_BEND}，食{_iv} 中{_mv} 直门{RF_ENTRY_FINGER_R}，"
                          f"TIP仍并拢）→ 保留资格等勾弯帧", flush=True)
                    return E.NONE
                _rf_reason = "指尖未并拢(3D)" if not together else "并拢时手指已弯(超宽限)"
                self._entry_t0 = None
                self._entry_ok = False
                self._entry_ok_at = None
                self._entry_grace_n = 0
                self._hist.clear()
                self._di_down = self._di_up = 0.0
                self._prev_r = None
                if TWO_FINGER_DIAG:
                    self._diag_geom_n += 1
                    if self._diag_geom_n % 30 == 1:
                        _tipv = (self._last_parts[3] if self._last_parts else -1.0)
                        print(f"[2F-DIAG-HB] state=IDLE {_rf_reason} "
                              f"TIP={_tipv:.3f}✗(≤{TWO_FINGER_TIP_ENTRY_R}) "
                              f"idx={self._last_idx_bend if self._last_idx_bend is not None else -1:.3f} "
                              f"mid={self._last_mid_bend if self._last_mid_bend is not None else -1:.3f}"
                              f"(直门{RF_ENTRY_FINGER_R}) "
                              f"parts[MCP,PIP,DIP,TIP,spr]={self._last_parts} bend={two_finger_bend:.3f}"
                              f"（bend 不喂状态机）", flush=True)
                return E.NONE
            else:
                # 低 bend 帧（勾弯进行中）：无资格 = 单指点击"弯到底才碰巧并拢" → 拒绝；
                # 有资格 = 正常右键勾弯，TIP 在深弯会炸开，本帧起忽略 TIP（保持交给 DIP 门控）
                #   资格保留语义同旧 entry_frames 冻结=3：确认过就一直有效（直到消费/中断）
                _has_entry = self._entry_ok
                if not _has_entry:
                    self._hist.clear()
                    self._di_down = self._di_up = 0.0
                    self._prev_r = None
                    # ⭐ 基线/平滑上下文一并作废：否则回弹帧用旧基线算出低 r，会被 down_abs
                    #   误判为"新起弯"（先弯后并拢 → 回弹，不是新的勾按）
                    self._sm = None
                    self._base = None
                    self._max_recent_r = None
                    if TWO_FINGER_DIAG:
                        self._diag_geom_n += 1
                        if self._diag_geom_n % 15 == 1:
                            _el = 0.0 if self._entry_t0 is None else now - self._entry_t0
                            print(f"[2F-DIAG] ✗ 先弯后并拢（bend={two_finger_bend:.3f}<{TWO_FINGER_ENTRY_BEND}，"
                                  f"资格 {_el:.2f}s/{TWO_FINGER_ENTRY_SEC:.2f}s）→ 判单指点击，右键不起势 "
                                  f"parts={self._last_parts}", flush=True)
                    return E.NONE

        # ---- 有手：墙钟 dt（计时钳 2 帧；EMA 吃真实间隔，保证低帧率不变性）----
        dt, _ema_dt = self._frame_dt(now)
        self._none_since = None

        # ---- 有手：双指单元合成 bend 单条 EMA 平滑（墙钟化，τ 同左键 50ms）----
        B = self._smooth(two_finger_bend, _ema_dt)
        r = B / max(self._base, 1e-6)        # 合成序列：双指单元一起弯 r 才降（并拢保证同步）

        # ⭐ [2F-SEG] 段内合成 r 谷底（纯观测，ARM 帧自身在转态处另设）
        if self.state in (_S.BEND_ARMED, _S.BEND_HOLD, _S.PRESSED):
            if self._rf_min_r is None or r < self._rf_min_r:
                self._rf_min_r = r
            # ⭐ 2026-10-04 段内 TIP 谷底 + 食/中弯度 min/max（检测器本体直接埋点）
            if self._last_parts is not None and len(self._last_parts) >= 4:
                _tipv = float(self._last_parts[3])
                if self._rf_tip_min is None or _tipv < self._rf_tip_min:
                    self._rf_tip_min = _tipv
            if self._last_idx_bend is not None:
                _v = self._last_idx_bend
                self._rf_idx_min = _v if self._rf_idx_min is None else min(self._rf_idx_min, _v)
                self._rf_idx_max = _v if self._rf_idx_max is None else max(self._rf_idx_max, _v)
            if self._last_mid_bend is not None:
                _v = self._last_mid_bend
                self._rf_mid_min = _v if self._rf_mid_min is None else min(self._rf_mid_min, _v)
                self._rf_mid_max = _v if self._rf_mid_max is None else max(self._rf_mid_max, _v)

        # ---- 合成序列的 prev / 近期伸直高值（右键自己的绝对兜底用；左键无近期伸直高值机制）----
        prev_r = self._prev_r
        if self._max_recent_r is None:
            self._max_recent_r = r
        elif r > self._max_recent_r:
            self._max_recent_r = r
        else:
            self._max_recent_r = 0.985 * self._max_recent_r + 0.015 * r

        di_down, di_up = self._compute_di(r, now)
        self._prev_r = r

        enter = self._p["ENTER_BEND_RATIO"]
        exit_r = self._p["EXIT_BEND_RATIO"]
        drop = 0.0 if self._max_recent_r is None else \
            (self._max_recent_r - r) / max(self._max_recent_r, 1e-6)

        # ---- down/up 判据：与左键模板逐字同公式（作用在合成 r 上）----
        down_delta = (di_down >= self._p["TAP_DOWN_DI_SUM"]) and (
            r <= min(self._p["MIN_ABS_BEND_RATIO_FOR_DELTA"], enter + 0.06))
        down_abs = (r < enter) and (drop > self._p["MIN_BEND_DROP"])
        down_trigger = down_delta or down_abs

        # 松开：进入方式决定优先级；差分（回弹变化量）与绝对回线互为兜底 OR —— 完全照搬左键
        # ⭐ 2026-09-19 差分松开加对称位置门 r>=enter（手指仍深弯时 Σ上 不允许判松开，同左键注释）
        if self._armed_by_delta:
            up_trigger = ((di_up >= self._p["TAP_UP_DI_SUM"]) and (r >= enter)) or (r >= exit_r)
        else:
            up_trigger = (r >= exit_r) or ((di_up >= self._p["TAP_UP_DI_SUM"]) and (r >= enter))

        # ---- 心跳：每 30 个有效帧一行（含四排指距+同步超张量，仅观察；判定只调 root 阈值）----
        self._diag_n += 1
        if TWO_FINGER_DIAG and self._diag_n % 30 == 0:
            _g = f"{self._last_gap:.3f}" if self._last_gap is not None else " n/a "
            if self._last_parts is not None and len(self._last_parts) >= 4:
                _parts = (f"[MCP={self._last_parts[0]:.3f} PIP={self._last_parts[1]:.3f} "
                          f"DIP={self._last_parts[2]:.3f} TIP={self._last_parts[3]:.3f}")
                if len(self._last_parts) >= 5:
                    _parts += f" spread={self._last_parts[4]:.3f}"
                _parts += "]"
            else:
                _parts = "[parts n/a]"
            print(f"[2F-DIAG-HB] state={self._state_name():>4} gap={_g:>5} "
                  f"TIP起势={'Y' if together else 'N'}(≤{TWO_FINGER_TIP_ENTRY_R}) "
                  f"DIP保持={'Y' if _dip_ok else 'N'}(≤{TWO_FINGER_DIP_HOLD_R}) "
                  f"TIP保持={'Y' if _tip_ok else 'N'}(≤{TWO_FINGER_TIP_HOLD_R}) "
                  f"资格={(0.0 if self._entry_t0 is None else now-self._entry_t0):.2f}s"
                  f"/{TWO_FINGER_ENTRY_SEC:.2f}s {_parts} "
                  f"r={r:.3f} enter={enter:.3f} exit={exit_r:.3f} "
                  f"di↓={di_down:.3f}/{self._p['TAP_DOWN_DI_SUM']:.3f} "
                  f"di↑={di_up:.3f}/{self._p['TAP_UP_DI_SUM']:.3f} drop={drop:.2f}", flush=True)

        # ================= 状态流转（左键模板）+ 右键分阶段门控 =================
        if self.state == _S.IDLE:
            # 走到这里 TIP 必并拢且"先并拢"顺序门已过；学习当前姿态基线（微弯也可，不要求伸直）
            self._learn_baseline_idle()
            if down_trigger and _has_entry:
                self.state = _S.BEND_ARMED
                self.armed_this_frame = True
                # ⭐ 一次性门票：IDLE→ARMED 立即消费资格。手势结束（触发/作废/尖峰/丢手）
                #   回 IDLE 后必须重新在高 bend 并拢满 TWO_FINGER_ENTRY_SEC 才能再起势——
                #   杜绝"一次右键后单指勾按凭旧资格连发右键"（2026-09-18 真机故障）
                self._entry_t0 = None
                self._entry_ok = False
                self._entry_ok_at = None
                self._entry_grace_n = 0
                # ARM 当帧 hold/total=0.0（旧帧版当帧=1，阈值按 N-1 间隔换算）
                self._hold_sec = 0.0
                self._total_sec = 0.0
                self._stable_frames = 0
                self._gap_break_frames = 0
                self._tip_break_frames = 0
                self._last_valid_t = now
                self._armed_by_delta = bool(down_delta and not down_abs)
                self._arm_t = now
                # ⭐ [2F-SEG] 段起点（纯观测）
                self._rf_arm_r = float(r)
                self._rf_min_r = float(r)
                self._rf_none_n = 0
                self._rf_kind = "差分" if (down_delta and not down_abs) else "绝对"
                # ⭐ 2026-10-04 ARM 当帧几何快照（检测器本体直接打印，校准指尖距门限用）
                self._rf_tip_arm = (float(self._last_parts[3])
                                    if self._last_parts is not None
                                    and len(self._last_parts) >= 4 else None)
                self._rf_tip_min = self._rf_tip_arm
                self._rf_idx_arm = self._last_idx_bend
                self._rf_mid_arm = self._last_mid_bend
                self._rf_idx_min = self._rf_idx_max = self._last_idx_bend
                self._rf_mid_min = self._rf_mid_max = self._last_mid_bend
                _tip_a = f"{self._rf_tip_arm:.2f}" if self._rf_tip_arm is not None else "n/a"
                _etip_a = f"{self._rf_entry_tip:.2f}" if self._rf_entry_tip is not None else "n/a"
                _idx_a = f"{self._rf_idx_arm:.2f}" if self._rf_idx_arm is not None else "n/a"
                _mid_a = f"{self._rf_mid_arm:.2f}" if self._rf_mid_arm is not None else "n/a"
                print(f"[2F-ARM!] 右检测器起ARM 入口={self._rf_kind} "
                      f"资格TIP{_etip_a} ARM当帧TIP{_tip_a}(门{TWO_FINGER_TIP_ENTRY_R}) "
                      f"合成r{float(r):.2f} | 食弯{_idx_a} 中弯{_mid_a}(资格直门{RF_ENTRY_FINGER_R})",
                      flush=True)
                _g = f"{self._last_gap:.3f}" if self._last_gap is not None else "n/a"
                self._diag(f"▶ 起弯 ARMED：gap={_g} r={r:.3f}"
                           f"（入口={'差分' if self._armed_by_delta else '绝对'}，先并拢资格已确认）")

        elif self.state == _S.BEND_ARMED:
            # 右键门控：DIP 排连续失效 N 帧才判张开（单帧抖动不解散）
            if not _dip_ok:
                self._gap_break_frames += 1
                if self._gap_break_frames >= TWO_FINGER_GAP_BREAK_FRAMES:
                    _g = f"{self._last_gap:.3f}" if self._last_gap is not None else "n/a"
                    _bf = self._gap_break_frames
                    self._max_recent_r = r   # 作废后不许弯着不动直接重新起势，须重新做一次弯屈动作
                    self._emit_rf("ARMED_DIP解散", r, dip=_dip_now)
                    self._to_idle()
                    self._diag(f"✘ ARMED 连续{_bf}帧 DIP 张开/不同步 DIP={_dip_now:.3f} → 解散")
                    return E.NONE
            else:
                self._gap_break_frames = 0
            # ⭐ 2026-10-04 TIP 指尖距持续门：手势中两指尖连续 2 帧张开 = 不是并拢双指勾按。
            #   替代 10-03 双指同弯门（真机证伪：真右键中弯摆幅仅 0.02~0.07，弯度不可分）。
            if not _tip_ok:
                self._tip_break_frames += 1
                if self._tip_break_frames >= TWO_FINGER_GAP_BREAK_FRAMES:
                    self._max_recent_r = r
                    self._emit_rf("ARMED_TIP解散", r, tip=_tip_now, dip=_dip_now)
                    self._to_idle()
                    self._diag(f"✘ ARMED 连续{self._tip_break_frames}帧两指尖张开 "
                               f"TIP={_tip_now:.3f}（门{TWO_FINGER_TIP_HOLD_R}）→ 解散")
                    return E.NONE
            else:
                self._tip_break_frames = 0
            # 模板第 2 帧确认：仍处于弯曲（残留 Σ下 或 r 仍在弯线附近）
            still_down = ((di_down >= max(0.0, self._p["TAP_DOWN_DI_SUM"] * 0.35))
                          or (r < enter + 0.03))
            if still_down:
                # ARMED 中每个有效帧都累计（无稳定门）；旧第2帧 hold=2>=MIN_BEND_HOLD
                self._hold_sec += dt
                self._total_sec += dt
                if _t_ge(self._hold_sec, self._p["MIN_HOLD_SEC"]):
                    self.state = _S.BEND_HOLD
                    self._diag(f"✔ ARMED 确认通过 → BEND_HOLD（r={r:.3f}）")
            else:
                self._emit_rf("尖峰拒绝", r, di_down=di_down)
                self._to_idle()                # 单帧尖峰（与左键同处理）
                self._diag(f"✘ ARMED 尖峰拒绝(第2帧未保持弯曲) r={r:.3f} Σ下={di_down:.3f} → IDLE")

        elif self.state == _S.BEND_HOLD:
            self._total_sec += dt

            # 右键门控：DIP 排连续 N 帧张开 → 作废（EXIT 之前先过门控）
            if not _dip_ok:
                self._gap_break_frames += 1
                if self._gap_break_frames >= TWO_FINGER_GAP_BREAK_FRAMES:
                    _g = f"{self._last_gap:.3f}" if self._last_gap is not None else "n/a"
                    _bf = self._gap_break_frames
                    self._max_recent_r = r   # 作废后不许弯着不动直接重新起势
                    self._emit_rf("HOLD_DIP作废", r, dip=_dip_now)
                    self._to_idle()
                    self._diag(f"✘ HOLD 连续{_bf}帧 DIP 张开/不同步 DIP={_dip_now:.3f} gap={_g} → 作废")
                    return E.NONE
            else:
                self._gap_break_frames = 0

            # ⭐ 2026-10-04 TIP 指尖距持续门（HOLD 中持续生效，替代被证伪的同弯门）：
            #   两指尖连续 2 帧张开 = 并拢意图已消失，绝不送右键。
            if not _tip_ok:
                self._tip_break_frames += 1
                if self._tip_break_frames >= TWO_FINGER_GAP_BREAK_FRAMES:
                    self._max_recent_r = r
                    self._emit_rf("HOLD_TIP作废", r, tip=_tip_now, dip=_dip_now)
                    self._to_idle()
                    self._diag(f"✘ HOLD 连续{self._tip_break_frames}帧两指尖张开 "
                               f"TIP={_tip_now:.3f}（门{TWO_FINGER_TIP_HOLD_R}）→ 作废")
                    return E.NONE
            else:
                self._tip_break_frames = 0

            # ---------------- EXIT 优先于 LONG_PRESS（同左键）----------------
            #   起势顺序门保证了双指同步意图，up_trigger 即双指一起回弹 → 直接判短按/长按
            if up_trigger:
                # 旧帧版：total<=TAP_MAX 或 hold<=LONG_PRESS；阈值 (N-1)/30 与 22/30（hold 严格<）
                is_tap = (_t_le(self._total_sec, self._p["TAP_MAX_SEC"])
                          or _t_lt(self._hold_sec, self._p["LONG_PRESS_SEC"]))
                if is_tap:
                    self.event_this_frame = E.RIGHT_TAP
                    self.right_count += 1
                    _tot, _hold = self._total_sec, self._hold_sec
                    self._emit_rf("右键送达", r, total=_tot, hold=_hold, di_up=di_up)
                    self._enter_cooldown()
                    self._diag(f"★★ 触发 RIGHT_TAP 右键！总时长={_tot:.3f}s 保持={_hold:.3f}s "
                               f"r={r:.3f}（变化量松开 Σ↑={di_up:.3f}）")
                else:
                    # 双指长按：v1 不映射右键拖拽 → 静默
                    _tot = self._total_sec
                    self._emit_rf("长按静默", r, total=_tot, hold=self._hold_sec)
                    self._enter_cooldown()
                    self._diag(f"● 双指长按后释放（总时长{_tot:.3f}s>窗）→ 静默不发右键")
                return self.event_this_frame

            # ---------------- 弯曲深度稳定 → 累计保持时长（同左键：基于 prev_r 的逐帧差值）----------------
            is_stable = True
            if prev_r is not None:
                delta = prev_r - r           # 正值 = r 继续变小 = 还在下压
                if delta > self._p["BEND_STABLE_THRESH"]:
                    is_stable = False
            if is_stable:
                self._stable_frames += 1
                if self._stable_frames >= self._p["BEND_STABLE_MIN_HOLD"]:
                    self._hold_sec += dt
            else:
                self._stable_frames = 0
            if _t_ge(self._hold_sec, self._p["LONG_PRESS_SEC"]):
                self.state = _S.PRESSED       # 长按按住中（v1 静默，等释放）
                self._diag(f"● 双指保持超 {self._p['LONG_PRESS_SEC']:.3f}s → PRESSED 长按"
                           f"（v1 静默，释放也不发右键）")

        elif self.state == _S.PRESSED:
            # v1 无右键拖拽：DIP/TIP 张开 或 双指回弹 即静默结束
            if not _hold_ok or up_trigger:
                self._emit_rf("长按结束", r, dip=_dip_now, tip=_tip_now)
                self._enter_cooldown()
                self._diag("PRESSED 结束（DIP/TIP张开/回弹）→ 冷却，无事件")

        return self.event_this_frame

    def _state_name(self) -> str:
        return {_S.IDLE: "IDLE", _S.BEND_ARMED: "ARMED", _S.BEND_HOLD: "HOLD",
                _S.PRESSED: "PRS", _S.COOLDOWN: "COOL", _S.TAPPED: "TAP",
                _S.WAIT_DBL: "WAIT"}.get(self.state, str(self.state))

    # ------------------------------------------------------------
    def metrics(self) -> dict:
        return {
            "state":        self.state,
            "active":       self.is_active,
            "armed":        self.armed_this_frame,
            "right_count":  self.right_count,
            "r":            round(self._sm / max(self._base, 1e-6), 3)
                            if (self._sm is not None and self._base is not None) else None,
            "cooldown_left": round(max(0.0, self._cooldown_until - _CLOCK()), 3),
        }
