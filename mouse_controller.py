import time
from collections import deque
import numpy as np
import pyautogui

"""
鼠标事件映射 + AR眼镜级 5 层悬停平滑 + 防抖
================================================

5 层流水线（替代原单一 EMA）：
  ① 5 帧滑动中值滤波 — 杀死 MediaPipe 偶发 1-2 帧尖峰直接锁死光标不动
  ② 静止方差锁        — 近 8 帧位移协方差 < 阈值 → 
  ③ 1€ Filter         — 速度自适应 α（移动慢=超平滑，移动快=超跟手）
  ④ Sigmoid 响应曲线  — 大动灵敏、小动不动
  ⑤ 手势模式切换      — POINT=悬停(最大平滑+大死区) / OPEN=浏览
                        ✨ 用户要求：FIST=取消所有指令/无指令（不再用于拖拽，也不触发任何动作）
                        TAP/PINCH/PEACE 前 → 锁定位置 3 帧，防止点击瞬间抖动漂移

同时保留：
  - 手势防抖 HOLD_FRAMES
  - 边缘死区（通过 ROI_SCALE 实现的可调节中心裁剪）
  - 点击稳定：食指 Tap-Click（仅手指屈伸，不依赖手掌整体运动）
"""

# ---------------- 防抖参数 ----------------
# HOLD_FRAMES：用户要求恢复原设计=4帧（点击类手势保持确认）
HOLD_FRAMES = 4
# 画面 → 屏幕 的"中心 ROI 裁剪缩放比"（用户可调参数）：
#   1.0 = 画面 100% 映射全屏（=旧 EDGE_MARGIN=0）
#   0.85 = 画面中心 85% 的矩形区域 映射 全屏（推荐默认）
#          → 好处：掌心到达屏幕边缘后，手还能继续向摄像头画面四周移动 ~7.5% 的范围
#            保证"手指可见"，而不是掌心刚碰画面边缘就锁死、手指还在抖动但光标不动
#   下限 0.3（画面中心 30% 映射全屏，预留 70% 活动范围给手指伸展）
# 注：此参数取代原先的 EDGE_MARGIN（EDGE_MARGIN 不再硬编码）
DEFAULT_ROI_SCALE = 0.70   # ✨ 原 0.85 再往内收缩：中心 70% 映射全屏，每边留白 15%（掌心活动区更小=更不易跑出镜头边缘，避免无法识别到掌心/整只手）
ROI_SCALE_MIN = 0.30
ROI_SCALE_MAX = 1.00
# ⭐ 2026-08-24 新增："安全子 ROI" 等比内缩比例（需求 1：避免指尖超出显示框边缘）
#   语义：在「--roi-scale 定义的原始 ROI 长方形」基础上，**再按中心等比乘 ROI_SAFE_SHRINK 内缩**，
#         作为「鼠标映射真正生效的区域」+「UI 上绘制的虚线矩形」。
#   例：roi_scale=0.70 → 原始 ROI 左右各 margin 15%；再 SAFE_SHRINK=0.82 → 安全子 ROI 每边再加 padding ≈9%
#       总每边留白 15%+9%×0.70≈21% → 掌心移到虚线框边缘时，整只手掌 + 向上伸展的指尖距离画面边缘还有缓冲，
#       不会顶出摄像头边界 → MediaPipe 仍能完整看到所有 landmark = 不会丢识别。
#   SAFE_SHRINK 越小=虚线框越小=指尖越不容易超，但手要覆盖全屏的动作幅度越大（反向=ROI 越小）。
ROI_SAFE_SHRINK = 0.82
ROI_SAFE_SHRINK_MIN = 0.50
ROI_SAFE_SHRINK_MAX = 1.00
# ✨ ROI 垂直偏移（新增 2026-08-20）
#   ⚠️ 只改 Y 方向位置，X 方向宽度和 Y 方向高度 100% 保持 roi_scale 原尺寸不变（严禁擅自改成正方形！）
#   语义：roi_offset_y ∈ [-1.0, +1.0]，控制「长方形 ROI 区域」在画面高度方向的位置：
#     长方形尺寸（原逻辑完全保留）：
#       宽 = frame_w - (1-roi_scale)*min(frame_w, frame_h)
#       高 = roi_scale * min(frame_w, frame_h)
#       水平方向：永远左右对称 margin=(1-roi_scale)*0.5*min(w,h)（和你之前版本一模一样）
#     偏移：
#       -1.0  → ROI 贴画面顶（margin_y_top=0，手指可能超顶）
#        0.0  → ROI 垂直居中（老默认）
#       +0.5  → ROI 往下偏（**新默认**，画面顶部多留 50% 可分配 Y margin 给向上伸展的食指/中指，避免顶部手指超出摄像头上边界）
#       +1.0  → ROI 贴画面底
DEFAULT_ROI_OFFSET_Y = 0.62  # ✨ 2026-08-25 组合根治B1：从0.50下移至0.62（=ROI整体向画面底部多挪12%）
#   目的：掌心锚点顶到安全矩形最上沿时，腕#0/食指MCP#5/小指MCP#17/食指TIP#8 整只手仍在画面内，不被上边缘clip
#   公式：top_margin = avail_y_margin * (0.5 + 0.5 * oy)；oy+0.50→顶空75%，oy+0.62→顶空81%，下方缩12%，X方向不变
ROI_OFFSET_Y_MIN = -1.0
ROI_OFFSET_Y_MAX = +1.0

# ---------------- AR Smoother 5 层参数 ----------------
# 1) 滑动中值窗口
MEDIAN_WINDOW = 5
# 2) 静止方差锁：近 N 帧屏幕像素位移的协方差 (px^2) 低于此值 → 锁死光标不动
# ⭐ 2026-08-24 用户反馈「难以稳定」：
#   LOCK_WINDOW 8→10（取更多帧算方差，统计更稳，不会因为 2-3 帧抖就判在动）
#   LOCK_THRESH_BROWSE 3.0→4.5（OPEN 浏览更宽松 = 4.5px² ≈ 2.1px 单维抖动就允许锁死，之前要 1.7px=更难锁）
#   LOCK_THRESH_HOVER 0.80→1.20（POINT 悬停也适当放宽，1.1px 单维就能锁）
LOCK_WINDOW = 10
#  不同模式下的锁死阈值 (px^2)，越大=越容易锁死=越稳
LOCK_THRESH_HOVER  = 1.20       # POINT 悬停：放宽到 1.2px²≈1.1px单维
LOCK_THRESH_BROWSE = 4.50       # OPEN 浏览：放宽到 4.5px²≈2.1px单维
LOCK_THRESH_DRAG   = 99999.0    # （保留兼容定义，但 FIST 不再用于拖拽）
# 3) 1€ Filter 核心参数（不同模式下 fc_min / beta 组合）
#    fc_min = 最小截止频率(Hz)，越小→慢动时越平滑
#    beta   = 速度自适应强度，越大→快动时越跟手
MODE_PARAMS = {
    # 手势名    (fc_min_Hz, beta,  deadzone_px, response_k)
    "POINT":  (0.30,  0.0040, 2.8, 3.0),   # 悬停：最强平滑+最大死区+柔响应
    "OPEN":   (0.80,  0.0100, 1.2, 1.6),   # 浏览：平衡
    # ✨ 用户要求：FIST = 取消所有指令（不再用于拖拽），因此参数用 OTHER 中性（实际 FIST 分支不触发任何动作，参数只是占位避免 KeyError）
    "FIST":   (1.00,  0.0150, 1.5, 1.4),
    "OTHER":  (1.00,  0.0150, 1.5, 1.4),   # 兜底：同浏览略强
    "PEACE":  (1.20,  0.0150, 1.0, 1.4),   # 点击前手势（中等）
    "PINCH":  (1.20,  0.0150, 1.0, 1.4),   # 同上
    "ROCK":   (1.20,  0.0150, 1.0, 1.4),
    "THUMB":  (1.20,  0.0150, 1.0, 1.4),
}
DEFAULT_PARAMS = MODE_PARAMS["OTHER"]
# 4) Sigmoid 响应：已通过 response_k (上) 控制，见 ARSmoother._sigmoid_gain
# 5) 点击前位置锁定帧数（防点击瞬间抖动漂移）
CLICK_FREEZE_FRAMES = 3

# cursor 模式
CURSOR_MODE_SMOOTH   = "smooth"    # 当前 AR 5 层平滑（悬停稳、但有滞后）
CURSOR_MODE_ABSOLUTE = "absolute"  # ✨ 掌心绝对坐标（方案A 3层：中值杀尖峰 + Sigmoid微移阻尼 + 静止方差锁，兼顾稳+跟手）
CURSOR_MODE_DIRECT   = "direct"    # ⭐ 2026-08-22 新增：严格跟随 UI 掌心绿圈——只保留 3 帧中值杀 1-2 帧飞点，无 Sigmoid 阻尼、无静止锁，无论手掌如何翻转都 1:1 跟手不卡顿

# absolute 模式下为了避免 MediaPipe 1~2 帧偶发尖峰跳到屏幕角落，保留一个轻量的 3 帧中值
ABS_MEDIAN_WINDOW = 3

# ✨✨ 方案 A 新增：absolute 模式「大范围跟手 + 小范围不晃」的 3 层流水线参数（只作用在 absolute 模式，不影响 smooth）
#   Layer 2 Sigmoid：微移阻尼增益（响应曲线拐点，k 越大=拐点越陡=越严格压制小抖动）
#     ⚠️ 2026-08-22 复原（用户确认 Sigmoid/静止锁不会造成卡顿，参数应与工程约定一致，不做折中放宽）
#       参数：MIN_G=0.20 / INFLECT=11px / K=0.35（之前的官方 baseline）
#         Δ=1px → 增益≈0.22（阻尼 4.5 倍，纯噪声 1px 抖→0.22px）
#         Δ=3px → 增益≈0.25（阻尼 4 倍）
#         Δ=5px → 增益≈0.29（阻尼 3.4 倍 = 用户之前约定的 5px 压缩边界）
#         Δ=11px→ 增益≈0.60（拐点中点，阻尼 1.7 倍）
#         Δ>20px→ 增益→1.0（0 衰减 = 大范围跟手指哪打哪）
#   ⭐ 2026-08-24 用户反馈「难以稳定」：
#     · INFLECTION_PX 11.0 → 14.0（拐点右移，0~14px 的小/中幅度移动都保持更强阻尼，移动不抖=更易稳定；18px+ 才开始近 100% 跟手=大范围跟手不丢）
#     · ABS_LOCK_WINDOW 6→10（取帧更多=方差统计更稳，1-2 帧偶发飞点不会破锁）
#     · ABS_LOCK_THRESH 3.5→5.6（σ<5.6px 就锁=更容易锁；单维 2.4px 抖动就锁，之前是 1.9px=更难锁）
#     · UNLOCK=5.6*1.6=9.0px（累计位移超过 9px 才解锁，慢移动解锁更"果断"=不拖泥带水；RELOCK_COOLDOWN 同步加大=解锁后 20 帧不重锁=移动连续）
ABS_SIGMOID_K     = 0.35
ABS_SIGMOID_MIN_G = 0.20  # 最小增益（Δ→0）：工程约定 baseline
#   Layer 3 静止方差锁：连续 N 帧内，掌心样本方差 σ² < threshold² → 硬锁死光标到一个点（钉住=完全不漂移）
ABS_LOCK_WINDOW   = 10
ABS_LOCK_THRESH   = 5.6

# pyautogui 安全设置
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.0

# ⭐ OS 双击模式（OS_DOUBLE_CLICK=True）下，第二次点击坐标吸附到第一击：
#   OS 双击位置容差只有 ±2px（SM_CXDOUBLECLK=4），两次手势落点漂移常达 10~30px 必然失败。
#   真双击语义上就是"同一个目标连点两下"，因此 OS 双击时间窗内、落点漂移 ≤ 本阈值时，
#   把第二击坐标强制吸到第一击精确像素（Δpos=0，稳过位置门槛）；超过阈值说明是两个不同
#   目标的连续单击，不吸附。单位：屏幕像素。
TAP_DBL_SNAP_PX = 60
# ⭐ 慢节奏双击补救（OS_DOUBLE_CLICK=False 延迟模式）的位置门控阈值：
#   体感勾指"双击"实测间隔 1.5~3.3s，远超 OS 500ms 双击窗，补发 click#1 后第二击若发普通 click，
#   资源管理器会把"选中后慢速再点文件名"判为重命名。补救窗口内第二击落点距 click#1 ≤ 本阈值时
#   判定为"同一图标双击意图"→ 原子 doubleClick（内部两次点击几十ms，OS 必判双击打开）并吸附坐标；
#   超过阈值说明手已移到别的图标 → 降级普通 click。阈值比 TAP_DBL_SNAP_PX 宽（慢节奏手漂更大，
#   且桌面图标尺寸普遍 50~100px）。单位：屏幕像素。
RESCUE_DBL_SNAP_PX = 100
# ⭐ 点击诊断日志开关：True 时打印 [CLICK-DIAG]（双击间隔/漂移/吸附、补救双击判定）。
#   2026-09-15：日常使用默认关闭（每次点击打印+flush 会在控制台 I/O 上阻塞主循环）；排查双击问题时临时改 True
CLICK_DIAG = False
# ⭐ 2026-08-22 补充：显式锁死"动作内部最小耗时 / 最小休眠"为 0
#   PAUSE=0 只保证"两个动作之间"不强制 sleep 0.1s；
#   MINIMUM_DURATION（默认 0，已安全）是 moveTo/click 内部 duration 参数若 < MINIMUM_DURATION 则直接置 0（duration=0 → 无过渡，瞬时跳转）；
#   MINIMUM_SLEEP（默认 0.05s）是 pyautogui 做任何 move/click 后强制短暂休眠的兜底——我们自己有 EMA/1€/Sigmoid 多层平滑，不需要 pyautogui 再加 50ms 拖尾，否则 30fps 输入下叠加 50ms 主动 sleep = 连续 moveTo 的理论上限 20fps，体感明显顿。
pyautogui.MINIMUM_DURATION = 0.0
pyautogui.MINIMUM_SLEEP = 0.0


# ============================================================================
# 1€ Filter (Casiez et al. 2012) — 体感输入标准平滑算法
#   - 相比固定 α EMA：慢动作 α→0 超平滑，快动作 α→1 超跟手
#   - 无延迟/过冲矛盾（尤其适合 30fps 摄像头这种低采样率场景）
# ============================================================================
class _OneEuroFilter1D:
    def __init__(self, fc_min: float = 1.0, beta: float = 0.007,
                 fc_d: float = 1.0, rate: float = 30.0):
        self.fc_min = float(fc_min)
        self.beta   = float(beta)
        self.fc_d   = float(fc_d)
        self.rate   = float(rate)
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    @staticmethod
    def _alpha(cutoff: float, rate: float) -> float:
        tau = 1.0 / (2 * np.pi * cutoff)
        te = 1.0 / max(rate, 1e-6)
        return 1.0 / (1.0 + tau / te)

    def reset(self):
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    def apply(self, x: float, t: float | None = None) -> float:
        if t is None:
            t = time.time()
        if self._x_prev is None:
            self._x_prev = float(x)
            self._t_prev = t
            return float(x)

        te = 1.0 / max(1e-6, (t - self._t_prev) if (t - self._t_prev) > 1e-4 else (1.0 / self.rate))
        rate = 1.0 / max(te, 1e-6)
        # 先对速度做一个更宽截止的低通
        dx = (x - self._x_prev) * rate
        a_d = self._alpha(self.fc_d, rate)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        # 根据 |dx_hat| 动态提高截止频率（动作越快 → cutoff 越高 → alpha 越大 → 越跟手）
        cutoff = self.fc_min + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, rate)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t
        return float(x_hat)


class _TwoEuroFilter2D:
    def __init__(self, fc_min: float, beta: float):
        self.xf = _OneEuroFilter1D(fc_min=fc_min, beta=beta)
        self.yf = _OneEuroFilter1D(fc_min=fc_min, beta=beta)

    def reset(self):
        self.xf.reset()
        self.yf.reset()

    def apply(self, x, y, t=None):
        if t is None:
            t = time.time()
        return self.xf.apply(x, t), self.yf.apply(y, t)

    def set_params(self, fc_min: float, beta: float):
        """运行时切换模式参数（平滑重置少量内部状态）"""
        # 只改参数，不清历史（避免切换瞬间光标跳）
        self.xf.fc_min = fc_min; self.xf.beta = beta
        self.yf.fc_min = fc_min; self.yf.beta = beta


# ============================================================================
# AR 悬停式平滑 5 层流水线
# ============================================================================
class ARSmoother:
    def __init__(self, mode: str = CURSOR_MODE_SMOOTH):
        self._mode = mode
        self._median_buf_x: deque[float] = deque(maxlen=MEDIAN_WINDOW)
        self._median_buf_y: deque[float] = deque(maxlen=MEDIAN_WINDOW)
        self._lock_buf_x:   deque[float] = deque(maxlen=LOCK_WINDOW)
        self._lock_buf_y:   deque[float] = deque(maxlen=LOCK_WINDOW)
        self._euro = _TwoEuroFilter2D(fc_min=DEFAULT_PARAMS[0], beta=DEFAULT_PARAMS[1])
        # 锁死时光标位置（smooth 模式 Layer 2）
        self._locked_x: float | None = None
        self._locked_y: float | None = None
        self._last_params = DEFAULT_PARAMS
        # 点击前冻结（两种模式都生效）
        self._freeze_x: float | None = None
        self._freeze_y: float | None = None
        self._freeze_left = 0
        self._last_out_x = None
        self._last_out_y = None
        # absolute 模式 Layer 1：3 帧中值杀尖峰
        self._abs_buf_x: deque[float] = deque(maxlen=ABS_MEDIAN_WINDOW)
        self._abs_buf_y: deque[float] = deque(maxlen=ABS_MEDIAN_WINDOW)
        # ✨✨ absolute 模式 Layer 3：静止方差锁（样本缓冲 + 锁位置）
        self._abs_lock_buf_x: deque[float] = deque(maxlen=ABS_LOCK_WINDOW)
        self._abs_lock_buf_y: deque[float] = deque(maxlen=ABS_LOCK_WINDOW)
        self._abs_locked_x: float | None = None   # 当前被锁死的固定光标 X（None=未锁）——=输出给鼠标的像素位置
        self._abs_locked_y: float | None = None   # 当前被锁死的固定光标 Y（None=未锁）
        self._abs_locked_ref_ax: float | None = None  # ⭐ 锁成立时 Layer1(ax) 参考值（用于解锁判定时比较手掌是否真的移动）
        self._abs_locked_ref_ay: float | None = None  # ⭐ 锁成立时 Layer1(ay) 参考值
        self._abs_lock_cooldown: int = 0  # ⭐ 刚解锁后的冷却帧数：此期间不允许重新锁死（防止解锁→重锁抖动）

    # ---------- 外部：模式切换 ----------
    def set_mode(self, mode: str):
        """切换 cursor 模式：smooth / absolute / direct"""
        if mode == self._mode:
            return
        assert mode in (CURSOR_MODE_SMOOTH, CURSOR_MODE_ABSOLUTE, CURSOR_MODE_DIRECT), f"unknown cursor mode: {mode}"
        self._mode = mode
        self.reset()

    @property
    def mode(self) -> str:
        return self._mode

    # ---------- 外部：手势模式切换 ----------
    def notify_gesture(self, gesture: str):
        fc_min, beta, dz, k = MODE_PARAMS.get(gesture, DEFAULT_PARAMS)
        if (fc_min, beta) != (self._euro.xf.fc_min, self._euro.xf.beta):
            self._euro.set_params(fc_min, beta)
        self._last_params = (fc_min, beta, dz, k)

    # ---------- 外部：点击前冻结 ----------
    def freeze_for_click(self):
        """在检测到 TAP/PINCH/PEACE 即将触发点击时调用，锁定位置 N 帧。"""
        if self._last_out_x is not None and self._freeze_left <= 0:
            self._freeze_x = self._last_out_x
            self._freeze_y = self._last_out_y
            self._freeze_left = CLICK_FREEZE_FRAMES

    def reset(self):
        for b in (self._median_buf_x, self._median_buf_y,
                  self._lock_buf_x,   self._lock_buf_y,
                  self._abs_buf_x,    self._abs_buf_y,
                  self._abs_lock_buf_x, self._abs_lock_buf_y):
            b.clear()
        self._euro.reset()
        self._locked_x = None
        self._locked_y = None
        self._last_out_x = None
        self._last_out_y = None
        self._freeze_left = 0
        self._freeze_x = None
        self._freeze_y = None
        # ✨✨ absolute 模式 Layer 3 锁状态清空
        self._abs_locked_x = None
        self._abs_locked_y = None
        self._abs_locked_ref_ax = None
        self._abs_locked_ref_ay = None
        self._abs_lock_cooldown = 0

    # ---------- 内部工具 ----------
    @staticmethod
    def _sigmoid_gain(delta_px: float, response_k: float) -> float:
        """
        Sigmoid 非线性响应（第 4 层 / smooth 模式用）：
          delta_px 很小（<3px微抖）→ 增益≈0.2  阻尼抖动
          delta_px 很大（>20px真实移动）→ 增益→1.0  不衰减、保证跟手
        response_k 越小 → 拐点越靠近 0（更严的抖动抑制）
        """
        if response_k <= 1e-6:
            return 1.0
        s = 1.0 / (1.0 + np.exp(-response_k * (delta_px - 4.0)))
        # scale to ~0.22..1.0 之间，避免过小增益导致不动
        return 0.22 + 0.78 * s

    @staticmethod
    def _abs_sigmoid_gain(delta_px: float) -> float:
        """✨ 方案 A：absolute 模式 Layer 2 专用 Sigmoid 增益（用 ABS_* 参数，拐点/最小增益可独立调）"""
        if ABS_SIGMOID_K <= 1e-6:
            return 1.0
        # ⭐ 2026-08-24 用户反馈「难以稳定」：INFLECTION_PX 11.0 → 14.0
        #   K=0.35 / MIN_G=0.20 / INFLECT=14px：
        #     Δ≤6px → 增益 0.21~0.28（更强阻尼压制微抖=更易稳定）
        #     Δ=14px → 增益≈0.60（拐点中点，1.7x阻尼，仍顺滑）
        #     Δ≥22px → 增益→1.0（大范围 0 衰减=指哪打哪不丢）
        INFLECTION_PX = 14.0
        s = 1.0 / (1.0 + np.exp(-ABS_SIGMOID_K * (float(delta_px) - INFLECTION_PX)))
        min_g = float(max(0.0, min(1.0, ABS_SIGMOID_MIN_G)))
        amp = 1.0 - min_g
        return float(min_g + amp * s)

    # ---------- 主入口：5 层流水线 ----------
    def filter(self, raw_x: float, raw_y: float, gesture: str | None = None):
        """输入=归一化坐标映射后的「屏幕像素原始值」，输出=AR平滑后屏幕像素。

        三种模式：
          - CURSOR_MODE_SMOOTH  : 完整 5 层 AR 悬停流水线（方差锁/1€/Sigmoid 等）
          - CURSOR_MODE_ABSOLUTE: 方案 A 3 层流水线 → 大范围跟手 + 小范围不晃
          - CURSOR_MODE_DIRECT  : ⭐ 严格跟随 UI 掌心绿圈（仅 3 帧中值杀飞点，无阻尼无锁）
        """
        # 点击前冻结（三种模式都生效：点击瞬间锁死避免漂移）
        if self._freeze_left > 0:
            self._freeze_left -= 1
            fx, fy = self._freeze_x, self._freeze_y
            self._last_out_x = fx
            self._last_out_y = fy
            return fx, fy

        # ============================================================
        # ⭐⭐ CURSOR_MODE_DIRECT（严格跟手模式：1:1 匹配 UI 掌心绿圈）
        #   目标：无论手掌如何翻转/倾斜，鼠标和预览画面的绿圈严格同步，绝不再"绿圈动了鼠标卡"
        #   流水线：
        #     Layer 1：3 帧中值（杀 1~2 帧 MediaPipe 偶发飞点，保留画面上看得到的那种跟手感）
        #     无 Layer 2 Sigmoid 阻尼
        #     无 Layer 3 静止方差锁
        #   ⚠️ 原理：landmarks 端已做 EMA α=0.85（+ 掌心大位移硬重置），UI 绿圈已经跟手且稳定；
        #           鼠标端再叠任何阻尼/锁，都会导致"绿圈动鼠标不动"的视觉脱节。
        #           尤其是手掌翻转时，掌心在 3D 中旋转 → X/Y 投影会出现 2~5px 高频微抖，
        #           absolute 模式的 Sigmoid+静止锁会把这种抖动误判为"应该阻尼/锁死"，表现为卡顿；
        #           但 UI 上绿圈直接画（无额外锁），所以看起来"绿圈准、鼠标卡"——
        #           direct 模式就是让鼠标端和绿圈用完全一样的策略，从根上消除脱节。
        # ============================================================
        if self._mode == CURSOR_MODE_DIRECT:
            # Layer 1：3 帧中值（仅杀偶发 1 帧飞点，不引入任何延迟）
            self._abs_buf_x.append(float(raw_x))
            self._abs_buf_y.append(float(raw_y))
            if len(self._abs_buf_x) >= ABS_MEDIAN_WINDOW and len(self._abs_buf_y) >= ABS_MEDIAN_WINDOW:
                mx = float(np.median(self._abs_buf_x))
                my = float(np.median(self._abs_buf_y))
            else:
                mx = float(raw_x)
                my = float(raw_y)
            # 直接输出（无 sigmoid / 无锁 / 无 EMA）——和 UI 绿圈完全同步
            self._last_out_x = float(mx)
            self._last_out_y = float(my)
            return float(mx), float(my)

        # ============================================================
        # ✨✨ CURSOR_MODE_ABSOLUTE（方案 A 3 层流水线：大范围跟手 + 小范围不晃）
        # ============================================================
        if self._mode == CURSOR_MODE_ABSOLUTE:
            # ============================================================
            # Layer 1：3 帧中值（杀 MediaPipe 1~2 帧偶发尖峰，和原逻辑一致）
            #   作用：过滤 ±30px 的 1 帧"飞点"（比如手掌瞬间识别错位跳到角落）
            # ============================================================
            self._abs_buf_x.append(float(raw_x))
            self._abs_buf_y.append(float(raw_y))
            if len(self._abs_buf_x) >= ABS_MEDIAN_WINDOW and len(self._abs_buf_y) >= ABS_MEDIAN_WINDOW:
                ax = float(np.median(self._abs_buf_x))
                ay = float(np.median(self._abs_buf_y))
            else:
                ax = float(raw_x)
                ay = float(raw_y)

            # ============================================================
            # Layer 2：Sigmoid 非线性微移增益压缩（抑制小抖动，但大位移完全不衰减=跟手）
            #   ⚠️ 2026-08-22 放宽：Δ≤2px 阻尼 1.7 倍；Δ=5px 增益≈0.87；Δ>9px 增益→1.0
            #   核心：鼠标端阻尼必须匹配 UI 绿圈（EMA 0.85）的响应速度，不能再叠加强阻尼导致视觉脱节
            # ============================================================
            if self._last_out_x is None or self._last_out_y is None:
                sx, sy = float(ax), float(ay)
            else:
                dx = float(ax) - float(self._last_out_x)
                dy = float(ay) - float(self._last_out_y)
                delta_px = float(np.sqrt(dx * dx + dy * dy))
                if ABS_SIGMOID_K <= 1e-6:
                    g = 1.0
                else:
                    g = float(self._abs_sigmoid_gain(delta_px))
                sx = float(self._last_out_x) + dx * g
                sy = float(self._last_out_y) + dy * g

            # ============================================================
            # Layer 3：静止方差锁（对准小按钮的核心杀招 → 光标"钉住"完全不动）
            #   ⚠️ 2026-08-22 关键修复：
            #     ① 判定/解锁全部改用 Layer1 原始 (ax,ay)，不再用 Sigmoid 压缩后的 sx/sy——
            #        之前用 sx/sy 导致 Sigmoid 把真实移动压小，解锁累计量永远凑不够→鼠标死锁不动
            #     ② 锁点本身仍取最近 Layer2 (sx,sy) 均值（保证输出连续不跳）
            #     ③ UNLOCK_DIST 累计阈值从 2.0×降到 1.5×（更灵敏解锁）
            # ============================================================
            UNLOCK_DIST = float(ABS_LOCK_THRESH) * 1.6   # 8.0px
            RELOCK_COOLDOWN_FRAMES = ABS_LOCK_WINDOW * 2  # ⭐ 解锁后 10 帧内不允许重锁（防止缓慢移动时解锁-重锁抖动）
            # ⭐ 用 Layer1 原始 ax/ay 送入锁判定缓冲（真实反映手掌有没有在动）
            self._abs_lock_buf_x.append(float(ax))
            self._abs_lock_buf_y.append(float(ay))
            # 解锁冷却计数（每帧 -1，到 0 才允许再次锁死）
            if self._abs_lock_cooldown > 0:
                self._abs_lock_cooldown -= 1

            buf_full = len(self._abs_lock_buf_x) >= ABS_LOCK_WINDOW and len(self._abs_lock_buf_y) >= ABS_LOCK_WINDOW
            # ① 已在锁死状态 → 先检查是否需要解锁（基于原始 Layer1 值判断手掌是否真动）
            if self._abs_locked_x is not None and self._abs_locked_y is not None:
                # 单点距离：原始 Layer1 (ax,ay) 到「锁点对应的原始参考点」的距离
                dist_to_lock = float(np.sqrt(
                    (float(ax) - float(self._abs_locked_ref_ax)) ** 2 +
                    (float(ay) - float(self._abs_locked_ref_ay)) ** 2
                ))
                # 累计位移趋势解锁：缓冲内所有 Layer1 原始点相对参考点的累计
                cum_disp = 0.0
                try:
                    for xi, yi in zip(self._abs_lock_buf_x, self._abs_lock_buf_y):
                        cum_disp += abs(float(xi) - float(self._abs_locked_ref_ax)) + abs(float(yi) - float(self._abs_locked_ref_ay))
                except Exception:
                    cum_disp = 0.0
                # ⭐ 累计阈值从 2.0×UNLOCK 降到 1.5×：手掌持续移动能更快解锁
                unlock_trend = cum_disp > (1.5 * UNLOCK_DIST)

                if dist_to_lock <= UNLOCK_DIST and not unlock_trend:
                    # 仍然在锁点附近 + 累计也没动 → 继续锁死，输出固定光标位置
                    lx, ly = float(self._abs_locked_x), float(self._abs_locked_y)
                    self._last_out_x = lx
                    self._last_out_y = ly
                    return lx, ly
                else:
                    # 真实动作 → 解锁；清空锁缓冲 + 启动冷却（N 帧内不再重锁）
                    self._abs_locked_x = None
                    self._abs_locked_y = None
                    self._abs_locked_ref_ax = None
                    self._abs_locked_ref_ay = None
                    self._abs_lock_cooldown = RELOCK_COOLDOWN_FRAMES
                    self._abs_lock_buf_x.clear()
                    self._abs_lock_buf_y.clear()
                    self._abs_lock_buf_x.append(float(ax))
                    self._abs_lock_buf_y.append(float(ay))
                    buf_full = False
            # ② 未锁死 → 尝试满足「缓冲满 + 冷却已过 + Layer1 原始 σ < 阈值」→ 进入锁死
            if buf_full and self._abs_locked_x is None and self._abs_lock_cooldown <= 0:
                std_x = float(np.std(self._abs_lock_buf_x))
                std_y = float(np.std(self._abs_lock_buf_y))
                if std_x < ABS_LOCK_THRESH and std_y < ABS_LOCK_THRESH:
                    # 锁光标位置 = 最近的 Layer2 (sx,sy) 均值（不跳）
                    # 参考点（用于解锁判断）= 最近的 Layer1 (ax,ay) 均值
                    self._abs_locked_ref_ax = float(np.mean(self._abs_lock_buf_x))
                    self._abs_locked_ref_ay = float(np.mean(self._abs_lock_buf_y))
                    lock_x = float(sx)
                    lock_y = float(sy)
                    self._abs_locked_x = lock_x
                    self._abs_locked_y = lock_y
                    self._last_out_x = lock_x
                    self._last_out_y = lock_y
                    return lock_x, lock_y
            # ③ 正常（不满足锁死条件）→ 输出 Layer2 结果
            self._last_out_x = float(sx)
            self._last_out_y = float(sy)
            return float(sx), float(sy)

        # ========= CURSOR_MODE_SMOOTH（原 5 层流水线）=========
        if gesture:
            self.notify_gesture(gesture)
        fc_min, beta, deadzone_px, response_k = self._last_params

        # -------- 点击冻结：优先返回冻结坐标 --------
        if self._freeze_left > 0:
            self._freeze_left -= 1
            ox, oy = float(self._freeze_x), float(self._freeze_y)
            self._last_out_x, self._last_out_y = ox, oy
            return ox, oy

        # ===== Layer 1: 滑动中值（杀 1~2 帧 MediaPipe 尖峰）=====
        self._median_buf_x.append(float(raw_x))
        self._median_buf_y.append(float(raw_y))
        # 求中值（窗口不满时用平均，等价）
        if len(self._median_buf_x) < MEDIAN_WINDOW:
            mx = sum(self._median_buf_x) / len(self._median_buf_x)
            my = sum(self._median_buf_y) / len(self._median_buf_y)
        else:
            mx = float(np.median(self._median_buf_x))
            my = float(np.median(self._median_buf_y))

        # ===== Layer 2: 静止方差锁（8 帧不动 → 直接锁死）=====
        self._lock_buf_x.append(mx)
        self._lock_buf_y.append(my)
        lock_thresh = {
            "POINT": LOCK_THRESH_HOVER,
            "OPEN":  LOCK_THRESH_BROWSE,
            # ✨ 用户要求：FIST = 取消所有指令/无指令 → 不再用 DRAG=99999（不锁死），改为 BROWSE=3.0 中性（实际上 FIST 分支不触发任何动作，这里只是占位避免 KeyError）
            "FIST":  LOCK_THRESH_BROWSE,
        }.get(gesture if gesture else "OTHER", LOCK_THRESH_BROWSE)

        locked_now = False
        if len(self._lock_buf_x) >= LOCK_WINDOW:
            var_x = float(np.var(self._lock_buf_x))
            var_y = float(np.var(self._lock_buf_y))
            if (var_x + var_y) < lock_thresh:
                locked_now = True
                if self._locked_x is None:
                    self._locked_x = mx
                    self._locked_y = my
                # 用锁死值替换中值输出
                mx, my = float(self._locked_x), float(self._locked_y)
        if not locked_now:
            self._locked_x = None
            self._locked_y = None

        # ===== Layer 3: 1€ Filter（速度自适应平滑）=====
        t = time.time()
        ex, ey = self._euro.apply(mx, my, t)

        # ===== Layer 4: Sigmoid 非线性响应（小幅阻尼/大幅跟手）=====
        if self._last_out_x is None:
            ox, oy = float(ex), float(ey)
        else:
            dx = ex - self._last_out_x
            dy = ey - self._last_out_y
            dist = float(np.hypot(dx, dy))
            # 死区叠加：非常小的位移直接归零
            if dist <= deadzone_px:
                gain = 0.0
            else:
                gain = self._sigmoid_gain(dist, response_k)
            # 新目标 = 上次输出 + 经过 sigmoid gain 缩放后的 delta
            ox = float(self._last_out_x + dx * gain)
            oy = float(self._last_out_y + dy * gain)

        self._last_out_x, self._last_out_y = ox, oy
        return ox, oy


# ============================================================================
class MouseController:
    def __init__(self, screen_size=None, cursor_mode: str = CURSOR_MODE_SMOOTH,
                 roi_scale: float = DEFAULT_ROI_SCALE,
                 roi_offset_y: float = DEFAULT_ROI_OFFSET_Y,
                 roi_safe_shrink: float | None = None, **_ignored):
        if screen_size is None:
            self.sw, self.sh = pyautogui.size()
        else:
            self.sw, self.sh = screen_size
        # ROI scale：画面中心多少%映射到整个屏幕（解决"掌心锁死、手指不可见"）
        roi_scale = float(roi_scale)
        if roi_scale < ROI_SCALE_MIN: roi_scale = ROI_SCALE_MIN
        if roi_scale > ROI_SCALE_MAX: roi_scale = ROI_SCALE_MAX
        self._roi_scale = roi_scale
        # ROI 垂直偏移：⚠️ 只改Y方向位置，X宽和Y高与原roi_scale定义完全一致（长方形，不是正方形）
        roi_offset_y = float(roi_offset_y)
        if roi_offset_y < ROI_OFFSET_Y_MIN: roi_offset_y = ROI_OFFSET_Y_MIN
        if roi_offset_y > ROI_OFFSET_Y_MAX: roi_offset_y = ROI_OFFSET_Y_MAX
        self._roi_offset_y = roi_offset_y
        # ROI 安全内缩：在原 ROI 长方形基础上中心等比再缩多少（0.50~1.00）；None=用全局默认 ROI_SAFE_SHRINK
        if roi_safe_shrink is None:
            self._roi_safe_shrink = float(ROI_SAFE_SHRINK)
        else:
            s = float(roi_safe_shrink)
            if s < ROI_SAFE_SHRINK_MIN: s = ROI_SAFE_SHRINK_MIN
            if s > ROI_SAFE_SHRINK_MAX: s = ROI_SAFE_SHRINK_MAX
            self._roi_safe_shrink = s
        # 摄像头画面尺寸（默认 None = 未知 → _norm_to_screen 退化为"兼容旧模式"）；
        # 由主循环 hands_tasks / main.py 在第一次拿到 frame 时调用 set_frame_size 填入。
        # 只有填入后，_norm_to_screen 的坐标映射才和 compute_roi_bbox_px 绘制的绿框 100% 精确对齐。
        self._cam_frame_w: int | None = None
        self._cam_frame_h: int | None = None
        # AR 悬停 5 层平滑器（支持 smooth / absolute 两种 cursor 策略）
        self._smoother = ARSmoother(mode=cursor_mode)
        # 防抖状态
        self._last_gesture = None
        self._hold_frames = 0
        # 拖拽状态
        self._drag_active = False
        # ⭐ 2026-08-22 新增：同帧共享"最新屏幕坐标缓存"
        #   避免 move_only() + handle() + 食指 Tap-Click 段各自独立调用 _smooth() → smoother 缓冲/锁/EMA 被同一帧更新3次 → 状态混乱、点击坐标严重错位
        #   规则：每一帧对同一个右手 cursor_anchor_norm → 只跑一次 _norm_to_screen + _smooth → 结果存在这里
        #         move_only() 输出 = last_screen_pt（移动）
        #         handle() 防抖/移动分支直接用 last_screen_pt，不再重跑 _smooth
        #         食指 Tap-Click 的 right_screen_pt 也直接用 last_screen_pt（100% 与绿圈对齐）
        self.last_screen_norm: tuple[float, float] | None = None  # 上次跑 smooth 用的归一化锚点（用于判"是否同帧同坐标"）
        self.last_screen_px:   tuple[float, float] | None = None  # 上次输出屏幕像素坐标（gx,gy 原始 raw 像素）
        self.last_smoothed_px: tuple[float, float] | None = None  # 上次输出平滑后屏幕像素坐标（sx,sy 已过 smoother）

    # ---------------- 外部：ROI / cursor 模式切换 ----------------
    def set_roi_scale(self, roi_scale: float):
        """动态调整画面中心映射全屏的矩形区域比例（0.30~1.00）。"""
        roi_scale = float(roi_scale)
        if roi_scale < ROI_SCALE_MIN: roi_scale = ROI_SCALE_MIN
        if roi_scale > ROI_SCALE_MAX: roi_scale = ROI_SCALE_MAX
        self._roi_scale = roi_scale

    @property
    def roi_scale(self) -> float:
        return self._roi_scale

    def set_roi_offset_y(self, roi_offset_y: float):
        """动态调整长方形 ROI 的上下位置（-1.0 贴顶, 0 居中, +1.0 贴底）；X宽/Y高尺寸完全不变。"""
        roi_offset_y = float(roi_offset_y)
        if roi_offset_y < ROI_OFFSET_Y_MIN: roi_offset_y = ROI_OFFSET_Y_MIN
        if roi_offset_y > ROI_OFFSET_Y_MAX: roi_offset_y = ROI_OFFSET_Y_MAX
        self._roi_offset_y = roi_offset_y

    @property
    def roi_offset_y(self) -> float:
        return self._roi_offset_y

    def set_roi_safe_shrink(self, roi_safe_shrink: float):
        """动态调整安全子 ROI 等比内缩比例（0.50~1.00）：越小=虚线框越小=指尖越不易超框。"""
        s = float(roi_safe_shrink)
        if s < ROI_SAFE_SHRINK_MIN: s = ROI_SAFE_SHRINK_MIN
        if s > ROI_SAFE_SHRINK_MAX: s = ROI_SAFE_SHRINK_MAX
        self._roi_safe_shrink = s

    @property
    def roi_safe_shrink(self) -> float:
        return self._roi_safe_shrink

    def set_frame_size(self, frame_w: int, frame_h: int):
        """主循环在拿到第一帧后调用，让 _norm_to_screen 计算的坐标与预览绿框 100% 对齐。"""
        self._cam_frame_w = int(max(1, frame_w))
        self._cam_frame_h = int(max(1, frame_h))

    @staticmethod
    def compute_roi_bbox_px(frame_w: int, frame_h: int, roi_scale: float, roi_offset_y: float = 0.0,
                            roi_safe_shrink: float | None = None):
        """✨ 统一工具：计算 ROI 矩形在画面中的像素坐标 (x1,y1,x2,y2)。
        供预览绘制、diagonal_stretch 等所有 UI 代码共用（保证与 _norm_to_screen 的映射范围完全一致）。

        ⚠️ 严格遵循用户原逻辑，只做最小增量：
          1. X 方向：100% 保留原 margin 计算方式（左右对称 margin=(1-rs)*0.5*min(fw,fh)）
              → 宽度 = fw - 2*margin，**绝不改成正方形**！
          2. Y 方向：高度仍 = rs*min(fw,fh)（和原来完全一样，高度不变）
              → 仅通过 roi_offset_y 把"可用 Y margin"在 顶/底 之间重新分配
              → 顶 margin = (h - roi_h) * (0.5 + 0.5*oy)，实现整体上下移动。
          3. ⭐ 2026-08-24 新增：roi_safe_shrink "安全子 ROI"等比内缩
              · 在步骤 1+2 算出的「原始 ROI 矩形」中心位置保持不变的前提下，
                宽×高 再乘以 roi_safe_shrink（默认 ROI_SAFE_SHRINK=0.82）得到更小的"安全子矩形"。
              · 这个安全子矩形 = UI 画的虚线框 = 鼠标坐标真正映射生效的区域；
              · 但 landmark/手势检测仍传入**整帧画面**（检测范围不受此框影响）
                → 掌心/指尖落在安全子矩形外但仍在画面内：MediaPipe 仍能正常识别=不会丢手
                → 刚好满足需求 2：鼠标映射限定在虚线框内，手势识别 = 全画面覆盖。
        """
        fw = float(max(1, int(frame_w))); fh = float(max(1, int(frame_h)))
        rs = float(max(ROI_SCALE_MIN, min(ROI_SCALE_MAX, float(roi_scale))))
        oy = float(max(ROI_OFFSET_Y_MIN, min(ROI_OFFSET_Y_MAX, float(roi_offset_y))))
        if roi_safe_shrink is None:
            s = float(ROI_SAFE_SHRINK)
        else:
            s = float(max(ROI_SAFE_SHRINK_MIN, min(ROI_SAFE_SHRINK_MAX, float(roi_safe_shrink))))
        m = (1.0 - rs) * 0.5 * min(fw, fh)       # 原逻辑：X方向margin（左右各 m）
        x1 = m
        x2 = fw - m
        roi_h_px = rs * min(fw, fh)              # Y方向高度：原逻辑，绝对不变
        avail_y_margin = fh - roi_h_px           # 可分配给顶部/底部的总 Y margin（=原 2*m 当画面正方形时）
        top_margin_px = avail_y_margin * (0.5 + 0.5 * oy)  # oy=-1→全给底（顶=0）；oy=0→各一半（居中）；oy=+1→全给顶
        y1 = top_margin_px
        y2 = y1 + roi_h_px
        # ⭐ SAFE_SHRINK 等比内缩：中心不动，宽高 × s
        cx = 0.5 * (x1 + x2); cy = 0.5 * (y1 + y2)
        nw = max(1.0, (x2 - x1) * s); nh = max(1.0, (y2 - y1) * s)
        sx1 = cx - 0.5 * nw; sx2 = cx + 0.5 * nw
        sy1 = cy - 0.5 * nh; sy2 = cy + 0.5 * nh
        return (int(round(sx1)), int(round(sy1)), int(round(sx2)), int(round(sy2)))

    def set_cursor_mode(self, mode: str):
        """切换光标策略：smooth（AR悬停5层）/ absolute（掌心绝对坐标直接跳）"""
        self._smoother.set_mode(mode)

    @property
    def cursor_mode(self) -> str:
        return self._smoother.mode

    # ---------------- 内部工具 ----------------
    def _eval_anchor_once(self, anchor_norm, gesture_hint: str | None = None):
        """⭐ 2026-08-22 新增：同帧同一归一化锚点 → 只跑一次 norm_to_screen + smoother.filter。

        返回 (gx, gy, sx, sy, ix, iy)：
          (gx,gy) = ROI 映射后的原始屏幕像素（未平滑）
          (sx,sy) = 经过 smoother 后的屏幕像素
          (ix,iy) = int(round(sx,sy)) 用于 pyautogui

        缓存命中逻辑：若传入的 anchor_norm 与上次调用完全一致（浮点比较 |Δ|<1e-9），直接返回缓存 → 保证 move_only / handle / tap-click 同帧共享完全相同的平滑结果，smoother 状态每帧只更新一次。
        """
        nx, ny = float(anchor_norm[0]), float(anchor_norm[1])
        # 缓存命中（归一化锚点没变化 = 同一帧调用多次）→ 直接用缓存
        if (self.last_screen_norm is not None
                and abs(nx - self.last_screen_norm[0]) < 1e-9
                and abs(ny - self.last_screen_norm[1]) < 1e-9
                and self.last_screen_px is not None
                and self.last_smoothed_px is not None):
            gx, gy = self.last_screen_px
            sx, sy = self.last_smoothed_px
            return gx, gy, sx, sy, int(round(sx)), int(round(sy))
        # 缓存未命中 → 实际执行映射 + 平滑一次 → 写缓存
        gx, gy = self._norm_to_screen(nx, ny)
        sx, sy = self._smoother.filter(gx, gy, gesture=gesture_hint)
        self.last_screen_norm = (nx, ny)
        self.last_screen_px   = (float(gx), float(gy))
        self.last_smoothed_px = (float(sx), float(sy))
        return float(gx), float(gy), float(sx), float(sy), int(round(sx)), int(round(sy))

    def invalidate_anchor_cache(self):
        """外部在一帧完全处理完后可调用，清空缓存以强制下一帧重新评估；
        也可不清（下一帧 anchor_norm 自然不同就会自动重跑）。"""
        self.last_screen_norm = None
        self.last_screen_px = None
        self.last_smoothed_px = None

    def _smooth(self, nx: float, ny: float, gesture: str | None = None):
        """⚠️ 直接调用 _smooth 会绕过同帧缓存——请优先使用 _eval_anchor_once()。
        保留此接口兼容外部代码（如 Tap-Click 段重构前的直接调用）。"""
        return self._smoother.filter(nx, ny, gesture=gesture)

    def _norm_to_screen(self, norm_x: float, norm_y: float):
        """landmark 归一化坐标 -> 屏幕像素。

        映射规则（用户可调参数 roi_scale + roi_offset_y）——⚠️ 只改 Y 位置，尺寸 100% 保留原逻辑：
          设 fw=1280 fh=720 roi_scale=0.70 roi_offset_y=0.5：
            m=(1-0.7)*0.5*720=108px
            x1=108 x2=1280-108=1172 → 宽度=1064px（**完全不变**，和你原来的版本一模一样）
            roi_h=0.7*720=504px（高度也完全不变）
            avail_y_margin=720-504=216px
            y1=216*(0.5+0.25)=162 y2=162+504=666 → 顶边从原 108px 下移到 162px（多 54px 顶部空间给手指伸展）
          ROI 内部坐标 0..1 → 线性映射屏幕 (0,0) ~ (sw,sh)。

        ⭐ 2026-08-24 追加：在上述「原始 ROI」基础上，再做一次 SAFE_SHRINK=0.82 的等比内缩（需求1+2）：
          · 最终映射到屏幕的区域 = 这个缩小后的「安全子矩形」= UI 画的虚线框；
          · 手势识别本身仍是整帧检测（detector.detect(frame) 传入未裁剪的整画面），不在此 ROI 范围内限制。
        """
        r = float(self._roi_scale)
        oy = float(self._roi_offset_y)
        s  = float(self._roi_safe_shrink)
        nx = float(norm_x); ny = float(norm_y)

        # ===== 精确模式（已调用 set_frame_size，和 compute_roi_bbox_px 100% 对齐）=====
        # ⚠️ 严格遵循经验 ID 977325：只在【原精确计算结果末尾】追加 SAFE_SHRINK 补丁，
        #    不修改原 roi_scale / roi_offset_y 的任何推导逻辑与变量名（避免引入变量命名错乱破坏链路）。
        if self._cam_frame_w is not None and self._cam_frame_h is not None:
            fw = float(self._cam_frame_w); fh = float(self._cam_frame_h)
            m = (1.0 - r) * 0.5 * min(fw, fh)        # 原逻辑 X 方向 margin（左右对称）
            x1_px = m
            x2_px = fw - m
            roi_w_px = max(x2_px - x1_px, 1e-6)      # 原始 ROI 宽度（长方形，原精确模式变量名）
            roi_h_px = r * min(fw, fh)               # 原始 ROI 高度：原逻辑不变
            roi_h_px = max(roi_h_px, 1e-6)
            avail_y_margin = fh - roi_h_px
            y1_px = avail_y_margin * (0.5 + 0.5 * oy)
            y2_px = y1_px + roi_h_px
            # ⭐ 单点增量：在原矩形 (x1_px,y1_px)~(x2_px,y2_px) 中心等比内缩 SAFE_SHRINK
            cx = 0.5 * (x1_px + x2_px); cy = 0.5 * (y1_px + y2_px)
            nw = max(roi_w_px * s, 1e-6); nh = max(roi_h_px * s, 1e-6)
            sx1_px = cx - 0.5 * nw; sy1_px = cy - 0.5 * nh
            # 用缩后的安全子矩形做 u/v 映射（= 鼠标坐标严格限定在虚线框内）
            u = (nx * fw - sx1_px) / nw
            v = (ny * fh - sy1_px) / nh
            u = 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)
            v = 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)
            return u * self.sw, v * self.sh

        # ===== 兼容模式（未设置 frame_size，保留原 margin 近似算法，比例不变）=====
        margin = (1.0 - r) * 0.5
        avail_y_norm = 1.0 - r
        margin_y_top_norm = avail_y_norm * (0.5 + 0.5 * oy)
        denom_x = max(1.0 - 2.0 * margin, 1e-6)   # 原始 ROI X 归一化宽度（原变量名无后缀）
        denom_y = max(r, 1e-6)                    # 原始 ROI Y 归一化高度
        x1_norm = margin                          # 原始 ROI X 起点
        y1_norm = margin_y_top_norm               # 原始 ROI Y 起点
        # ⭐ 单点增量：中心等比 SAFE_SHRINK
        cx_norm = x1_norm + 0.5 * denom_x
        cy_norm = y1_norm + 0.5 * denom_y
        nw_norm = max(denom_x * s, 1e-6)
        nh_norm = max(denom_y * s, 1e-6)
        sx1_norm = cx_norm - 0.5 * nw_norm
        sy1_norm = cy_norm - 0.5 * nh_norm
        x = (nx - sx1_norm) / nw_norm
        y = (ny - sy1_norm) / nh_norm
        x = 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
        y = 0.0 if y < 0.0 else (1.0 if y > 1.0 else y)
        return x * self.sw, y * self.sh

    # ---------------- 对外主接口 ----------------
    def move_only(self, cursor_anchor_norm, gesture_hint: str | None = None):
        """⭐ 2026-08-22 新增：先移动鼠标（与 UI 绿圈严格 1:1 对齐），手势只负责点击语义。

        完全不做点击/拖拽判断，只做：
          anchor_norm → _eval_anchor_once（同帧只跑 1 次 smooth）→ pyautogui.moveTo
        例外不移动：
          - FIST（用户明确握拳=不动=纯空指令）
          - drag_active=True：拖拽期间由拖拽分支专门 move，避免 moveTo 打断 mouseDown 按下状态
        ⚠️ 每次调用都会先移动鼠标，保证"预览画绿圈的这一帧"和"屏幕光标位置"严格同步——
           不会因上游门控（双手 / OPEN / 拉伸状态）跳过移动而造成"绿圈动了、鼠标钉住"的视觉脱节。

        Args:
            cursor_anchor_norm: (nx, ny) 归一化 0~1，掌心锚点 = 和画绿圈用的完全同一份
            gesture_hint: 可选，传给 smoother 做平滑的手势上下文（不决定移不移动）
        """
        # 1. 坐标映射 + 平滑（⭐ 同帧只跑 1 次：如果 move_only 之后 handle 再传同一份 cursor_anchor_norm 直接命中缓存）
        _, _, sx, sy, ix, iy = self._eval_anchor_once(cursor_anchor_norm, gesture_hint=gesture_hint)
        # 2. 不移动的唯二例外：FIST 或 drag_active（拖拽分支会处理 move）
        if gesture_hint == "FIST":
            return
        if self._drag_active:
            return
        pyautogui.moveTo(ix, iy)

    def handle(self, gesture: str, fingertip_norm, cursor_anchor_norm=None):
        """手势 → 鼠标事件（仅处理点击/拖拽语义；移动请先调用 move_only 保证对齐）。

        ⚠️ 2026-08-22：**强烈建议调用顺序 = move_only(cursor_anchor_norm, gesture_hint=gesture) 在前，handle(gesture, ...) 在后**
        这样：鼠标位置永远先与绿圈对齐；handle 内部只判断"要不要点击/长按/拖拽"，不会因手势门控导致"该帧不移动"。
        为保持兼容：若外部未调 move_only，handle 内部仍会按原逻辑（防抖 + 手势分支）尝试 moveTo，不会回归；
        但若外部已先调用 move_only，handle 内部的 moveTo 会被下面的门控语义控制，不过坐标是一样的（幂等）。

        Args:
            gesture: 如 PEACE / PINCH / OPEN / POINT / FIST / ROCK / THUMB_UP / OTHER
            fingertip_norm: 手势识别用锚点（食指TIP 8，归一化 0~1），保留兼容旧调用
            cursor_anchor_norm: 实际鼠标光标锚点，默认掌心 = (lms[0]+lms[9])/2。
                                None 时回退用 fingertip_norm（兼容）。
        """
        anchor_norm = cursor_anchor_norm if cursor_anchor_norm is not None else fingertip_norm
        # 1. 坐标映射 + AR 5 层平滑
        #   ⚠️ 2026-08-22 改为用 _eval_anchor_once：如果 move_only() 已经在本帧用同一份 anchor_norm 跑过 → 直接命中缓存，smoother 不会再被二次更新
        #      （否则：同一帧 move_only → handle → tap-click 三段各跑一次 smooth，缓冲/锁被更新 3 次 → 状态全乱）
        _, _, sx, sy, ix, iy = self._eval_anchor_once(anchor_norm, gesture_hint=gesture)

        # 2. 防抖状态机
        if gesture != self._last_gesture:
            self._last_gesture = gesture
            self._hold_frames = 0
            # ✨ 用户要求：FIST = 取消所有指令/无指令。手势切换到FIST时：
            #   - 不做任何 moveTo（哪怕 drag 没激活）
            #   - 仅保留：若正在拖拽 → 稍后在 FIST 分支释放（这里先不 move，避免手势切换瞬间跳一下鼠标）
            if gesture != "FIST" and not self._drag_active:
                pyautogui.moveTo(ix, iy)
            return
        self._hold_frames += 1

        # 3. 移动类手势：未满 HOLD_FRAMES 也持续移动
        # ✨ 2026-08-22 修复对齐问题：手掌倾斜/翻转时，MediaPipe 可能偶尔把 POINT/OPEN 误判成 ROCK/PINCH/PEACE/THUMB。
        #   原代码只有 OPEN/POINT/OTHER 在稳定后才 moveTo，若手势"稳定落在"ROCK/PEACE/...（hold_frames >= N）则再也不 moveTo → 鼠标钉死，但 UI 绿圈继续移动 → 用户感知为"无规律卡顿"。
        #   修复原则：
        #     - 移动锚点永远 = cursor_anchor_norm（掌心），和预览画面绿圈画的坐标完全一致（Sigmoid+静止锁只负责平顺，不会改变锚点）
        #     - 手势 = 只是"是否能点击"的语义，不应该决定"鼠标移不移动"
        #     - 不移动的例外：FIST（用户明确=握拳取消无指令，不动鼠标）、drag_active（拖拽期间专门分支处理 move，避免 moveTo 打断按下状态）
        if gesture in ("OPEN", "POINT", "OTHER"):
            if self._drag_active:
                pyautogui.mouseUp()
                self._drag_active = False
            pyautogui.moveTo(ix, iy)
            return
        # ✨ 补充：稳定的 ROCK/PINCH/PEACE/THUMB/OTHER 等非 FIST 且非拖拽 → 也每帧 moveTo（否则鼠标钉死=对齐失控）
        if not self._drag_active and gesture != "FIST":
            pyautogui.moveTo(ix, iy)
            return

        # 未满 HOLD_FRAMES 的点击类手势 → 仅移动
        # ✨ 用户要求：FIST = 无指令 → 哪怕未满 HOLD_FRAMES（前 3 帧）也不移动（避免握拳时鼠标还漂一下）
        if self._hold_frames < HOLD_FRAMES and gesture != "FIST":
            pyautogui.moveTo(ix, iy)
            return

        # 4. 点击类手势：✨ 用户要求「PINCH/PEACE/ROCK/THUMB 全部置空 = 不做任何操作」→ 不再冻结位置，不触发任何 click/rightClick/doubleClick
        if gesture in ("PEACE", "PINCH", "ROCK", "THUMB"):
            # self._smoother.freeze_for_click()   # 置空=不点击，所以也不需要冻结光标
            pass

        if gesture == "PEACE":
            # ✨ 用户要求置空：不做任何操作（原行为=单击左键 → 已废弃）
            pass
        elif gesture == "PINCH":
            # ✨ 用户要求置空：不做任何操作（原行为=单击左键 → 已废弃）
            pass
        elif gesture == "ROCK":
            # ✨ 用户要求置空：不做任何操作（原行为=单击右键 → 已废弃）
            pass
        elif gesture == "THUMB":
            # ✨ 用户要求置空：不做任何操作（原行为=双击左键 → 已废弃）
            pass
        elif gesture == "FIST":
            # ✨ 用户要求：FIST = 取消所有指令 / 无指令
            #   ① 若当前正在拖拽（可能是之前 Tap-Click 长按进入的拖拽）→ 立刻释放 mouseUp，取消正在进行的任何拖拽
            if self._drag_active:
                pyautogui.mouseUp()
                self._drag_active = False
            #   ② 不做 moveTo（握拳=鼠标不能动），不做任何点击/按下，完全空
            #   ③ HOLD_FRAMES 满足进入也不触发任何事 = 纯空指令=取消/无操作
            pass
        # 兜底：其它未映射的新手势也移动一下
        else:
            pyautogui.moveTo(ix, iy)

    # ---------------- 辅助 ----------------
    def _consume_gesture(self):
        """点击后重置计数，避免短时间内连发。"""
        self._hold_frames = -99999  # 直到下一次手势切换前不再触发

    def reset(self):
        """强制释放拖拽并清空状态（画面失焦/退出时调用）。"""
        if self._drag_active:
            pyautogui.mouseUp()
            self._drag_active = False
        self._sx = self._sy = None
        self._last_gesture = None
        self._hold_frames = 0

    # ---------------- 深度戳点击事件接口（Push-Click via camera distance）----------------
    def handle_depth_event(self, depth_event: int, screen_pt):
        """配合 depth_click_detector.py 使用。

        Args:
            depth_event: E.{NONE,DEPRESS,RELEASE,DRAG_MOVE} from DepthClickDetector.update()
            screen_pt:   (x_pixel, y_pixel) 屏幕坐标（已平滑），若未映射传 None 则跳过移动
        """
        # 延迟 import：避免 mouse_controller 依赖 depth_click_detector（反向依赖破环）
        from depth_click_detector import E as _DE
        if screen_pt is None:
            return
        ix, iy = int(screen_pt[0]), int(screen_pt[1])

        if depth_event == _DE.DEPRESS:
            # 向前戳触发 → 按下左键（若之前处于手势拖拽，先释放旧的）
            if self._drag_active:
                try: pyautogui.mouseUp()
                except Exception: pass
            pyautogui.mouseDown(ix, iy)
            self._drag_active = True
        elif depth_event == _DE.RELEASE:
            # 向后松开 → 释放左键
            pyautogui.mouseUp()
            self._drag_active = False
        elif depth_event == _DE.DRAG_MOVE:
            # 保持按住 → 拖拽移动
            pyautogui.moveTo(ix, iy)
            if not self._drag_active:
                # 保护：若处于 HOLD 但 drag_active=False（边界情况），自动补按下
                try:
                    pyautogui.mouseDown(ix, iy)
                    self._drag_active = True
                except Exception:
                    pass
        # NONE：什么都不做（不影响手势路径的鼠标状态）

    # ---------------- 食指 Tap-Click 事件接口（手指屈伸点击，手掌不动）----------------
    def handle_finger_tap_event(self, tap_event: int, screen_pt):
        """配合 finger_tap_click_detector.py 使用。

        Args:
            tap_event:   E.{NONE,TAP_CLICK,LONG_DOWN,LONG_UP,DRAG_MOVE}
            screen_pt:   (x_pixel, y_pixel) 屏幕坐标，未映射传 None 则跳过移动
        """
        from finger_tap_click_detector import E as _TE
        if screen_pt is None:
            if tap_event in (_TE.TAP_CLICK, _TE.DOUBLE_TAP, _TE.RESCUE_DBL_TAP, _TE.LONG_DOWN):
                print(f"[点击异常] 事件 ev={tap_event} 被丢弃：触发帧手不在画面(screen_pt=None)")
            return
        ix, iy = int(screen_pt[0]), int(screen_pt[1])

        if tap_event == _TE.TAP_CLICK:
            # ===== [CLICK-DIAG] 双击可行性诊断（OS 双击=时间+位置双门槛）=====
            try:
                import time as _t, ctypes as _ct
                _now = _t.time()
                _t0  = getattr(self, "_dbg_last_click_t", None)
                _xy0 = getattr(self, "_dbg_last_click_xy", None)
                _os_ms = _ct.windll.user32.GetDoubleClickTime()
                _os_rect = _ct.windll.user32.GetSystemMetrics(36)  # SM_CXDOUBLECLK
                if _t0 is not None and _xy0 is not None:
                    _raw_ix, _raw_iy = ix, iy
                    _gap_ms = (_now - _t0) * 1000.0
                    _dpos = ((ix - _xy0[0]) ** 2 + (iy - _xy0[1]) ** 2) ** 0.5
                    _time_ok = _gap_ms < _os_ms
                    # 位置吸附：OS 双击窗内且漂移 ≤ TAP_DBL_SNAP_PX → 吸到第一击精确像素
                    _snap_note = ""
                    if _time_ok and _dpos <= TAP_DBL_SNAP_PX:
                        ix, iy = _xy0
                        _snap_note = f" [吸附 {_raw_ix},{_raw_iy}→({ix},{iy}) Δpos归零]"
                    _pos_ok = _dpos <= _os_rect or _snap_note != ""
                    _verdict = "✔OS将识别为双击" if (_time_ok and _pos_ok) else \
                               ("✗间隔超%.0fms(实测%.0fms)" % (_os_ms, _gap_ms) if not _time_ok else "") + \
                               ("✗位置漂移%.0fpx超吸附阈(容差%.0fpx)" % (_dpos, TAP_DBL_SNAP_PX) if not _pos_ok else "")
                    if CLICK_DIAG:
                        print(f"[CLICK-DIAG] 第{getattr(self,'_dbg_click_n',1)+1}击 落点({_raw_ix},{_raw_iy}) "
                              f"Δt={_gap_ms:6.0f}ms(限{_os_ms}) 漂移={_dpos:5.1f}px(吸附阈{TAP_DBL_SNAP_PX}) → {_verdict}{_snap_note}",
                              flush=True)
                else:
                    if CLICK_DIAG:
                        print(f"[CLICK-DIAG] 第1击 ({ix},{iy})（下一次点击将打印间隔/落点差）", flush=True)
                self._dbg_last_click_t = _now
                self._dbg_last_click_xy = (ix, iy)
                self._dbg_click_n = getattr(self, "_dbg_click_n", 0) + 1
            except Exception:
                pass
            # 食指"勾一下回弹" → 完整 click：mouseDown+mouseUp
            # （即时模式=每次立刻 click，双击交给 OS 时间+位置双门槛判定）
            pyautogui.click(ix, iy, button="left")
            # 兜底保护：若因为历史遗留进入了drag状态，先释放
            if self._drag_active:
                try: pyautogui.mouseUp()
                except Exception: pass
            self._drag_active = False
        elif tap_event == _TE.DOUBLE_TAP:
            # 双击窗口内完成第二次"勾指回弹" → doubleClick
            # 第一次单击挂起未发，所以物理点击恰好 2 次 = 原生双击（不会多出第 3 次点击）
            pyautogui.doubleClick(ix, iy, button="left")
            if self._drag_active:
                try: pyautogui.mouseUp()
                except Exception: pass
            self._drag_active = False
        elif tap_event == _TE.RESCUE_DBL_TAP:
            # 慢节奏双击补救：补发 click#1 后 3.0s 补救窗内的第二击（真机两次勾指间隔 1.5~3.3s）。
            # 位置门控：落点与 click#1 ≤ RESCUE_DBL_SNAP_PX = 同一图标双击意图 → 原子 doubleClick
            # （内部两次点击间隔几十ms，OS 必判双击打开，规避"选中后慢点击=重命名"）；
            # 落点已移到别处 = 普通单击别的图标。
            _xy0 = getattr(self, "_dbg_last_click_xy", None)
            if _xy0 is not None:
                _dpos = ((ix - _xy0[0]) ** 2 + (iy - _xy0[1]) ** 2) ** 0.5
            else:
                _dpos = 0.0
            if _xy0 is not None and _dpos <= RESCUE_DBL_SNAP_PX:
                # 同一图标 → 吸附到 click#1 精确像素，原子双击打开
                if CLICK_DIAG:
                    print(f"[CLICK-DIAG] 🔁 补救双击(慢节奏) 落点({ix},{iy}) 距click#1 {_dpos:.1f}px≤{RESCUE_DBL_SNAP_PX}px "
                          f"→ doubleClick({_xy0[0]},{_xy0[1]}) 原子双击打开", flush=True)
                pyautogui.doubleClick(_xy0[0], _xy0[1], button="left")
                # 双击已由本次原子操作完成，清点击记录避免后续干扰
                self._dbg_last_click_xy = None
                self._dbg_last_click_t = None
            else:
                # 手已移到别的图标 → 降级普通单击
                if CLICK_DIAG:
                    print(f"[CLICK-DIAG] 补救第二击 落点({ix},{iy}) 距click#1 {_dpos:.1f}px>{RESCUE_DBL_SNAP_PX}px "
                          f"→ 判定为不同目标，降级普通click", flush=True)
                pyautogui.click(ix, iy, button="left")
                self._dbg_last_click_t = time.time()
                self._dbg_last_click_xy = (ix, iy)
            if self._drag_active:
                try: pyautogui.mouseUp()
                except Exception: pass
            self._drag_active = False
        elif tap_event == _TE.LONG_DOWN:
            # 食指长期保持弯曲 → 按住开始拖拽
            if self._drag_active:
                try: pyautogui.mouseUp()
                except Exception: pass
            pyautogui.mouseDown(ix, iy)
            self._drag_active = True
        elif tap_event == _TE.LONG_UP:
            # 食指回到伸直 → 释放拖拽
            pyautogui.mouseUp()
            self._drag_active = False
        elif tap_event == _TE.DRAG_MOVE:
            # 长按中持续移动
            pyautogui.moveTo(ix, iy)
            if not self._drag_active:
                try:
                    pyautogui.mouseDown(ix, iy)
                    self._drag_active = True
                except Exception:
                    pass
