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
  ⑤ 手势模式切换      — POINT=悬停(最大平滑+大死区) / OPEN=浏览 / FIST=拖拽(无死区)
                        TAP/PINCH/PEACE 前 → 锁定位置 3 帧，防止点击瞬间抖动漂移

同时保留：
  - 手势防抖 HOLD_FRAMES
  - 边缘死区 EDGE_MARGIN
  - 握拳拖拽状态机
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

# ---------------- AR Smoother 5 层参数 ----------------
# 1) 滑动中值窗口
MEDIAN_WINDOW = 5
# 2) 静止方差锁：近 N 帧屏幕像素位移的协方差 (px^2) 低于此值 → 锁死光标不动
LOCK_WINDOW = 8
#  不同模式下的锁死阈值 (px^2)，越大=越容易锁死=越稳
LOCK_THRESH_HOVER  = 0.80       # POINT 悬停：极稳（0.8px²≈0.9px单维抖动就锁）
LOCK_THRESH_BROWSE = 3.0        # OPEN 浏览：中等
LOCK_THRESH_DRAG   = 99999.0    # FIST 拖拽：不锁死（拖拽需要连续跟手）
# 3) 1€ Filter 核心参数（不同模式下 fc_min / beta 组合）
#    fc_min = 最小截止频率(Hz)，越小→慢动时越平滑
#    beta   = 速度自适应强度，越大→快动时越跟手
MODE_PARAMS = {
    # 手势名    (fc_min_Hz, beta,  deadzone_px, response_k)
    "POINT":  (0.30,  0.0040, 2.8, 3.0),   # 悬停：最强平滑+最大死区+柔响应
    "OPEN":   (0.80,  0.0100, 1.2, 1.6),   # 浏览：平衡
    "FIST":   (1.50,  0.0250, 0.0, 1.2),   # 拖拽：最跟手+零死区+更线性
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
CURSOR_MODE_ABSOLUTE = "absolute"  # ✨ 掌心绝对坐标直接跳（像触摸屏指哪打哪，方案A 3层流水线：中值杀尖峰 + Sigmoid微移阻尼 + 静止方差锁）

# absolute 模式下为了避免 MediaPipe 1~2 帧偶发尖峰跳到屏幕角落，保留一个轻量的 3 帧中值
ABS_MEDIAN_WINDOW = 3

# ✨✨ 方案 A 新增：absolute 模式「大范围跟手 + 小范围不晃」的 3 层流水线参数（只作用在 absolute 模式，不影响 smooth）
#   Layer 2 Sigmoid：微移阻尼增益（响应曲线拐点，k 越大=拐点越陡=越严格压制小抖动）
#     调参语义：ABS_SIGMOID_K = 0.35  → 位移 <3px 增益≈0.2（5倍阻尼）；>20px 增益→1.0（完全不衰减=跟手）
#     不想要 Sigmoid（嫌麻烦）→ 设 0.0 即可直接关闭这一层
ABS_SIGMOID_K     = 0.35
ABS_SIGMOID_MIN_G = 0.20  # 最小增益（Δ→0 时的阻尼）；0.20 = 5 倍阻尼（1px抖→0.2px输出）
#   Layer 3 静止方差锁：连续 N 帧内，掌心样本方差 σ² < threshold² → 硬锁死光标到一个点（钉住=完全不漂移）
#     调参语义：ABS_LOCK_WINDOW=10 帧 / ABS_LOCK_THRESH=2.5 px
#       → 连续 10 帧中（30fps≈333ms）每帧抖动都 < ±2.5px → 锁定；锁定后 只要出现 1 帧 > 4px 立刻解锁（响应真实动作）
ABS_LOCK_WINDOW   = 10   # 连续多少帧进入锁死判定（太短锁不住太长反应慢）
ABS_LOCK_THRESH   = 2.5  # 静止判定阈值（像素）：方差 σ < 这个值就视为「手没动=纯噪声」

# pyautogui 安全设置
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.0


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
        self._abs_locked_x: float | None = None   # 当前被锁死的固定光标 X（None=未锁）
        self._abs_locked_y: float | None = None   # 当前被锁死的固定光标 Y（None=未锁）

    # ---------- 外部：模式切换 ----------
    def set_mode(self, mode: str):
        """切换 cursor 模式：smooth / absolute"""
        if mode == self._mode:
            return
        assert mode in (CURSOR_MODE_SMOOTH, CURSOR_MODE_ABSOLUTE), f"unknown cursor mode: {mode}"
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
        # ✨ 用户要求：sigmoid 增益压缩「增大到 5px」
        #   目标：Δ ≤ 5px 全部都落在「接近最小增益 0.20」的位置 → 阻尼 ≥ 3.3 倍以上
        #   计算方法（确保 5px 处真正在 Sigmoid 贴底区= inflection - (2/K) ≈ 5）：
        #     K=0.35 → 2/K ≈ 5.71 → inflection ≈ 5 + 5.71 ≈ 11.0
        #   最终 INFLECTION_PX = 11.0：
        #     Δ=1px → 增益≈0.22（几乎贴底）
        #     Δ=3px → 增益≈0.25
        #     Δ=5px → 增益≈0.29（=刚好在 5px 压缩边界，3.4 倍阻尼）
        #     Δ=11px→ 增益≈0.60（拐点中点）
        #     Δ=20px→ 增益≈0.98（完全不衰减，大范围跟手）
        INFLECTION_PX = 11.0
        s = 1.0 / (1.0 + np.exp(-ABS_SIGMOID_K * (float(delta_px) - INFLECTION_PX)))
        min_g = float(max(0.0, min(1.0, ABS_SIGMOID_MIN_G)))
        amp = 1.0 - min_g
        # 输出 ∈ [min_g, 1.0]；默认 min_g=0.20 → [0.20, 1.0]
        return float(min_g + amp * s)

    # ---------- 主入口：5 层流水线 ----------
    def filter(self, raw_x: float, raw_y: float, gesture: str | None = None):
        """输入=归一化坐标映射后的「屏幕像素原始值」，输出=AR平滑后屏幕像素。

        两种模式：
          - CURSOR_MODE_SMOOTH: 完整 5 层 AR 悬停流水线（方差锁/1€/Sigmoid 等）
          - CURSOR_MODE_ABSOLUTE: 方案 A 3 层流水线 → 大范围跟手（指哪打哪）+ 小范围不晃（微移阻尼+静止锁）
        """
        # 点击前冻结（两种模式都生效：点击瞬间锁死避免漂移）
        if self._freeze_left > 0:
            self._freeze_left -= 1
            fx, fy = self._freeze_x, self._freeze_y
            self._last_out_x = fx
            self._last_out_y = fy
            return fx, fy

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
            #   ✨ 用户要求「sigmoid 增益压缩增大到 5px」：压缩生效边界=5px（Δ≤5px 全压）
            #   原理：用「Layer1 输出 - 上一帧输出」计算位移 Δ；套 Sigmoid 得增益 g(Δ)（拐点=11px）：
            #     Δ ≤ 5px  → g ∈ [0.22, 0.29]          → 阻尼 ≥ 3.4 倍（5px 内微抖 1px→0.22px 输出,完全不晃）
            #     Δ = 11px → g ≈ 0.60                  → 拐点, ~1.7 倍阻尼（微调场景略稳）
            #     Δ > 20px → g → 1.0                   → 0 衰减（大范围移动=和以前的absolute一样跟手）
            #   关键：绝对位置不做 EMA/1€（这两种引入相位滞后=大范围拖尾），只对「这一帧和上一帧的差值」按增益缩放
            #         → 大范围仍然是绝对指哪打哪，不会有滞后
            # ============================================================
            if self._last_out_x is None or self._last_out_y is None:
                # 第一帧：没有上一帧，直接用 Layer1 输出（避免除以 0 / 跳变）
                sx, sy = float(ax), float(ay)
            else:
                dx = float(ax) - float(self._last_out_x)
                dy = float(ay) - float(self._last_out_y)
                delta_px = float(np.sqrt(dx * dx + dy * dy))
                if ABS_SIGMOID_K <= 1e-6:
                    # 参数设为 0 → 关闭 Layer2
                    g = 1.0
                else:
                    g = float(self._abs_sigmoid_gain(delta_px))
                sx = float(self._last_out_x) + dx * g
                sy = float(self._last_out_y) + dy * g

            # ============================================================
            # Layer 3：静止方差锁（对准小按钮的核心杀招 → 光标"钉住"完全不动）
            #   判定：连续 ABS_LOCK_WINDOW=10 帧的 Layer2 输出，在 X/Y 两个方向上 σ（标准差）< ABS_LOCK_THRESH=2.5px
            #         → 视为「手掌完全没动 = MediaPipe 纯噪声抖动」→ 硬锁死到这 10 帧的均值点（之后直接输出锁点，不再随噪声漂）
            #   解锁：已锁死的情况下，这一帧 Layer2 输出与锁点的距离 > UNLOCK=4.0px（说明手真的动了）→ 立刻解锁
            # ============================================================
            UNLOCK_DIST = float(ABS_LOCK_THRESH) * 1.6   # 解锁阈值 = 锁定阈值的 1.6 倍，避免锁/解临界反复切换
            self._abs_lock_buf_x.append(float(sx))
            self._abs_lock_buf_y.append(float(sy))

            buf_full = len(self._abs_lock_buf_x) >= ABS_LOCK_WINDOW and len(self._abs_lock_buf_y) >= ABS_LOCK_WINDOW
            # ① 已在锁死状态 → 先检查是否需要解锁
            if self._abs_locked_x is not None and self._abs_locked_y is not None:
                dist_to_lock = float(np.sqrt(
                    (float(sx) - float(self._abs_locked_x)) ** 2 +
                    (float(sy) - float(self._abs_locked_y)) ** 2
                ))
                if dist_to_lock <= UNLOCK_DIST:
                    # 仍然在锁点附近 → 继续锁死，输出固定点
                    lx, ly = float(self._abs_locked_x), float(self._abs_locked_y)
                    self._last_out_x = lx
                    self._last_out_y = ly
                    return lx, ly
                else:
                    # 手真的动了 → 解锁；清空锁缓冲（下一轮再满 10 帧才重新锁）
                    self._abs_locked_x = None
                    self._abs_locked_y = None
                    self._abs_lock_buf_x.clear()
                    self._abs_lock_buf_y.clear()
                    self._abs_lock_buf_x.append(float(sx))
                    self._abs_lock_buf_y.append(float(sy))
                    buf_full = False  # 刚清空 → 缓冲不满，不做锁判定
            # ② 未锁死 → 尝试满足「缓冲满 + 方差 < 阈值」→ 进入锁死
            if buf_full and self._abs_locked_x is None:
                std_x = float(np.std(self._abs_lock_buf_x))
                std_y = float(np.std(self._abs_lock_buf_y))
                if std_x < ABS_LOCK_THRESH and std_y < ABS_LOCK_THRESH:
                    # 满足静止条件 → 锁死到均值位置
                    lock_x = float(np.mean(self._abs_lock_buf_x))
                    lock_y = float(np.mean(self._abs_lock_buf_y))
                    self._abs_locked_x = lock_x
                    self._abs_locked_y = lock_y
                    # 锁死首帧：直接返回锁点
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
            "FIST":  LOCK_THRESH_DRAG,
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
                 roi_scale: float = DEFAULT_ROI_SCALE, **_ignored):
        if screen_size is None:
            self.sw, self.sh = pyautogui.size()
        else:
            self.sw, self.sh = screen_size
        # ROI scale：画面中心多少%映射到整个屏幕（解决"掌心锁死、手指不可见"）
        roi_scale = float(roi_scale)
        if roi_scale < ROI_SCALE_MIN: roi_scale = ROI_SCALE_MIN
        if roi_scale > ROI_SCALE_MAX: roi_scale = ROI_SCALE_MAX
        self._roi_scale = roi_scale
        # AR 悬停 5 层平滑器（支持 smooth / absolute 两种 cursor 策略）
        self._smoother = ARSmoother(mode=cursor_mode)
        # 防抖状态
        self._last_gesture = None
        self._hold_frames = 0
        # 拖拽状态
        self._drag_active = False

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

    def set_cursor_mode(self, mode: str):
        """切换光标策略：smooth（AR悬停5层）/ absolute（掌心绝对坐标直接跳）"""
        self._smoother.set_mode(mode)

    @property
    def cursor_mode(self) -> str:
        return self._smoother.mode

    # ---------------- 内部工具 ----------------
    def _smooth(self, nx: float, ny: float, gesture: str | None = None):
        return self._smoother.filter(nx, ny, gesture=gesture)

    def _norm_to_screen(self, norm_x: float, norm_y: float):
        """landmark 归一化坐标 -> 屏幕像素。

        映射规则（用户可调参数 roi_scale）：
          设 roi = 0.85：
            画面中心 85% 的 [0.075, 0.925] × [0.075, 0.925] 区域 映射 屏幕 (0,0) ~ (W,H)
            掌心超出 ROI 区域（画面四周 7.5% 每边）会被 clamp 到屏幕边缘，但：
             ✨ 这 7.5% 正好提供"手指还能自由伸展但掌心已到屏幕边缘"的活动空间，
             彻底解决原 EDGE_MARGIN 机制下「掌心一碰边缘就锁死、手指仍在画面但动了无效」问题。
        """
        r = self._roi_scale
        margin = (1.0 - r) * 0.5   # 四周留白 = (1-r)/2
        denom = max(r, 1e-6)
        x = (float(norm_x) - margin) / denom
        y = (float(norm_y) - margin) / denom
        x = 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
        y = 0.0 if y < 0.0 else (1.0 if y > 1.0 else y)
        return x * self.sw, y * self.sh

    # ---------------- 对外主接口 ----------------
    def handle(self, gesture: str, fingertip_norm, cursor_anchor_norm=None):
        """手势 → 鼠标事件。

        Args:
            gesture: 如 PEACE / PINCH / OPEN / POINT / FIST / ROCK / THUMB_UP / OTHER
            fingertip_norm: 手势识别用锚点（食指TIP 8，归一化 0~1），保留兼容旧调用
            cursor_anchor_norm: 实际鼠标光标锚点，默认掌心 = (lms[0]+lms[9])/2。
                                None 时回退用 fingertip_norm（兼容）。
        """
        anchor_norm = cursor_anchor_norm if cursor_anchor_norm is not None else fingertip_norm
        # 1. 坐标映射 + AR 5 层平滑（把当前手势作为模式上下文传入）
        gx, gy = self._norm_to_screen(*anchor_norm)
        sx, sy = self._smooth(gx, gy, gesture=gesture)
        ix, iy = int(round(sx)), int(round(sy))

        # 2. 防抖状态机
        if gesture != self._last_gesture:
            self._last_gesture = gesture
            self._hold_frames = 0
            if not self._drag_active:
                pyautogui.moveTo(ix, iy)
            return
        self._hold_frames += 1

        # 3. 移动类手势：未满 HOLD_FRAMES 也持续移动
        if gesture in ("OPEN", "POINT", "OTHER"):
            if self._drag_active:
                pyautogui.mouseUp()
                self._drag_active = False
            pyautogui.moveTo(ix, iy)
            return

        # 未满 HOLD_FRAMES 的点击类手势 → 仅移动
        if self._hold_frames < HOLD_FRAMES:
            pyautogui.moveTo(ix, iy)
            return

        # 4. 点击类手势：触发前先冻结位置
        if gesture in ("PEACE", "PINCH", "ROCK", "THUMB"):
            self._smoother.freeze_for_click()

        if gesture == "PEACE":
            pyautogui.click(ix, iy, button="left")
            self._consume_gesture()
        elif gesture == "PINCH":
            pyautogui.click(ix, iy, button="left")
            self._consume_gesture()
        elif gesture == "ROCK":
            pyautogui.click(ix, iy, button="right")
            self._consume_gesture()
        elif gesture == "THUMB":
            pyautogui.doubleClick(ix, iy)
            self._consume_gesture()
        elif gesture == "FIST":
            # 握拳 → 拖拽 (按下 + 移动)
            if not self._drag_active:
                pyautogui.mouseDown(ix, iy)
                self._drag_active = True
            pyautogui.moveTo(ix, iy)
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
            return
        ix, iy = int(screen_pt[0]), int(screen_pt[1])

        if tap_event == _TE.TAP_CLICK:
            # 食指"勾一下回弹" → 完整 click：mouseDown+mouseUp
            pyautogui.click(ix, iy, button="left")
            # 兜底保护：若因为历史遗留进入了drag状态，先释放
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
