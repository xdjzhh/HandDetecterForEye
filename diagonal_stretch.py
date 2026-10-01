# -*- coding: utf-8 -*-
"""
双手对角拉伸 → 选中当前前台窗口 → 放大缩小（保持窗口中心不变）
==============================================================
语义（符合用户直觉）：
  两只手分别落在「画面左上 + 右下」或「左下 + 右上」两个对角象限 →
  持续保持 8 帧 ≈ 270ms → 视为"选中" GetForegroundWindow 前台窗口（记住它的位置/尺寸）→
  之后两手间距变大 = 窗口等比放大；两手间距变小 = 窗口等比缩小（中心不动）→
  任意一只手离开画面 or 距离骤减回到 200px 以下 → 结束，回到 IDLE。

依赖：
  Windows + ctypes.user32（纯标准库，无需 pip 新依赖）
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import math
import time
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


# ============================================================
# Win32 API（ctypes 签名，严格与 user32.dll 导出符号一致）
# ============================================================
user32 = ctypes.WinDLL("user32", use_last_error=True)


class RECT(ctypes.Structure):
    _fields_ = [
        ("left",   wintypes.LONG),
        ("top",    wintypes.LONG),
        ("right",  wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


GetForegroundWindow = user32.GetForegroundWindow
GetForegroundWindow.argtypes = []
GetForegroundWindow.restype  = wintypes.HWND

GetWindowRect = user32.GetWindowRect
GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(RECT)]
GetWindowRect.restype  = wintypes.BOOL

SetWindowPos = user32.SetWindowPos
SetWindowPos.argtypes = [
    wintypes.HWND, wintypes.HWND,
    wintypes.INT,  wintypes.INT,
    wintypes.INT,  wintypes.INT,
    wintypes.UINT,
]
SetWindowPos.restype = wintypes.BOOL

# ✨ 新增：跳鼠标光标到屏幕指定像素点（选中窗口后把光标放中心，避免误点其他窗口）
SetCursorPos = user32.SetCursorPos
SetCursorPos.argtypes = [wintypes.INT, wintypes.INT]
SetCursorPos.restype  = wintypes.BOOL

# SetWindowPos 常用标志：不改 Z 序 / 不抢激活焦点
SWP_NOZORDER   = 0x0004
SWP_NOACTIVATE = 0x0010
SWP_NOSIZE     = 0x0001   # 移动窗口专用：只改 x/y，不改 w/h
SWP_FLAGS      = SWP_NOZORDER | SWP_NOACTIVATE
SWP_MOVE_FLAGS = SWP_NOZORDER | SWP_NOACTIVATE | SWP_NOSIZE
HWND_TOP       = 0  # 仅当需要插 Z 序时用，我们用 SWP_NOZORDER 所以此参数无效，填 0

# ✨ 新增：根据屏幕坐标点 (x,y) 找到该点下的窗口（=用户掌心覆盖的窗口）
WindowFromPoint = user32.WindowFromPoint
WindowFromPoint.argtypes = [wintypes.POINT]
WindowFromPoint.restype  = wintypes.HWND

# ✨ 新增：把窗口强制置为前台（激活到最前，便于操作）
SetForegroundWindow_API = user32.SetForegroundWindow
SetForegroundWindow_API.argtypes = [wintypes.HWND]
SetForegroundWindow_API.restype  = wintypes.BOOL

# ✨✨ 解决 WPS/Word/Chrome 命中子控件(标签页)问题的核心API：把命中点的 HWND 一直追溯到"顶层所有者窗口"=应用主窗口
#   GetAncestor(hwnd, GA_ROOTOWNER=3) = 跳过所有子控件/子对话框，一直追溯到"拥有者"窗口（这就是应用主窗口，绝对不会是标签页/工具栏/菜单栏子控件）
#   GetAncestor(GA_ROOT=2) = 爬到顶层父窗口（作为 GA_ROOTOWNER 的兜底）
GetAncestor = user32.GetAncestor
GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
GetAncestor.restype  = wintypes.HWND
GA_PARENT     = 1
GA_ROOT       = 2
GA_ROOTOWNER  = 3   # ✨ 主要用这个：追溯到"顶层所有者窗口"=应用主窗口

# ✨ 校验 HWND 是否为"可移动顶层"（不是子控件 = 不是标签页）：GetWindowLongPtr 取样式
#   64 位/32 位 Windows 都用 GetWindowLongPtrW（Win9x 以上都支持，比 GetWindowLong 安全）
# ✨ 兼容性补丁（经验 866818/1293388）：
#   旧版 Python / 32 位 Python 的 ctypes.wintypes 缺少 ULONG_PTR / LONG_PTR 这两个"指针长度整数"类型 → 手动补上，避免 AttributeError
if not hasattr(wintypes, 'ULONG_PTR') or not hasattr(wintypes, 'LONG_PTR'):
    import ctypes as _ct_local
    _ptr_size = _ct_local.sizeof(_ct_local.c_void_p)  # 64 位=8 / 32 位=4
    if _ptr_size == 8:
        # 64 位 Windows：LPTR = 64 位 longlong
        if not hasattr(wintypes, 'ULONG_PTR'):
            wintypes.ULONG_PTR = _ct_local.c_ulonglong
        if not hasattr(wintypes, 'LONG_PTR'):
            wintypes.LONG_PTR  = _ct_local.c_longlong
    else:
        # 32 位 Windows：LPTR = 32 位 long
        if not hasattr(wintypes, 'ULONG_PTR'):
            wintypes.ULONG_PTR = _ct_local.c_ulong
        if not hasattr(wintypes, 'LONG_PTR'):
            wintypes.LONG_PTR  = _ct_local.c_long
try:
    GetWindowLongPtrW = user32.GetWindowLongPtrW
except Exception:
    GetWindowLongPtrW = user32.GetWindowLongW   # 32 位系统兜底
GetWindowLongPtrW.argtypes = [wintypes.HWND, wintypes.INT]
GetWindowLongPtrW.restype  = wintypes.ULONG_PTR  # 64 位上是 64 位值（此时 ULONG_PTR 已通过上面补丁存在）
GWL_STYLE = -16   # 窗口样式偏移（GWL_EXSTYLE=-20 暂时不用）
# 关键样式位：
WS_CHILD    = 0x40000000   # 带这个 = 子控件/标签页/工具栏 → 必须排除（绝对不能直接移动子控件！）
WS_CAPTION  = 0x00C00000   # 带这个 = 有标题栏（标题栏窗口 = 正常可拖动的顶层主窗口；WPS/Word/Chrome 的主窗口必然带）
WS_POPUP    = 0x80000000   # 带这个 = 弹窗样式（某些系统弹窗/右键菜单/对话框也可能带；兜底允许）
WS_OVERLAPPED = 0x00000000 # 默认普通顶层窗口样式


# ============================================================
# 状态常量
# ============================================================
class _S:
    IDLE           = "IDLE"              # 无有效双手捏合形态
    ARMED_CONFIRM  = "ARMED_CONFIRM"     # 捏合+距离条件连续计数中（防抖动）
    SCALING        = "SCALING"           # 缩放中（每帧更新窗口 size）


class _MODE:
    """双手模式：同一时刻只能处于其中一种（互斥）"""
    SCALE = "SCALE"   # 双手非OPEN=捏合 → 拉伸缩放
    MOVE  = "MOVE"    # 双手OPEN=五指张开 → 移动窗口位置（不改 size）


class _MS:
    """移动窗口状态机（与_SCALE状态机独立，互斥切换）"""
    IDLE          = "IDLE"
    MOVE_CONFIRM  = "MOVE_CONFIRM"   # 双手五指张开 确认中
    MOVE_ACTIVE   = "MOVE_ACTIVE"    # 已选中掌心下的窗口 正在移动

# 墙钟时长（time.monotonic 秒，2026-10-01 统一由帧计数改为秒，物理时长不再随帧率漂移）
MOVE_CONFIRM_SEC  = 0.10   # 移动窗口确认（原 3 帧 @30fps）


class _M:
    """拉伸模式：按「进入 SCALING 瞬间两手的水平/垂直距离比例」判定，判定一次就不再改。"""
    HORIZONTAL = "HORIZONTAL"   # 两手主要左右排开 → 只改宽度（拉宽/拉窄），高度不变
    VERTICAL   = "VERTICAL"     # 两手主要上下排开 → 只改高度（拉高/拉低），宽度不变
    DIAGONAL   = "DIAGONAL"     # 接近对角 → 保留原等比缩放（宽高同比例）


# ------------------------------------------------------------
# 🔥 用户新方案：按「两手连线与水平线夹角θ」直接分三档判拉伸方向（一次判定永不改）
#    θ = atan2(|垂直距离|, |水平距离|) → 0° 为水平，90° 为纯垂直（不管左右手，只看倾斜角度，取 0~90° 绝对值）
#      [0°, 20°]  包含边界 → HORIZONTAL 拉宽（只改宽度）
#      (20°, 70°) 开区间  → DIAGONAL   对角等比缩放（宽高同比例）
#      [70°, 90°] 包含边界 → VERTICAL   拉高（只改高度）
# ------------------------------------------------------------
MODE_ANGLE_HORIZ_MAX_DEG = 20.0   # 最大多少度以内仍算水平拉宽（包含边界=20°）
MODE_ANGLE_VERT_MIN_DEG  = 70.0   # 最小多少度以上才算垂直拉高（包含边界=70°）
# 原基于比值/最小像素的阈值已删除（新方案完全基于几何角度，更直观符合用户直觉）


@dataclass
class StretchInfo:
    state: str = _S.IDLE
    hwnd: int = 0
    ratio: float = 1.0                # 当前缩放比例（相对初始值；单维模式下=该维度的比例）
    win_w: int = 0
    win_h: int = 0
    diag_dist_px: float = 0.0         # 当前两掌心像素距离
    p1: Optional[Tuple[float, float]] = None  # 画面里两只掌心像素坐标（画连线/中心用）
    p2: Optional[Tuple[float, float]] = None
    confirm_sec: float = 0.0           # 缩放确认累计秒数
    left_pinch_r: float = -1.0        # 左手捏合比（<0=未检测），‖TIP4-TIP8‖/palm_width，越小越紧
    right_pinch_r: float = -1.0       # 右手捏合比
    stretch_mode: str = _M.DIAGONAL   # ✨ 当前拉伸模式：水平/垂直/对角
    need_frame_redraw: bool = False   # UI：是否需要画捏合提示框/缩放比
    ui_pinch_active: bool = False     # ✨ UI专用：双手+都非OPEN=立刻True（提前显示连线/锚点，视觉先反馈；状态机确认仍用于真改窗口）
    # -------- ✨ 新增：双手五指张开 → 移动窗口 模式 --------
    mode: str = _MODE.SCALE           # 当前双手互斥模式：SCALE（捏合缩放）/ MOVE（张开移动）
    hands_mode: str = "DUAL"          # ✨ 新增：MOVE模式下 是「SINGLE单手」还是「DUAL双手」张开（UI画法不同）
    ui_move_active: bool = False      # UI专用：双手+都OPEN五指张开=立刻True（提前显示移动连线，视觉先反馈）；单手+OPEN也立刻True
    move_state: str = _MS.IDLE        # 移动状态机当前状态：IDLE/MOVE_CONFIRM/MOVE_ACTIVE
    move_hwnd: int = 0                # 正在移动的窗口 HWND
    move_confirm_sec: float = 0.0     # 移动确认累计秒数
    move_window_rect: Optional[Tuple[int,int,int,int]] = None  # (left, top, right, bottom) 当前移动窗口的屏幕矩形
    move_delta_xy: Tuple[float, float] = (0.0, 0.0)           # 本次移动累计偏移 (Δx, Δy) 像素
    move_need_frame_redraw: bool = False  # UI：移动模式是否需要画提示卡片


# ============================================================
# 主类：双手对角拉伸 → 窗口缩放状态机
# ============================================================
class DiagonalStretchManager:
    # ✨✨ 稳定性常数（防止 ACTIVE 时单帧抖动导致"指令中断/重新识别"）：
    #   LOST=0.13s ≈ 4 帧 @30fps；既能覆盖 MediaPipe 偶发漏检 / OPEN-PINCH 单帧抖动，又能保证手势真正消失时及时退出
    SCALE_LOST_MAX_SEC: float = 0.13  # SCALE_ACTIVE 下，连续多久不满足条件才真正退出到 IDLE
    MOVE_LOST_MAX_SEC:  float = 0.13  # MOVE_ACTIVE 下，连续多久不满足条件才真正退出到 IDLE
    # OPEN/PINCH 判定稳定（混合双条件：至少 N 个一致样本 AND 至少持续 N 秒，低帧率防单帧坏数据、高帧率不引入额外延迟）
    OPEN_STABLE_N:   int   = 2
    OPEN_STABLE_SEC:  float = 0.067

    def __init__(self, *,
                 min_hold_sec: float = 0.10,
                 min_diag_px: float = 120.0,  # ✨ 原 220 太严，降到 120（两手靠得稍近也能进入拉伸确认帧）
                 # ✨ 新增：双手捏合判定阈值（与 gesture_recognizer._pinch_distance 同尺度，‖TIP4-TIP8‖/palm_width）
                 #   注意：当前捏合判定已改为「非OPEN即捏合」，这里的 pinch_enter/hold 仅用于滞回（enter<=hold）兼容，实际阈值不影响判定结果
                 pinch_enter: float = 0.40,
                 pinch_hold:  float = 0.50,
                 # ✨ 2026-10-01 第二批墙钟化：以下两个参数改为"墙钟物理量"，低帧率不再失真
                 #   smooth_tau_sec：EMA 时间常数（原 smooth_alpha=0.75 @30fps → τ=(1/30)/ln4≈0.024s）
                 #     每帧 alpha = 1 - exp(-dt/τ)：30fps→0.75，10fps→0.984，120fps→0.293
                 smooth_tau_sec: float = 0.024,
                 # max_step_per_sec：缩放比例每秒最大变化量（原 0.18/帧 @30fps → 5.4/s），每帧限幅 = 该值×dt
                 max_step_per_sec: float = 5.4,
                 min_win_w: int = 240, min_win_h: int = 160,
                 max_win_w: int = 5120, max_win_h: int = 2880,
                 sensitivity: float = 1.0,
                 ):
        self.state = _S.IDLE
        self.info = StretchInfo()

        # 参数
        self.min_hold_sec   = max(0.0, float(min_hold_sec))   # 双手捏合确认墙钟时长
        self.min_diag_px        = float(max(60.0, min_diag_px))
        # 捏合阈值滞回约束：enter 必须 <= hold（经验 630282 阈值体系 enter < exit/hold 才稳定）
        if pinch_enter > pinch_hold:
            pinch_enter, pinch_hold = pinch_hold, pinch_enter
        self._pinch_enter = float(max(0.0, min(1.0, pinch_enter)))
        self._pinch_hold  = float(max(0.0, min(1.0, pinch_hold)))
        self.smooth_tau_sec   = max(1e-4, float(smooth_tau_sec))
        self.max_step_per_sec  = max(0.01, float(max_step_per_sec))
        self.min_win_w, self.min_win_h = int(min_win_w), int(min_win_h)
        self.max_win_w, self.max_win_h = int(max_win_w), int(max_win_h)
        self.sensitivity        = 0.2 if sensitivity < 0.2 else (3.0 if sensitivity > 3.0 else float(sensitivity))

        # -------- ✨ 新增：移动窗口(MOVE)状态机（双手五指张开触发） --------
        self._move_state: str = _MS.IDLE
        self._move_confirm: float = 0.0   # 确认累计秒数
        self._move_hwnd: int = 0
        # 进入 MOVE_ACTIVE 瞬间的"参考手掌心屏幕坐标"（主手=左手=用户视角右手）
        self._move_ref_sx: float = 0.0
        self._move_ref_sy: float = 0.0
        # 进入 MOVE_ACTIVE 瞬间的窗口位置和大小快照（只移动，不缩放）
        self._move_left0: float = 0.0
        self._move_top0:  float = 0.0
        self._move_w0:    float = 0.0
        self._move_h0:    float = 0.0

        # 运行时状态（SCALING 时保存的初始快照 + 双手捏合状态记忆用于滞回）
        self._d0: float | None = None
        self._dx0: float = 0.0   # ✨ 进入 SCALING 瞬间：两手水平距离分量 |rx-lx|
        self._dy0: float = 0.0   # ✨ 进入 SCALING 瞬间：两手垂直距离分量 |ry-ly|
        self._stretch_mode: str = _M.DIAGONAL   # ✨ 本次拉伸主导模式（按进入瞬间判定，之后不变）
        self._hwnd: int = 0
        self._cx0: float = 0.0
        self._cy0: float = 0.0
        self._w0: float = 0.0
        self._h0: float = 0.0
        self._smooth_ratio: float | None = None
        self._last_ratio: float = 1.0
        self._confirm: float = 0.0   # 捏合确认累计秒数
        self._left_pinch_prev: bool = False   # 上一帧左手是否处于捏合（用于滞回 hold 判定）
        self._right_pinch_prev: bool = False  # 上一帧右手是否处于捏合

        # ✨✨ 2026-08-13 新增：状态机抖动克星（解决用户「移动窗口到一半突然重新识别」）
        self._scale_lost_sec: float = 0.0   # SCALE_ACTIVE 下：连续多久不满足条件
        self._move_lost_sec:  float = 0.0   # MOVE_ACTIVE 下：连续多久不满足条件
        # OPEN 手势判定的"稳定滞回缓存"：
        # tag(手) → (已提交输出, 候选新值, 候选样本数, 候选累计秒)；样本数+时长双条件才提交
        self._open_stable: dict[str, tuple[bool, bool, int, float]] = {}

    # ------------------------------------------------------------
    # ✨ 判定稳定滞回助手：对「手势 OPEN/PINCH」这类连续判定做"连续 N 个一致样本且持续 N 秒才切换输出" → 彻底防单帧抖动
    #   tag: 区分不同手/来源（"L"=左手 / "R"=右手 / "S"=单手）
    #   raw: 这一帧的原始判定结果（True/False）
    #   return: 经过稳定化后的最终判定结果（不会一帧就变，必须样本数+时长双满足）
    # ------------------------------------------------------------
    def _stabilize_bool(self, tag: str, raw: bool, dt: float = 0.0) -> bool:
        try:
            N   = int(max(1, DiagonalStretchManager.OPEN_STABLE_N))
            SEC = float(max(0.0, DiagonalStretchManager.OPEN_STABLE_SEC))
            cur, pend_v, pend_n, pend_sec = self._open_stable.get(
                tag, (bool(raw), bool(raw), 0, 0.0))
            if bool(raw) == bool(cur):
                # 与已提交输出一致：清候选，输出不变
                self._open_stable[tag] = (bool(cur), bool(cur), 0, 0.0)
                return bool(cur)
            # 与输出相反：累计候选（同值续计，异值重开）
            if int(pend_n) > 0 and bool(pend_v) == bool(raw):
                pend_n, pend_sec = int(pend_n) + 1, float(pend_sec) + float(dt)
            else:
                pend_n, pend_sec = 1, float(dt)
            if pend_n >= N and pend_sec >= SEC:
                # 样本数与时长双满足 → 提交翻转
                self._open_stable[tag] = (bool(raw), bool(raw), 0, 0.0)
                return bool(raw)
            self._open_stable[tag] = (bool(cur), bool(raw), pend_n, pend_sec)
            return bool(cur)
        except Exception:
            # 任意异常 → 直接返回原始结果（保证不会因为稳定器崩导致手势不可用）
            return bool(raw)

    # ------------------------------------------------------------
    # ✨ 工具：ACTIVE→LOST 容忍墙钟检测（MOVE/SCALE 共用逻辑）
    #   作用：当 状态机=ACTIVE 且"当前帧条件不满足"时，不立刻reset，先累计 lost 秒；到 MAX 才真的 reset
    #   返回：True=已经被容忍（不要reset继续保持ACTIVE）；False=忍无可忍必须reset
    # ------------------------------------------------------------
    def _try_tolerate_lost_scale(self, dt: float = 0.0) -> bool:
        """SCALE_ACTIVE 用：返回 True=已容忍，保持 ACTIVE（不要 reset）"""
        self._scale_lost_sec = float(self._scale_lost_sec) + float(dt)
        if self._scale_lost_sec < float(DiagonalStretchManager.SCALE_LOST_MAX_SEC):
            return True
        return False

    def _try_tolerate_lost_move(self, dt: float = 0.0) -> bool:
        """MOVE_ACTIVE 用：返回 True=已容忍，保持 ACTIVE（不要 reset）"""
        self._move_lost_sec = float(self._move_lost_sec) + float(dt)
        if self._move_lost_sec < float(DiagonalStretchManager.MOVE_LOST_MAX_SEC):
            return True
        return False

    # ------------------------------------------------------------
    # 静态辅助：两向量夹角（度），0°~180°；返回 inf 表示向量无效（0向量）
    # ------------------------------------------------------------
    @staticmethod
    def _angle_deg(v1, v2) -> float:
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1e-6 or n2 < 1e-6:
            return float("inf")
        cos = float(np.dot(v1, v2)) / (n1 * n2)
        cos = max(-1.0, min(1.0, cos))
        return float(math.degrees(math.acos(cos)))

    # ------------------------------------------------------------
    # 静态辅助：单手握「五指张开(OPEN)」判定 —— 三规则必须同时满足（缺一不可，避免手指并拢/重合被误判为张开）
    #   规则(1) 共线度：每根手指的「掌指→近节→中节→远节」三段弯曲角都 < 40°（手指明显伸直=共线）
    #   规则(2) 散开度：5 个指尖点的「外接矩形面积 / 掌宽²」≥ 0.25（五指明显分开=不是并拢/重合的状态）
    #   规则(3) 外于掌：每个指尖到 wrist(0) 的距离 > 对应掌指关节(MCP/CMC)到 wrist 的 1.15 倍（指尖明显在掌外=不弯回手心/贴手心）
    #   只有三规则 AND 通过 → 返回 True（真·五指张开=OPEN → 非捏合）；其他任何情况（手指重合/并拢/部分弯曲）→ False=视作捏合手势
    # ------------------------------------------------------------
    @staticmethod
    def _is_open_palm(lms) -> bool:
        try:
            # ===== 通用：2D 关键点向量（只用 x,y，符合 MediaPipe 2D 场景量纲一致） =====
            def P(i):
                return np.array([lms[i].x, lms[i].y], dtype=np.float64)

            wrist = P(0)
            # 掌宽尺度：index_mcp(5) → pinky_mcp(17)
            p5  = P(5)
            p17 = P(17)
            palm_width = float(np.linalg.norm(p5 - p17)) + 1e-6

            # ===== 规则(1) 共线度：每根手指的三段向量夹角都必须 < 40°（明显伸直） =====
            # 每根手指：(MCP_idx, PIP_idx, DIP_idx, TIP_idx)
            fingers_def = [
                ("index",  5,  6,  7,  8),
                ("middle", 9, 10, 11, 12),
                ("ring",   13,14, 15, 16),
                ("pinky",  17,18, 19, 20),
            ]
            # 拇指单独定义（CMC=1 / MCP=2 / IP=3 / TIP=4）
            thumb_def = ("thumb", 1, 2, 3, 4)

            def _finger_bend_deg(defn) -> float:
                _, i_mcp, i_pip, i_dip, i_tip = defn
                a = P(i_pip) - P(i_mcp)   # MCP→PIP 段
                b = P(i_tip) - P(i_pip)   # PIP→TIP 段
                return DiagonalStretchManager._angle_deg(a, b)

            for fd in fingers_def:
                deg = _finger_bend_deg(fd)
                if not math.isfinite(deg) or deg >= 30.0:
                    return False   # 任意一根手指弯得太厉害（≥30°） → 不是张开
            tdeg = _finger_bend_deg(thumb_def)
            if not math.isfinite(tdeg) or tdeg >= 35.0:  # 拇指略放宽到 35°（拇指本身更易斜）
                return False

            # ===== 规则(2) 散开度：5 指尖必须足够分散（外接矩形≥0.35掌宽²）→ 排除并拢/重合（用户反馈的手指重合状态） =====
            tips = np.stack([P(4), P(8), P(12), P(16), P(20)], axis=0)  # (5,2)
            spread_x = float(tips[:, 0].max() - tips[:, 0].min())
            spread_y = float(tips[:, 1].max() - tips[:, 1].min())
            spread_area_norm = (spread_x * spread_y) / (palm_width * palm_width)
            if spread_area_norm < 0.35:
                return False   # 指尖太集中（并拢/手指重合）→ 绝对不算五指张开

            # ===== 规则(3) 外于掌：5 指尖都必须「显著位于掌外（wrist 之外，比 MCP 更远）」 =====
            mcp_refs = [1, 5, 9, 13, 17]   # 对应 5 指的掌指关节（拇指用 CMC=1，四指用 MCP）
            tip_refs = [4, 8, 12, 16, 20]
            for i_mcp, i_tip in zip(mcp_refs, tip_refs):
                d_tip  = float(np.linalg.norm(P(i_tip) - wrist))
                d_mcp  = float(np.linalg.norm(P(i_mcp) - wrist)) + 1e-6
                ratio = d_tip / d_mcp
                if ratio < 1.15:
                    return False   # 指尖不比掌指关节远（指尖弯回手心/贴手掌）→ 不是张开

            return True
        except Exception:
            return False

    # ------------------------------------------------------------
    # ✨✨ 关键静态辅助：把「命中点取得的 HWND」强制升级为「应用顶层主窗口」
    #   解决 WPS/Word/Chrome 等多标签页应用：
    #     掌心落在 标签栏/工具栏/菜单栏/地址栏 → WindowFromPoint 返回「子控件 HWND（标签页本身）」
    #     → 如果直接 SetWindowPos 移动子控件 → 完全没用（用户要的是整个应用窗口移动）
    #   处理步骤：
    #     ① 用 GetAncestor(hwnd_hit, GA_ROOTOWNER) → 跳到"顶层所有者窗口"=应用主窗口（MSDN明确：跳过所有子控件/子对话框）
    #     ② 若结果为空 → 兜底用 GetAncestor(GA_ROOT) 爬到最顶层父窗口
    #     ③ 最终强校验：必须「不带 WS_CHILD（=不是子控件）」并且「带 WS_CAPTION 或 WS_POPUP（=有标题栏的正常窗口）」
    #     ④ 不通过强校验 → 返回 0 交给上层兜底（用前台窗口），绝不返回子控件给移动/缩放用
    # ------------------------------------------------------------
    @staticmethod
    def _promote_hit_hwnd_to_top_level(hwnd_hit: int) -> int:
        try:
            hwnd_hit = int(hwnd_hit or 0)
            if hwnd_hit == 0:
                return 0
            # ① 主路：GA_ROOTOWNER → 顶层所有者（=应用主窗口，最标准）
            hwnd_top = int(GetAncestor(wintypes.HWND(hwnd_hit), wintypes.UINT(GA_ROOTOWNER)) or 0)
            if hwnd_top == 0:
                # ② 兜底：GA_ROOT → 顶层父
                hwnd_top = int(GetAncestor(wintypes.HWND(hwnd_hit), wintypes.UINT(GA_ROOT)) or 0)
            if hwnd_top == 0:
                # 再兜底：原始命中点自己也可能是顶层（比如单窗口小应用）
                hwnd_top = hwnd_hit

            # ③ 强校验：是否为"可移动顶层窗口"（非子控件 + 有标题栏/或弹窗样式）
            try:
                style = int(GetWindowLongPtrW(wintypes.HWND(hwnd_top), wintypes.INT(GWL_STYLE)) or 0)
            except Exception:
                style = 0
            # 只要带 WS_CHILD → 绝对不是顶层主窗口，返回 0（避免移动子控件）
            if (style & WS_CHILD) != 0:
                return 0
            # 必须带 WS_CAPTION（标题栏窗口）或 WS_POPUP（弹窗）→ 才算可移动的顶层窗口
            has_caption_or_popup = ((style & WS_CAPTION) != 0) or ((style & WS_POPUP) != 0)
            if not has_caption_or_popup:
                # 极少数顶层窗口可能是无标题（全屏应用/全屏播放）→ 兜底允许只要不带 WS_CHILD 也算（避免误判合法窗口）
                if hwnd_top == hwnd_hit:
                    return hwnd_top   # 是命中窗口自己且非 WS_CHILD → 允许
                return 0

            # 必须是可交互的可见窗口（再拿一次窗口矩形校验）
            rc = RECT()
            ok = bool(GetWindowRect(wintypes.HWND(hwnd_top), ctypes.byref(rc)))
            if not ok or (rc.right - rc.left) < 80 or (rc.bottom - rc.top) < 60:
                # 极小的不可见/残留 HWND → 直接丢弃
                return 0

            return int(hwnd_top)
        except Exception:
            # 任意异常 → 安全返回 0（交给上层用前台窗口兜底）
            return 0

    # ------------------------------------------------------------
    # 静态辅助：单手捏合判定 → 🔥 用户新方案：除「五指张开(OPEN)」外，其他所有姿态都视作捏合手势
    #   返回: (is_pinching, pinch_r)
    #   pinch_r 保留原 TIP4-TIP8 距离比值（仅用于 UI 显示，不再作为是否捏合的判定依据）
    #   带滞回：上一帧已捏合 = 用 hold 阈值判断"是否仍捏合"（原逻辑保留，但判定主逻辑已改为 OPEN/非OPEN）
    # ------------------------------------------------------------
    @staticmethod
    def _hand_pinch(lms, enter_thresh: float, hold_thresh: float, was_pinching: bool, force_open: bool | None = None) -> Tuple[bool, float]:
        # ① 主判定：用户新方案 → 只要不是「张开手掌(OPEN)」= 其他都视为捏合手势
        #   ✨ 如果传入了 force_open（外部已稳定化后的 OPEN 判定）→ 直接用它，保证 pinch 和 open 的结果 100% 一致，不会出现"open稳定/pinch抖动"的错位
        if force_open is None:
            is_open = DiagonalStretchManager._is_open_palm(lms)
        else:
            is_open = bool(force_open)
        pinch_by_rule = bool(not is_open)
        # ② 仍计算原 TIP4-TIP8 距离比值（仅用于 UI 显示，便于用户观察）
        try:
            palm_width = float(np.linalg.norm(np.array([lms[5].x, lms[5].y], dtype=np.float64)
                                              - np.array([lms[17].x, lms[17].y], dtype=np.float64))) + 1e-6
            pinch_d = float(np.linalg.norm(np.array([lms[4].x, lms[4].y], dtype=np.float64)
                                           - np.array([lms[8].x, lms[8].y], dtype=np.float64)))
            r = pinch_d / palm_width
        except Exception:
            r = 999.0
        # ③ 滞回（与原逻辑对齐：已捏合时更难释放，防抖动）→ 但主规则仍是「非OPEN即捏合」，这里仅作安全兜底
        if was_pinching:
            # 进入"捏合保持"后，只有明确张开手掌(OPEN=True)才释放
            return pinch_by_rule, r
        else:
            # 未捏合状态：只要满足「非OPEN = 捏合规则」就进入捏合
            return pinch_by_rule, r


    # ------------------------------------------------------------
    # 静态辅助：单手"捏合锚点"像素坐标 = 拇指TIP(4) + 食指TIP(8) 的几何中点（归一化坐标乘 W/H）
    #   ✨ 需求：所有缩放/距离计算/连线/锚点 不再用掌心，统一用"拇食重合点"，跟用户实际捏合操作点严格同步
    #   返回 None 代表 lms 缺失（fallback 到传入的掌心兜底）
    # ------------------------------------------------------------
    @staticmethod
    def _pinch_anchor_px(lms, W: int, H: int) -> Optional[Tuple[float, float]]:
        if lms is None:
            return None
        try:
            t4x, t4y = float(lms[4].x), float(lms[4].y)
            t8x, t8y = float(lms[8].x), float(lms[8].y)
            cx_n = (t4x + t8x) * 0.5
            cy_n = (t4y + t8y) * 0.5
            return float(cx_n) * float(W), float(cy_n) * float(H)
        except Exception:
            return None

    # ------------------------------------------------------------
    # 主入口：每帧调用
    #   left_px_xy / right_px_xy: 画面（左=Left hand label / 右=Right hand label）两只手掌心像素坐标 (x, y)
    #                              ✨仅作 fallback：当 _pinch_anchor_px 拿不到捏合点时才用此值
    #   left_lms / right_lms: 对应每只手的 21 个 MediaPipe landmarks（用于捏合判定 + 捏合锚点计算）
    #   frame_size: (W, H) 画面像素尺寸
    #   ✨ 新增：left_palm_screen_xy / right_palm_screen_xy → 两只手掌心的「屏幕绝对坐标」(屏幕像素x,y)
    #        → MOVE模式（五指张开）时，用左手掌心屏幕坐标找到该点下的窗口(WindowFromPoint)，并按手掌位移差移动窗口
    #   缺一只手时传 None。
    # ------------------------------------------------------------
    def update(self,
               left_px_xy: Optional[Tuple[float, float]],
               right_px_xy: Optional[Tuple[float, float]],
               frame_size: Tuple[int, int],
               left_lms=None,
               right_lms=None,
               left_palm_screen_xy: Optional[Tuple[float, float]] = None,
               right_palm_screen_xy: Optional[Tuple[float, float]] = None,
               dt: float = 0.0,
               ) -> StretchInfo:
        W, H = int(frame_size[0]), int(frame_size[1])
        info = self.info
        info.state = self.state
        info.confirm_sec = self._confirm
        info.left_pinch_r = -1.0
        info.right_pinch_r = -1.0
        info.stretch_mode = self._stretch_mode
        info.move_state = self._move_state
        info.move_hwnd = self._move_hwnd
        info.move_confirm_sec = self._move_confirm

        # ✨ 需求2：所有缩放锚点 → 不再用掌心，统一用「拇食重合点」= TIP4(4)+TIP8(8) 中点
        #   先用 lms 算捏合锚点；拿不到（lms缺失/异常）才 fallback 传进来的掌心像素。
        pinch_left = self._pinch_anchor_px(left_lms, W, H)
        pinch_right = self._pinch_anchor_px(right_lms, W, H)
        left_px_xy  = pinch_left  if pinch_left  is not None else left_px_xy
        right_px_xy = pinch_right if pinch_right is not None else right_px_xy
        info.p1 = left_px_xy   # UI 连线点也同步用捏合锚点（用户手在哪，圆圈就画在哪，不漂移）；SINGLE MOVE 下 p2=None
        info.p2 = right_px_xy

        # ---------- 条件 0-①：两只手都就绪 → 进入"双手互斥"模式（SCALE缩放 / DUAL MOVE双手张开移动） ----------
        both_hands_ready = not (left_px_xy is None or right_px_xy is None or left_lms is None or right_lms is None)
        # ---------- 条件 0-②：只有一只手就绪（恰好一只）→ 判这只手是否OPEN → 走"SINGLE MOVE 单手张开移动" ----------
        single_ready = False
        single_lms = None
        single_px_xy: Optional[Tuple[float,float]] = None
        single_screen_xy: Optional[Tuple[float,float]] = None
        single_is_left = True
        if not both_hands_ready:
            # 左手单独存在
            if left_px_xy is not None and left_lms is not None and right_lms is None and right_px_xy is None:
                single_ready = True
                single_is_left = True
                single_lms = left_lms
                single_px_xy = left_px_xy
                single_screen_xy = left_palm_screen_xy
            # 右手单独存在
            elif right_px_xy is not None and right_lms is not None and left_lms is None and left_px_xy is None:
                single_ready = True
                single_is_left = False
                single_lms = right_lms
                single_px_xy = right_px_xy
                single_screen_xy = right_palm_screen_xy
            else:
                # 两只手都不存在 / 或 状态混杂（比如有lms没px）→ 都缺
                both_hands_ready = False
                single_ready = False

        # ---------- 都缺（双手没检测到 + 单手也没检测到） → ✨ ACTIVE 时先 LOST 容忍 8 帧，超了再真退 IDLE ----------
        if (not both_hands_ready) and (not single_ready):
            info.hands_mode = "DUAL"
            scale_keep = False
            move_keep  = False
            # ① SCALE_ACTIVE：先容忍（8 帧内回来就能无缝续上）
            if self.state == _S.SCALING:
                scale_keep = bool(self._try_tolerate_lost_scale(dt))
            # ② MOVE_ACTIVE：先容忍
            if self._move_state == _MS.MOVE_ACTIVE:
                move_keep = bool(self._try_tolerate_lost_move(dt))
            # 两边都在 ACTIVE+容忍中 → 不 reset，保持状态（UI/快照/锚点都保留，等手回来）
            if scale_keep or move_keep:
                info.ui_pinch_active = bool(self.state == _S.SCALING)   # 继续按原状态画 UI
                info.ui_move_active  = bool(self._move_state == _MS.MOVE_ACTIVE)
                info.need_frame_redraw = bool(self.state == _S.SCALING)
                info.move_need_frame_redraw = bool(self._move_state == _MS.MOVE_ACTIVE)
                info.state = str(self.state)
                info.move_state = str(self._move_state)
                info.mode = (_MODE.MOVE if (self._move_state == _MS.MOVE_ACTIVE) else _MODE.SCALE)
                # 🔴 容忍期内：不做任何实际窗口移动/缩放（等手重新出现再续上原来的偏移量）
                info.scale_delta = 0.0
                info.move_delta_xy = (0.0, 0.0)
                return info
            # 超容忍或不在 ACTIVE → 回 IDLE（原逻辑）
            info.ui_pinch_active = False
            info.ui_move_active  = False
            self._reset_to_idle()
            self._reset_move_to_idle()
            info.need_frame_redraw = False
            info.move_need_frame_redraw = False
            info.state = _S.IDLE
            info.move_state = _MS.IDLE
            info.mode = _MODE.SCALE
            return info

        # ============================================================
        # 🔀 分支 A：恰好一只手 + 这只手 OPEN → SINGLE MOVE（单手张开移动窗口）
        # ============================================================
        if single_ready and not both_hands_ready:
            s_open_raw = bool(DiagonalStretchManager._is_open_palm(single_lms))
            s_open = bool(self._stabilize_bool("S", s_open_raw, dt))   # ✨ OPEN/PINCH 稳定化（样本数+时长双条件）
            # ✅ 单手 OPEN → 进入 MOVE（SINGLE）模式；其他单手姿态（非OPEN=捏合/POINT/握拳）→ 都不触发 SINGLE MOVE（让单手控鼠点击正常工作）
            if not s_open:
                info.hands_mode = "SINGLE"
                # ✨ 关键：如果当前已是 MOVE_ACTIVE → 先容忍（等 8 帧再退），避免"单帧 OPEN/PINCH 抖一下就中断移动"
                if self._move_state == _MS.MOVE_ACTIVE:
                    move_keep = bool(self._try_tolerate_lost_move(dt))
                    if move_keep:
                        info.ui_move_active = True
                        info.ui_pinch_active = False
                        info.mode = _MODE.MOVE
                        info.move_need_frame_redraw = True
                        info.need_frame_redraw = False
                        info.move_state = str(self._move_state)
                        info.state = _S.IDLE
                        info.move_delta_xy = (0.0, 0.0)   # 容忍期内：不动窗口（等手势回来再继续原来的偏移）
                        info.move_confirm_sec = MOVE_CONFIRM_SEC
                        return info
                # 不在 MOVE_ACTIVE（或容忍超时）→ 回 IDLE（让正常单手控鼠）
                info.ui_pinch_active = False
                info.ui_move_active  = False
                # 这只手虽然单手存在但不是OPEN → SCALE/MOVE状态机都IDLE，让main.py走正常单手控鼠
                self._reset_to_idle()
                self._reset_move_to_idle()
                info.need_frame_redraw = False
                info.move_need_frame_redraw = False
                info.state = _S.IDLE
                info.move_state = _MS.IDLE
                info.mode = _MODE.SCALE
                return info
            # ===== SINGLE + OPEN = SINGLE MOVE 进入状态机 =====
            # ✨ 手势重新满足了（回来了/稳定住了）→ 清 move lost 计数（后面会正常更新窗口位置，无缝继续原来的偏移）
            self._move_lost_sec = 0.0
            info.mode = _MODE.MOVE
            info.hands_mode = "SINGLE"
            info.ui_move_active = True   # 🔴 单手+OPEN立刻显示MOVE UI（和双手MOVE同样的预判定体验）
            info.ui_pinch_active = False
            info.need_frame_redraw = False
            self._reset_to_idle()        # 互斥：SCALE 状态机必须 IDLE
            info.state = _S.IDLE
            # ✨ 单手指令的移动窗口：画面 UI 锚点位置 = 该手掌心（single_px_xy）
            info.p1 = single_px_xy
            info.p2 = None               # 另一只手不存在，UI不画
            return self._run_move_state_machine(
                info,
                left_px_xy, right_px_xy,
                left_palm_screen_xy, right_palm_screen_xy,
                hands_mode="SINGLE",
                single_lms=single_lms,
                single_px_xy=single_px_xy,
                single_screen_xy=single_screen_xy,
                single_is_left=single_is_left,
                dt=dt,
            )

        # ============================================================
        # 🔀 分支 B：双手都就绪（以下沿用原双手互斥逻辑）
        # ============================================================
        lx, ly = float(left_px_xy[0]),  float(left_px_xy[1])
        rx, ry = float(right_px_xy[0]), float(right_px_xy[1])

        # ---------- 双手每只手的 OPEN/捏合 判定 ----------
        # ✨✨ OPEN/PINCH 稳定化：连续 2 帧一致才切换输出（_stabilize_bool），从源头消除单帧抖动导致 ACTIVE 误退出
        left_open_raw  = bool(DiagonalStretchManager._is_open_palm(left_lms))
        right_open_raw = bool(DiagonalStretchManager._is_open_palm(right_lms))
        left_open  = bool(self._stabilize_bool("L", left_open_raw, dt))
        right_open = bool(self._stabilize_bool("R", right_open_raw, dt))

        # ---------- SCALE 模式：双手捏合判定（非OPEN=捏合） ✨ 传 force_open=稳定后的left_open/right_open → 保证 pinch/open 结果 100% 一致 ----------
        left_pinch,  left_r  = self._hand_pinch(left_lms,  self._pinch_enter, self._pinch_hold, self._left_pinch_prev,  force_open=left_open)
        right_pinch, right_r = self._hand_pinch(right_lms, self._pinch_enter, self._pinch_hold, self._right_pinch_prev, force_open=right_open)
        self._left_pinch_prev  = left_pinch
        self._right_pinch_prev = right_pinch
        info.left_pinch_r  = float(left_r)
        info.right_pinch_r = float(right_r)

        # ---------- 两个 UI 门控 + 模式互斥（先写 UI flag，保证画面立刻显示，再按互斥模式进入对应状态机） ----------
        #   MOVE=双手都 OPEN（五指张开）；SCALE=双手都非OPEN（都捏合）
        #   其他混杂情况（一只开一只闭）→ 两个 UI 都关，状态机都回 IDLE
        both_open  = bool(left_open  and right_open)
        both_pinched = bool(left_pinch and right_pinch)

        info.ui_move_active  = both_open
        info.ui_pinch_active = both_pinched and not both_open   # 避免两者同时 True（理论上互斥）

        if not (both_open or both_pinched):
            # 一只张开一只捏合 → 混杂姿态
            # ✨✨ ACTIVE 时先容忍（8 帧）：避免某只手 OPEN/PINCH 单帧抖了一下就立刻中断缩放/移动
            scale_keep = False
            move_keep  = False
            if self.state == _S.SCALING:
                scale_keep = bool(self._try_tolerate_lost_scale(dt))
            if self._move_state == _MS.MOVE_ACTIVE:
                move_keep = bool(self._try_tolerate_lost_move(dt))
            if scale_keep or move_keep:
                info.ui_pinch_active = bool(self.state == _S.SCALING)
                info.ui_move_active  = bool(self._move_state == _MS.MOVE_ACTIVE)
                info.need_frame_redraw = bool(self.state == _S.SCALING)
                info.move_need_frame_redraw = bool(self._move_state == _MS.MOVE_ACTIVE)
                info.state = str(self.state)
                info.move_state = str(self._move_state)
                info.mode = (_MODE.MOVE if (self._move_state == _MS.MOVE_ACTIVE) else _MODE.SCALE)
                info.scale_delta = 0.0
                info.move_delta_xy = (0.0, 0.0)
                return info
            # 容忍超时/不在 ACTIVE → 回 IDLE（原逻辑）
            self._reset_to_idle()
            self._reset_move_to_idle()
            info.need_frame_redraw = False
            info.move_need_frame_redraw = False
            info.mode = _MODE.SCALE
            info.state = _S.IDLE
            info.move_state = _MS.IDLE
            return info

        # ============================================================
        # 🔀 模式互斥分支一：双手五指张开(BOTH_OPEN) → 移动窗口 MOVE 状态机（DUAL 双手模式）
        # ============================================================
        if both_open:
            # ✨ 手势稳定匹配了 → 清 lost 计数（之前的容忍期作废，全新无缝继续）
            self._move_lost_sec = 0.0
            self._scale_lost_sec = 0.0
            info.mode = _MODE.MOVE
            info.hands_mode = "DUAL"
            info.need_frame_redraw = False  # SCALE 分支 UI 关
            self._reset_to_idle()           # 互斥：走 MOVE 时 SCALE 状态机必须 IDLE
            info.state = _S.IDLE
            return self._run_move_state_machine(
                info, left_px_xy, right_px_xy,
                left_palm_screen_xy, right_palm_screen_xy,
                hands_mode="DUAL",
                single_lms=None,
                single_px_xy=None,
                single_screen_xy=None,
                single_is_left=True,
                dt=dt,
            )

        # ============================================================
        # 🔀 模式互斥分支二：双手都非OPEN(都捏合) → 拉伸缩放 SCALE 状态机（原逻辑）
        # ============================================================
        # ✨ 手势稳定匹配了 → 清 scale lost 计数（之前容忍期作废，无缝续上原来的比例/锚点）
        self._scale_lost_sec = 0.0
        self._move_lost_sec = 0.0
        info.mode = _MODE.SCALE
        info.move_need_frame_redraw = False
        self._reset_move_to_idle()   # 互斥：走 SCALE 时 MOVE 状态机必须 IDLE
        info.move_state = _MS.IDLE

        # ---------- 计算两「捏合锚点」像素距离 d1（进入最小距离门槛 + SCALING 比例用） ----------
        dx = rx - lx
        dy = ry - ly
        d1 = math.sqrt(dx * dx + dy * dy)
        info.diag_dist_px = d1

        # ---------- 条件 2：两「捏合锚点」距离 ≥ min_diag_px（保留原门槛：两手凑太近不算"准备拉伸"） ----------
        if d1 < self.min_diag_px:
            # ✨ SCALING ACTIVE 时先容忍：手瞬时靠近（抖动）→ 先等 8 帧，不立刻退（避免一帧就近了就中断缩放）
            if self.state == _S.SCALING:
                keep = bool(self._try_tolerate_lost_scale(dt))
                if keep:
                    info.ui_pinch_active = True
                    info.need_frame_redraw = True
                    info.state = str(self.state)
                    info.scale_delta = 0.0
                    return info
            # 容忍超时或非 ACTIVE → 回 IDLE
            self._reset_to_idle()
            info.need_frame_redraw = False
            return info

        # ============ 到这里 = 双手都在 + 双手都捏合 + 距离足够（任意方向都算 → 已删对角约束）============
        # ---------- IDLE → 计数确认 ----------
        if self.state == _S.IDLE:
            self.state = _S.ARMED_CONFIRM
            # 双手出现即确认（0.10s 墙钟时长，物理手感不再随帧率漂移）
            self._confirm = 0.0
            info.state = _S.ARMED_CONFIRM
            info.confirm_sec = 0.0
            info.need_frame_redraw = True
            return info

        # ---------- ARMED_CONFIRM → 累计确认时长，达标则记录初始快照 → 切 SCALING ----------
        if self.state == _S.ARMED_CONFIRM:
            self._confirm += float(dt)
            info.confirm_sec = self._confirm
            if self._confirm >= self.min_hold_sec:
                # 拿当前前台窗口 HWND + 初始 rect
                hwnd = int(GetForegroundWindow() or 0)
                rc = RECT()
                ok = bool(GetWindowRect(wintypes.HWND(hwnd), ctypes.byref(rc)))
                if not ok or rc.right <= rc.left or rc.bottom <= rc.top:
                    # 窗口句柄非法 → 放弃这一次，回 IDLE 等下次
                    self._reset_to_idle()
                    return info
                # 快照
                self._hwnd = hwnd
                w0 = float(rc.right - rc.left)
                h0 = float(rc.bottom - rc.top)
                self._w0, self._h0 = w0, h0
                self._cx0 = float(rc.left + rc.right) * 0.5
                self._cy0 = float(rc.top  + rc.bottom) * 0.5
                self._d0 = float(d1) if d1 > 1e-3 else None
                # ✨ 新方案：两手捏合点连线 → 与水平线夹角θ（0°~90° 取绝对值）直接按区间判方向（完全替代原 dx0/dy0 比值）
                self._dx0 = abs(float(rx - lx))
                self._dy0 = abs(float(ry - ly))
                # θ = degrees( atan2(|垂直分量|, |水平分量|) ) → 0°=水平 90°=纯垂直
                # ✨ 用户要求：θ 区间再加一层 abs（确保万无一失，任何异常情况都不会出现负角度或>90°）
                theta_deg = abs(math.degrees(math.atan2(float(self._dy0), float(max(self._dx0, 1e-6)))))
                # 严格按用户定义的区间映射（边界含在两端，避免20°/70°卡在空挡）
                if theta_deg <= MODE_ANGLE_HORIZ_MAX_DEG:
                    # [0°, 20°] → 拉宽（水平 HORIZONTAL）
                    self._stretch_mode = _M.HORIZONTAL
                elif theta_deg >= MODE_ANGLE_VERT_MIN_DEG:
                    # [70°, 90°] → 拉高（垂直 VERTICAL）
                    self._stretch_mode = _M.VERTICAL
                else:
                    # (20°, 70°) → 对角等比缩放（DIAGONAL）
                    self._stretch_mode = _M.DIAGONAL
                info.stretch_mode = self._stretch_mode
                self._smooth_ratio = 1.0
                self._last_ratio = 1.0
                if self._d0 is not None:
                    self.state = _S.SCALING
                    info.state = _S.SCALING
                    info.hwnd = hwnd
                    info.ratio = 1.0
                    info.win_w = int(w0)
                    info.win_h = int(h0)
                    # ✨ 新增2：选中窗口瞬间 → 立即把鼠标光标跳到窗口几何中心（屏幕像素坐标 cx0/cy0 正好就是）
                    #    避免用户手势还没结束时，鼠标悬停在其他窗口上造成误点/误切前台
                    try:
                        SetCursorPos(wintypes.INT(int(round(self._cx0))),
                                     wintypes.INT(int(round(self._cy0))))
                    except Exception:
                        pass  # Win32 失败就跳过（安全兜底）
            info.need_frame_redraw = True
            return info

        # ---------- SCALING：每帧按「主导模式」改 size ----------
        # ✨ 模式说明：
        #   HORIZONTAL：两手水平排开 → 两手水平距离 dx 变化只改 w，h 保持 h0（拉宽/拉窄，中心不变）
        #   VERTICAL：  两手垂直排开 → 两手垂直距离 dy 变化只改 h，w 保持 w0（拉高/拉低，中心不变）
        #   DIAGONAL：  两手近对角 → 原等比缩放（宽高同比例，中心不变，用户原行为）
        if self.state == _S.SCALING and self._d0 is not None:
            mode = self._stretch_mode
            dx_now = abs(float(rx - lx))
            dy_now = abs(float(ry - ly))
            # 1) 按模式取对应维度的比例（彻底避免另一方向抖动干扰识别）
            if mode == _M.HORIZONTAL:
                # ✅ 水平模式：只看 dx / dx0，dy 再怎么抖都完全忽略（这是上下拉伸不准的根因）
                raw = dx_now / max(self._dx0, 1e-3)
            elif mode == _M.VERTICAL:
                # ✅ 垂直模式：只看 dy / dy0，dx 再怎么抖都完全忽略（这是上下拉伸不准的根因）
                raw = dy_now / max(self._dy0, 1e-3)
            else:
                # DIAGONAL：原欧氏距离（混 dx+dy 没问题，因为本来就是对角等比）
                raw = d1 / self._d0
            # 2) 灵敏度（非线性，sensitivity=2=更灵敏，0.5=更迟钝）
            raw_pow = math.pow(raw, self.sensitivity)
            # 3) EMA 平滑（消除 MediaPipe 单手尖峰）—— ✨ 墙钟化：alpha 由 dt 与时间常数换算
            if self._smooth_ratio is None:
                self._smooth_ratio = raw_pow
            else:
                # dt≤0（首帧/异常）回退按 30fps 一帧；dt>0.5s 视为重入，直接用原值
                _dt_eff = dt if 0.0 < dt <= 0.5 else 1.0 / 30.0
                _alpha = 1.0 - math.exp(-_dt_eff / self.smooth_tau_sec)
                self._smooth_ratio = (
                    _alpha * raw_pow
                    + (1.0 - _alpha) * self._smooth_ratio
                )
            # 4) 单帧限幅：防止一帧 d 突然跳 2 倍 → 窗口直接拉爆
            #    ✨ 墙钟化：每秒最大变化量 × dt（dt≤0 回退 30fps 一帧）
            _max_step = self.max_step_per_sec * _dt_eff
            lo = self._last_ratio - _max_step
            hi = self._last_ratio + _max_step
            clamped = lo if self._smooth_ratio < lo else (hi if self._smooth_ratio > hi else self._smooth_ratio)
            # 5) 根据主导模式决定目标 w / h（关键分岔点）
            w0, h0 = self._w0, self._h0
            if mode == _M.HORIZONTAL:
                # -------- 水平模式：只改宽度，高度 = h0 固定不变 --------
                tw = w0 * clamped
                th = h0
                if tw < self.min_win_w:
                    tw = float(self.min_win_w)
                    clamped = tw / max(w0, 1e-6)
                if tw > self.max_win_w:
                    tw = float(self.max_win_w)
                    clamped = tw / max(w0, 1e-6)
            elif mode == _M.VERTICAL:
                # -------- 垂直模式：只改高度，宽度 = w0 固定不变 --------
                tw = w0
                th = h0 * clamped
                if th < self.min_win_h:
                    th = float(self.min_win_h)
                    clamped = th / max(h0, 1e-6)
                if th > self.max_win_h:
                    th = float(self.max_win_h)
                    clamped = th / max(h0, 1e-6)
            else:
                # -------- 对角模式：原等比缩放（宽高同比例） --------
                tw = w0 * clamped
                th = h0 * clamped
                if tw < self.min_win_w:
                    tw = float(self.min_win_w)
                    clamped = tw / max(w0, 1e-6)
                if th < self.min_win_h:
                    th = float(self.min_win_h)
                    clamped = th / max(h0, 1e-6)
                if tw > self.max_win_w:
                    tw = float(self.max_win_w)
                    clamped = tw / max(w0, 1e-6)
                if th > self.max_win_h:
                    th = float(self.max_win_h)
                    clamped = th / max(h0, 1e-6)
            # 6) 保持中心 cx0/cy0 不变 → left/top
            left = int(round(self._cx0 - tw * 0.5))
            top  = int(round(self._cy0 - th * 0.5))
            iw, ih = int(round(tw)), int(round(th))
            # 7) 真实 Win32 改窗口（忽略返回值，失败就保持不动）
            try:
                SetWindowPos(
                    wintypes.HWND(self._hwnd),
                    wintypes.HWND(HWND_TOP),
                    wintypes.INT(left),   wintypes.INT(top),
                    wintypes.INT(iw),     wintypes.INT(ih),
                    wintypes.UINT(SWP_FLAGS),
                )
            except Exception:
                pass  # Win32 异常就跳过，下一帧继续
            self._last_ratio = clamped
            info.state = _S.SCALING
            info.hwnd = self._hwnd
            info.ratio = round(float(clamped), 3)
            info.win_w = iw
            info.win_h = ih
            info.stretch_mode = mode
            info.need_frame_redraw = True
            return info

        # 兜底
        info.need_frame_redraw = True
        return info

    # ============================================================
    # ✨ MOVE 状态机（单手张开 / 双手张开 → 选中掌心下的窗口并移动，鼠标固定窗口中心）
    #   hands_mode = "DUAL"  ：双手都张开，原逻辑，主手=左手（画面Left=用户视角右手）
    #   hands_mode = "SINGLE"：恰好一只手张开，用存在的那只手做主手（single_* 参数生效）
    # ============================================================
    def _run_move_state_machine(self, info: StretchInfo,
                                left_px_xy, right_px_xy,
                                left_palm_screen_xy,
                                right_palm_screen_xy,
                                hands_mode: str = "DUAL",
                                single_lms=None,
                                single_px_xy=None,
                                single_screen_xy=None,
                                single_is_left: bool = True,
                                dt: float = 0.0,
                                ) -> StretchInfo:
        # ---------- 主手掌心屏幕坐标 & 画面像素坐标：根据 SINGLE/DUAL 选择 ----------
        if hands_mode == "SINGLE":
            main_screen = single_screen_xy
        else:
            # 双手模式：主手=左手（画面 Left label = 用户视角右手，代表掌心位置）
            main_screen = left_palm_screen_xy
            if main_screen is None:
                main_screen = right_palm_screen_xy
        # ---------- IDLE → 进入确认帧 ----------
        if self._move_state == _MS.IDLE:
            self._move_state = _MS.MOVE_CONFIRM
            self._move_confirm = 0.0
            info.move_state = _MS.MOVE_CONFIRM
            info.move_confirm_sec = 0.0
            info.move_need_frame_redraw = True
            return info

        # ---------- MOVE_CONFIRM → 累计确认时长，达标进入 MOVE_ACTIVE ----------
        if self._move_state == _MS.MOVE_CONFIRM:
            self._move_confirm += float(dt)
            info.move_confirm_sec = self._move_confirm
            if self._move_confirm >= MOVE_CONFIRM_SEC:   # ✨ 单手/双手出现=必然要动作，确认墙钟时长
                # ① 取主手掌心屏幕坐标 → 找该点下的窗口
                hwnd = 0
                hwnd_hit = 0
                if main_screen is not None:
                    try:
                        pt = wintypes.POINT()
                        pt.x = wintypes.LONG(int(round(float(main_screen[0]))))
                        pt.y = wintypes.LONG(int(round(float(main_screen[1]))))
                        hwnd_hit = int(WindowFromPoint(pt) or 0)
                        # ✨✨ 核心修复（WPS/Word/Chrome 标签页问题）：把命中点 HWND 强制升级为「应用顶层主窗口」
                        #    GA_ROOTOWNER → 跳过所有子控件(标签栏/工具栏/菜单栏) → 最终必然得到：带标题栏的整个应用主窗口
                        hwnd = DiagonalStretchManager._promote_hit_hwnd_to_top_level(hwnd_hit)
                    except Exception:
                        hwnd = 0
                        hwnd_hit = 0
                # 找不到（命中点下的HWND升级失败=不合法子窗口）就兜底=当前前台窗口，并且前台窗口也要强制升级一次（防万一）
                if hwnd == 0:
                    try:
                        fg = int(GetForegroundWindow() or 0)
                        fg_top = DiagonalStretchManager._promote_hit_hwnd_to_top_level(fg)   # 前台也强制升级为顶层（确保前台也不是子控件）
                        hwnd = int(fg_top or fg or 0)
                    except Exception:
                        hwnd = 0
                # ② 拿窗口矩形
                rc = RECT()
                ok = bool(GetWindowRect(wintypes.HWND(hwnd), ctypes.byref(rc)))
                if not ok or rc.right <= rc.left or rc.bottom <= rc.top:
                    self._reset_move_to_idle()
                    info.move_need_frame_redraw = True
                    return info
                # ③ 置为前台窗口
                try:
                    SetForegroundWindow_API(wintypes.HWND(hwnd))
                except Exception:
                    pass
                # ④ 快照位置 + 参考手屏幕坐标
                self._move_hwnd = hwnd
                self._move_left0 = float(rc.left)
                self._move_top0  = float(rc.top)
                self._move_w0    = float(rc.right - rc.left)
                self._move_h0    = float(rc.bottom - rc.top)
                if main_screen is not None:
                    self._move_ref_sx = float(main_screen[0])
                    self._move_ref_sy = float(main_screen[1])
                else:
                    # 兜底：用当前窗口左上角当虚拟参考点（不推荐，有主手就不会走这里）
                    self._move_ref_sx = self._move_left0
                    self._move_ref_sy = self._move_top0
                # ⑤ 鼠标立刻固定到窗口中心（避免跳动误触）
                cx0 = self._move_left0 + self._move_w0 * 0.5
                cy0 = self._move_top0 + self._move_h0 * 0.5
                try:
                    SetCursorPos(wintypes.INT(int(round(cx0))),
                                 wintypes.INT(int(round(cy0))))
                except Exception:
                    pass
                # ⑥ 切换 MOVE_ACTIVE
                self._move_state = _MS.MOVE_ACTIVE
                info.move_state = _MS.MOVE_ACTIVE
                info.move_hwnd = hwnd
                info.move_window_rect = (int(rc.left), int(rc.top), int(rc.right), int(rc.bottom))
                info.move_delta_xy = (0.0, 0.0)
            info.move_need_frame_redraw = True
            return info

        # ---------- MOVE_ACTIVE：每帧按「主手位移差」移动窗口（只改 x/y，不改 w/h） ----------
        if self._move_state == _MS.MOVE_ACTIVE and self._move_hwnd != 0:
            # ① 计算当前主手屏幕坐标相对进入瞬间的偏移量（Δx/Δy）
            cur_sx, cur_sy = self._move_ref_sx, self._move_ref_sy
            if main_screen is not None:
                cur_sx = float(main_screen[0])
                cur_sy = float(main_screen[1])
            dx_screen = cur_sx - self._move_ref_sx
            dy_screen = cur_sy - self._move_ref_sy
            # ② 窗口新位置 = 进入时位置 + 偏移（只移动，不缩放）
            new_left = self._move_left0 + dx_screen
            new_top  = self._move_top0  + dy_screen
            i_left = int(round(new_left))
            i_top  = int(round(new_top))
            # ③ 实际移动窗口（SetWindowPos + SWP_NOSIZE → w/h 传 0 也没关系，标志会忽略）
            try:
                SetWindowPos(
                    wintypes.HWND(self._move_hwnd),
                    wintypes.HWND(HWND_TOP),
                    wintypes.INT(i_left),
                    wintypes.INT(i_top),
                    wintypes.INT(0),   # NOSIZE 忽略
                    wintypes.INT(0),
                    wintypes.UINT(SWP_MOVE_FLAGS),
                )
            except Exception:
                pass   # 异常就跳过
            # ④ 拿窗口最新矩形（给UI显示用 + 计算新中心）
            rc = RECT()
            ok = bool(GetWindowRect(wintypes.HWND(self._move_hwnd), ctypes.byref(rc)))
            if ok:
                cx = float(rc.left + rc.right) * 0.5
                cy = float(rc.top + rc.bottom) * 0.5
                info.move_window_rect = (int(rc.left), int(rc.top), int(rc.right), int(rc.bottom))
                # ⑤ 🔴 每帧都重新 SetCursorPos 到当前窗口中心 → 鼠标永久固定在窗口中心=不会跳动误触！
                try:
                    SetCursorPos(wintypes.INT(int(round(cx))),
                                 wintypes.INT(int(round(cy))))
                except Exception:
                    pass
            info.move_delta_xy = (float(dx_screen), float(dy_screen))
            info.move_hwnd = self._move_hwnd
            info.move_state = _MS.MOVE_ACTIVE
            info.move_need_frame_redraw = True
            return info

        info.move_need_frame_redraw = True
        return info

    # ---------- 工具：回到 IDLE（MOVE 状态机专属重置） ----------
    def _reset_move_to_idle(self):
        if self._move_state != _MS.IDLE:
            self._move_state = _MS.IDLE
        self._move_confirm = 0.0
        self._move_hwnd = 0
        self._move_ref_sx = 0.0
        self._move_ref_sy = 0.0
        self._move_left0 = 0.0
        self._move_top0 = 0.0
        self._move_w0 = 0.0
        self._move_h0 = 0.0
        self._move_lost_sec = 0.0   # ✨ 重置时清 lost 计时

    # ---------- 工具：回到 IDLE（清所有快照 + 清空捏合滞回状态 + 清空拉伸模式） ----------
    def _reset_to_idle(self):
        if self.state != _S.IDLE:
            self.state = _S.IDLE
        # ✨ 清空捏合滞回记忆：下次进重新按 enter 阈值严格判定
        self._left_pinch_prev  = False
        self._right_pinch_prev = False
        # ✨ 清空拉伸模式分量锚点：下次重新按进入瞬间判定
        self._dx0 = 0.0
        self._dy0 = 0.0
        self._stretch_mode = _M.DIAGONAL
        self._confirm = 0.0
        self._d0 = None
        self._hwnd = 0
        self._cx0 = self._cy0 = 0.0
        self._w0 = self._h0 = 0.0
        self._smooth_ratio = None
        self._last_ratio = 1.0
        self._scale_lost_sec = 0.0   # ✨ 重置时清 lost 计时
        self._open_stable.clear()      # ✨ 重置时清 OPEN 稳定缓存（下一轮从头开始攒）
