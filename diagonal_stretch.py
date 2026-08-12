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
    MOVE_CONFIRM  = "MOVE_CONFIRM"   # 双手五指张开 确认帧
    MOVE_ACTIVE   = "MOVE_ACTIVE"    # 已选中掌心下的窗口 正在移动


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
    confirm_frames: int = 0
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
    move_confirm_frames: int = 0      # 移动确认帧进度
    move_window_rect: Optional[Tuple[int,int,int,int]] = None  # (left, top, right, bottom) 当前移动窗口的屏幕矩形
    move_delta_xy: Tuple[float, float] = (0.0, 0.0)           # 本次移动累计偏移 (Δx, Δy) 像素
    move_need_frame_redraw: bool = False  # UI：移动模式是否需要画提示卡片


# ============================================================
# 主类：双手对角拉伸 → 窗口缩放状态机
# ============================================================
class DiagonalStretchManager:
    def __init__(self, *,
                 min_hold_frames: int = 8,
                 min_diag_px: float = 120.0,  # ✨ 原 220 太严，降到 120（两手靠得稍近也能进入拉伸确认帧）
                 # ✨ 新增：双手捏合判定阈值（与 gesture_recognizer._pinch_distance 同尺度，‖TIP4-TIP8‖/palm_width）
                 #   注意：当前捏合判定已改为「非OPEN即捏合」，这里的 pinch_enter/hold 仅用于滞回（enter<=hold）兼容，实际阈值不影响判定结果
                 pinch_enter: float = 0.40,
                 pinch_hold:  float = 0.50,
                 # 原平滑/窗口/灵敏度参数保持不变
                 smooth_alpha: float = 0.75,
                 max_step_per_frame: float = 0.18,
                 min_win_w: int = 240, min_win_h: int = 160,
                 max_win_w: int = 5120, max_win_h: int = 2880,
                 sensitivity: float = 1.0,
                 ):
        self.state = _S.IDLE
        self.info = StretchInfo()

        # 参数
        self._default_min_hold_frames = int(max(1, min_hold_frames))  # 构造时的原始默认值（IDLE 时保持）
        self.min_hold_frames          = self._default_min_hold_frames  # 运行时当前生效：双手出现→改为3，其余时间=默认
        self.min_diag_px        = float(max(60.0, min_diag_px))
        # 捏合阈值滞回约束：enter 必须 <= hold（经验 630282 阈值体系 enter < exit/hold 才稳定）
        if pinch_enter > pinch_hold:
            pinch_enter, pinch_hold = pinch_hold, pinch_enter
        self._pinch_enter = float(max(0.0, min(1.0, pinch_enter)))
        self._pinch_hold  = float(max(0.0, min(1.0, pinch_hold)))
        self.smooth_alpha       = 0.2 if smooth_alpha < 0.2 else (1.0 if smooth_alpha > 1.0 else float(smooth_alpha))
        self.max_step           = 0.01 if max_step_per_frame < 0.01 else (1.0 if max_step_per_frame > 1.0 else float(max_step_per_frame))
        self.min_win_w, self.min_win_h = int(min_win_w), int(min_win_h)
        self.max_win_w, self.max_win_h = int(max_win_w), int(max_win_h)
        self.sensitivity        = 0.2 if sensitivity < 0.2 else (3.0 if sensitivity > 3.0 else float(sensitivity))

        # -------- ✨ 新增：移动窗口(MOVE)状态机（双手五指张开触发） --------
        self._move_state: str = _MS.IDLE
        self._move_confirm: int = 0
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
        self._confirm: int = 0
        self._left_pinch_prev: bool = False   # 上一帧左手是否处于捏合（用于滞回 hold 判定）
        self._right_pinch_prev: bool = False  # 上一帧右手是否处于捏合

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
    # 静态辅助：单手捏合判定 → 🔥 用户新方案：除「五指张开(OPEN)」外，其他所有姿态都视作捏合手势
    #   返回: (is_pinching, pinch_r)
    #   pinch_r 保留原 TIP4-TIP8 距离比值（仅用于 UI 显示，不再作为是否捏合的判定依据）
    #   带滞回：上一帧已捏合 = 用 hold 阈值判断"是否仍捏合"（原逻辑保留，但判定主逻辑已改为 OPEN/非OPEN）
    # ------------------------------------------------------------
    @staticmethod
    def _hand_pinch(lms, enter_thresh: float, hold_thresh: float, was_pinching: bool) -> Tuple[bool, float]:
        # ① 主判定：用户新方案 → 只要不是「张开手掌(OPEN)」= 其他都视为捏合手势
        is_open = DiagonalStretchManager._is_open_palm(lms)
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
               right_palm_screen_xy: Optional[Tuple[float, float]] = None
               ) -> StretchInfo:
        W, H = int(frame_size[0]), int(frame_size[1])
        info = self.info
        info.state = self.state
        info.confirm_frames = self._confirm
        info.left_pinch_r = -1.0
        info.right_pinch_r = -1.0
        info.stretch_mode = self._stretch_mode
        info.move_state = self._move_state
        info.move_hwnd = self._move_hwnd
        info.move_confirm_frames = self._move_confirm

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

        # ---------- 都缺 → 直接回 IDLE（SCALE 和 MOVE 两个状态机都重置） ----------
        if (not both_hands_ready) and (not single_ready):
            info.ui_pinch_active = False   # 🔴 缺一只手时 UI 明确关连线
            info.ui_move_active  = False
            info.hands_mode = "DUAL"
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
            s_open = bool(DiagonalStretchManager._is_open_palm(single_lms))
            # ✅ 单手 OPEN → 进入 MOVE（SINGLE）模式；其他单手姿态（非OPEN=捏合/POINT/握拳）→ 都不触发 SINGLE MOVE（让单手控鼠点击正常工作）
            if not s_open:
                info.ui_pinch_active = False
                info.ui_move_active  = False
                info.hands_mode = "SINGLE"
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
            )

        # ============================================================
        # 🔀 分支 B：双手都就绪（以下沿用原双手互斥逻辑）
        # ============================================================
        lx, ly = float(left_px_xy[0]),  float(left_px_xy[1])
        rx, ry = float(right_px_xy[0]), float(right_px_xy[1])

        # ---------- 双手每只手的 OPEN/捏合 判定 ----------
        left_open  = bool(DiagonalStretchManager._is_open_palm(left_lms))
        right_open = bool(DiagonalStretchManager._is_open_palm(right_lms))

        # ---------- SCALE 模式：双手捏合判定（非OPEN=捏合） ----------
        left_pinch,  left_r  = self._hand_pinch(left_lms,  self._pinch_enter, self._pinch_hold, self._left_pinch_prev)
        right_pinch, right_r = self._hand_pinch(right_lms, self._pinch_enter, self._pinch_hold, self._right_pinch_prev)
        self._left_pinch_prev  = left_pinch
        self._right_pinch_prev = right_pinch
        info.left_pinch_r  = float(left_r)
        info.right_pinch_r = float(right_r)

        # ---------- � 两个 UI 门控 + 模式互斥（先写 UI flag，保证画面立刻显示，再按互斥模式进入对应状态机） ----------
        #   MOVE=双手都 OPEN（五指张开）；SCALE=双手都非OPEN（都捏合）
        #   其他混杂情况（一只开一只闭）→ 两个 UI 都关，状态机都回 IDLE
        both_open  = bool(left_open  and right_open)
        both_pinched = bool(left_pinch and right_pinch)

        info.ui_move_active  = both_open
        info.ui_pinch_active = both_pinched and not both_open   # 避免两者同时 True（理论上互斥）

        if not (both_open or both_pinched):
            # 一只张开一只捏合 → 混杂姿态，两个状态机都回 IDLE
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
            )

        # ============================================================
        # 🔀 模式互斥分支二：双手都非OPEN(都捏合) → 拉伸缩放 SCALE 状态机（原逻辑）
        # ============================================================
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
            self._reset_to_idle()
            info.need_frame_redraw = False
            return info

        # ============ 到这里 = 双手都在 + 双手都捏合 + 距离足够（任意方向都算 → 已删对角约束）============
        # ---------- IDLE → 计数确认 ----------
        if self.state == _S.IDLE:
            self.state = _S.ARMED_CONFIRM
            # ✨ 用户要求：双手出现即意味不是放大缩小就是移动窗口，因此确认帧从默认降到 3 帧
            self.min_hold_frames = 3
            self._confirm = 1
            info.state = _S.ARMED_CONFIRM
            info.confirm_frames = 1
            info.need_frame_redraw = True
            return info

        # ---------- ARMED_CONFIRM → 累计确认帧，达标则记录初始快照 → 切 SCALING ----------
        if self.state == _S.ARMED_CONFIRM:
            self._confirm += 1
            info.confirm_frames = self._confirm
            if self._confirm >= self.min_hold_frames:
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
            # 3) EMA 平滑（消除 MediaPipe 单手尖峰）
            if self._smooth_ratio is None:
                self._smooth_ratio = raw_pow
            else:
                self._smooth_ratio = (
                    self.smooth_alpha * raw_pow
                    + (1.0 - self.smooth_alpha) * self._smooth_ratio
                )
            # 4) 单帧限幅：防止一帧 d 突然跳 2 倍 → 窗口直接拉爆
            lo = self._last_ratio - self.max_step
            hi = self._last_ratio + self.max_step
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
            self._move_confirm = 1
            info.move_state = _MS.MOVE_CONFIRM
            info.move_confirm_frames = 1
            info.move_need_frame_redraw = True
            return info

        # ---------- MOVE_CONFIRM → 累计 3 帧，达标进入 MOVE_ACTIVE ----------
        if self._move_state == _MS.MOVE_CONFIRM:
            self._move_confirm += 1
            info.move_confirm_frames = self._move_confirm
            if self._move_confirm >= 3:   # ✨ 单手/双手出现=必然要动作，确认帧少
                # ① 取主手掌心屏幕坐标 → 找该点下的窗口
                hwnd = 0
                if main_screen is not None:
                    try:
                        pt = wintypes.POINT()
                        pt.x = wintypes.LONG(int(round(float(main_screen[0]))))
                        pt.y = wintypes.LONG(int(round(float(main_screen[1]))))
                        hwnd = int(WindowFromPoint(pt) or 0)
                    except Exception:
                        hwnd = 0
                # 找不到就兜底=当前前台窗口
                if hwnd == 0:
                    hwnd = int(GetForegroundWindow() or 0)
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
        self._move_confirm = 0
        self._move_hwnd = 0
        self._move_ref_sx = 0.0
        self._move_ref_sy = 0.0
        self._move_left0 = 0.0
        self._move_top0 = 0.0
        self._move_w0 = 0.0
        self._move_h0 = 0.0

    # ---------- 工具：回到 IDLE（清所有快照 + 恢复 min_hold_frames 默认值 + 清空捏合滞回状态 + 清空拉伸模式） ----------
    def _reset_to_idle(self):
        if self.state != _S.IDLE:
            self.state = _S.IDLE
        # ✨ 离开缩放/确认态时恢复默认的 min_hold_frames，保证 IDLE/单手场景下仍遵循构造参数
        self.min_hold_frames = self._default_min_hold_frames
        # ✨ 清空捏合滞回记忆：下次进重新按 enter 阈值严格判定
        self._left_pinch_prev  = False
        self._right_pinch_prev = False
        # ✨ 清空拉伸模式/分量锚点：下次重新按进入瞬间判定
        self._dx0 = 0.0
        self._dy0 = 0.0
        self._stretch_mode = _M.DIAGONAL
        self._confirm = 0
        self._d0 = None
        self._hwnd = 0
        self._cx0 = self._cy0 = 0.0
        self._w0 = self._h0 = 0.0
        self._smooth_ratio = None
        self._last_ratio = 1.0
