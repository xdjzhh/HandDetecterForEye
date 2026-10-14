# -*- coding: utf-8 -*-
"""
单手五指张开（OPEN）→ 选中掌心下的顶层窗口 → 拖移窗口位置
================================================================
2026-10-14：从已删除的 diagonal_stretch.py 中【仅恢复 SINGLE MOVE】分支。
  原模块的双手捏合缩放(SCALE)与双手张开移窗(DUAL MOVE)按用户要求不再恢复，
  本模块只保留：画面中恰好一只手 + 该手五指张开 → 移动窗口。

语义（与旧版单手分支完全一致）：
  1. 仅一只手在镜且 _is_open_palm 三规则判定为张开 → OPEN 稳定 2 样本/0.067s；
  2. 保持 MOVE_CONFIRM_SEC=0.10s → WindowFromPoint(掌心屏幕坐标) 命中窗口，
     经 GA_ROOTOWNER 升级为顶层主窗口（WPS/Chrome 标签页问题），失败兜底前台窗口；
  3. SetForegroundWindow + 快照窗口矩形/掌心参考点，光标锁到窗口中心；
  4. MOVE_ACTIVE：每帧按掌心屏幕坐标位移 SetWindowPos（只动 x/y 不改大小），
     光标持续重锁窗口中心，避免与掌心光标写入打架；
  5. 张开消失/手离开：先容忍 MOVE_LOST_MAX_SEC=0.13s（覆盖偶发漏检），超时回 IDLE。

依赖：Windows + ctypes.user32（纯标准库）。
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import math
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np


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
GetForegroundWindow.restype = wintypes.HWND

GetWindowRect = user32.GetWindowRect
GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(RECT)]
GetWindowRect.restype = wintypes.BOOL

SetWindowPos = user32.SetWindowPos
SetWindowPos.argtypes = [
    wintypes.HWND, wintypes.HWND,
    wintypes.INT, wintypes.INT,
    wintypes.INT, wintypes.INT,
    wintypes.UINT,
]
SetWindowPos.restype = wintypes.BOOL

# 移动窗口时把光标钉到窗口中心，避免跳动误触
SetCursorPos = user32.SetCursorPos
SetCursorPos.argtypes = [wintypes.INT, wintypes.INT]
SetCursorPos.restype = wintypes.BOOL

# SetWindowPos 标志：不改 Z 序 / 不抢激活焦点 / 不改大小
SWP_NOZORDER   = 0x0004
SWP_NOACTIVATE = 0x0010
SWP_NOSIZE     = 0x0001
SWP_MOVE_FLAGS = SWP_NOZORDER | SWP_NOACTIVATE | SWP_NOSIZE
HWND_TOP       = 0

# 按屏幕坐标找掌心覆盖的窗口
WindowFromPoint = user32.WindowFromPoint
WindowFromPoint.argtypes = [wintypes.POINT]
WindowFromPoint.restype = wintypes.HWND

SetForegroundWindow_API = user32.SetForegroundWindow
SetForegroundWindow_API.argtypes = [wintypes.HWND]
SetForegroundWindow_API.restype = wintypes.BOOL

# 命中点 HWND 向上追溯到顶层所有者窗口（跳过标签页/工具栏等子控件）
GetAncestor = user32.GetAncestor
GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
GetAncestor.restype = wintypes.HWND
GA_ROOT       = 2
GA_ROOTOWNER  = 3

# 旧版 Python 的 wintypes 可能缺 ULONG_PTR/LONG_PTR，手动补齐
if not hasattr(wintypes, 'ULONG_PTR') or not hasattr(wintypes, 'LONG_PTR'):
    import ctypes as _ct_local
    _ptr_size = _ct_local.sizeof(_ct_local.c_void_p)
    if _ptr_size == 8:
        if not hasattr(wintypes, 'ULONG_PTR'):
            wintypes.ULONG_PTR = _ct_local.c_ulonglong
        if not hasattr(wintypes, 'LONG_PTR'):
            wintypes.LONG_PTR = _ct_local.c_longlong
    else:
        if not hasattr(wintypes, 'ULONG_PTR'):
            wintypes.ULONG_PTR = _ct_local.c_ulong
        if not hasattr(wintypes, 'LONG_PTR'):
            wintypes.LONG_PTR = _ct_local.c_long
try:
    GetWindowLongPtrW = user32.GetWindowLongPtrW
except Exception:
    GetWindowLongPtrW = user32.GetWindowLongW  # 32 位兜底
GetWindowLongPtrW.argtypes = [wintypes.HWND, wintypes.INT]
GetWindowLongPtrW.restype = wintypes.ULONG_PTR
GWL_STYLE    = -16
WS_CHILD     = 0x40000000
WS_CAPTION   = 0x00C00000
WS_POPUP     = 0x80000000


# ============================================================
# 状态常量 / 时序参数
# ============================================================
class _MS:
    IDLE         = "IDLE"
    MOVE_CONFIRM = "MOVE_CONFIRM"   # 单手张开 确认中
    MOVE_ACTIVE  = "MOVE_ACTIVE"    # 已选中窗口 正在移动


MOVE_CONFIRM_SEC = 0.10    # 选中确认墙钟时长
MOVE_LOST_MAX_SEC = 0.13   # ACTIVE 下条件丢失容忍时长（≈4 帧 @30fps）
OPEN_STABLE_N   = 2        # OPEN 判定翻转所需连续样本
OPEN_STABLE_SEC = 0.067    # 及所需墙钟时长（双条件防抖）


@dataclass
class MoveInfo:
    move_state: str = _MS.IDLE
    move_hwnd: int = 0
    confirm_sec: float = 0.0
    window_rect: Optional[Tuple[int, int, int, int]] = None
    delta_xy: Tuple[float, float] = (0.0, 0.0)
    ui_active: bool = False       # 单手张开（已稳定）→ 立即给 UI 预反馈
    anchor_px: Optional[Tuple[float, float]] = None  # 画面中掌心像素（画圈用）
    need_redraw: bool = False


class WindowMover:
    """单手五指张开移窗状态机（无状态时每帧幂等回 IDLE，不影响正常控鼠）。"""

    def __init__(self):
        self.info = MoveInfo()
        self._move_state: str = _MS.IDLE
        self._move_confirm: float = 0.0
        self._move_hwnd: int = 0
        self._move_ref_sx: float = 0.0
        self._move_ref_sy: float = 0.0
        self._move_left0: float = 0.0
        self._move_top0: float = 0.0
        self._move_w0: float = 0.0
        self._move_h0: float = 0.0
        self._lost_sec: float = 0.0
        # OPEN 判定稳定缓存（单标签，单手场景只有一个 "S"）
        self._open_stable: dict = {}

    # ------------------------------------------------------------
    # 两向量夹角（度）；无效向量返回 inf
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
    # 单手五指张开判定 —— 三规则 AND（任一不过即非张开）：
    #  (1) 共线度：四指 MCP→PIP 与 PIP→TIP 夹角 <30°，拇指 <35°（手指明显伸直）
    #  (2) 散开度：五指尖外接矩形面积/掌宽² ≥ 0.35（排除并拢/重合）
    #  (3) 外于掌：每指尖到 wrist 距离 > 对应掌指关节到 wrist 的 1.15 倍
    # ------------------------------------------------------------
    @staticmethod
    def is_open_palm(lms) -> bool:
        try:
            def P(i):
                return np.array([lms[i].x, lms[i].y], dtype=np.float64)

            p5 = P(5)
            p17 = P(17)
            palm_width = float(np.linalg.norm(p5 - p17)) + 1e-6

            fingers_def = [
                (5, 6, 8),    # 食指 MCP/PIP/TIP
                (9, 10, 12),  # 中指
                (13, 14, 16), # 无名
                (17, 18, 20), # 小指
            ]

            def _bend_deg(i_mcp, i_pip, i_tip) -> float:
                a = P(i_pip) - P(i_mcp)
                b = P(i_tip) - P(i_pip)
                return WindowMover._angle_deg(a, b)

            for i_mcp, i_pip, i_tip in fingers_def:
                deg = _bend_deg(i_mcp, i_pip, i_tip)
                if not math.isfinite(deg) or deg >= 30.0:
                    return False
            tdeg = _bend_deg(1, 2, 4)
            if not math.isfinite(tdeg) or tdeg >= 35.0:
                return False

            tips = np.stack([P(4), P(8), P(12), P(16), P(20)], axis=0)
            spread_x = float(tips[:, 0].max() - tips[:, 0].min())
            spread_y = float(tips[:, 1].max() - tips[:, 1].min())
            if (spread_x * spread_y) / (palm_width * palm_width) < 0.35:
                return False

            wrist = P(0)
            for i_mcp, i_tip in zip((1, 5, 9, 13, 17), (4, 8, 12, 16, 20)):
                d_tip = float(np.linalg.norm(P(i_tip) - wrist))
                d_mcp = float(np.linalg.norm(P(i_mcp) - wrist)) + 1e-6
                if d_tip / d_mcp < 1.15:
                    return False

            return True
        except Exception:
            return False

    # ------------------------------------------------------------
    # 命中点 HWND → 顶层可移动主窗口（GA_ROOTOWNER + WS_CHILD/CAPTION 强校验）
    # ------------------------------------------------------------
    @staticmethod
    def _promote_hit_hwnd(hwnd_hit: int) -> int:
        try:
            hwnd_hit = int(hwnd_hit or 0)
            if hwnd_hit == 0:
                return 0
            hwnd_top = int(GetAncestor(wintypes.HWND(hwnd_hit), wintypes.UINT(GA_ROOTOWNER)) or 0)
            if hwnd_top == 0:
                hwnd_top = int(GetAncestor(wintypes.HWND(hwnd_hit), wintypes.UINT(GA_ROOT)) or 0)
            if hwnd_top == 0:
                hwnd_top = hwnd_hit
            try:
                style = int(GetWindowLongPtrW(wintypes.HWND(hwnd_top), wintypes.INT(GWL_STYLE)) or 0)
            except Exception:
                style = 0
            if (style & WS_CHILD) != 0:
                return 0
            if ((style & WS_CAPTION) == 0 and (style & WS_POPUP) == 0):
                if hwnd_top == hwnd_hit:
                    return hwnd_top  # 无标题顶层窗口（全屏应用）兜底允许
                return 0
            rc = RECT()
            ok = bool(GetWindowRect(wintypes.HWND(hwnd_top), ctypes.byref(rc)))
            if not ok or (rc.right - rc.left) < 80 or (rc.bottom - rc.top) < 60:
                return 0
            return int(hwnd_top)
        except Exception:
            return 0

    # ------------------------------------------------------------
    # 布尔判定稳定化：连续 OPEN_STABLE_N 个一致样本且累计 OPEN_STABLE_SEC 才翻转
    # ------------------------------------------------------------
    def _stabilize(self, raw: bool, dt: float) -> bool:
        try:
            # 注意（与旧版一致）：非张开帧 _reset 会清空缓存，故首个 raw 样本即 cur——
            # OPEN_STABLE_N 防抖只用于压制“已张开后的单帧抖动”；首次张开立即进入确认态，
            # 真正的误触门槛是 MOVE_CONFIRM_SEC=0.10s 的保持计时。
            cur, pend_v, pend_n, pend_sec = self._open_stable.get(
                "S", (bool(raw), bool(raw), 0, 0.0))
            if bool(raw) == bool(cur):
                self._open_stable["S"] = (bool(cur), bool(cur), 0, 0.0)
                return bool(cur)
            if int(pend_n) > 0 and bool(pend_v) == bool(raw):
                pend_n, pend_sec = int(pend_n) + 1, float(pend_sec) + float(dt)
            else:
                pend_n, pend_sec = 1, float(dt)
            if pend_n >= OPEN_STABLE_N and pend_sec >= OPEN_STABLE_SEC:
                self._open_stable["S"] = (bool(raw), bool(raw), 0, 0.0)
                return bool(raw)
            self._open_stable["S"] = (bool(cur), bool(raw), pend_n, pend_sec)
            return bool(cur)
        except Exception:
            return bool(raw)

    def _try_tolerate_lost(self, dt: float) -> bool:
        """ACTIVE 下条件丢失：容忍窗内返回 True 保持 ACTIVE。"""
        self._lost_sec += float(dt)
        return self._lost_sec < MOVE_LOST_MAX_SEC

    def _reset(self):
        self._move_state = _MS.IDLE
        self._move_confirm = 0.0
        self._move_hwnd = 0
        self._move_ref_sx = self._move_ref_sy = 0.0
        self._move_left0 = self._move_top0 = 0.0
        self._move_w0 = self._move_h0 = 0.0
        self._lost_sec = 0.0
        # 重置时清 OPEN 稳定缓存：下一轮从头开始攒（与旧版 _reset_move_to_idle 一致）
        self._open_stable.clear()

    # ------------------------------------------------------------
    # 主入口（每帧）
    #   n_hands    : 本帧画面中的手数量（0/1/2+）
    #   lms        : 唯一一只手的 21 个 2D landmarks（n_hands==1 时有效）
    #   px_xy      : 掌心锚点画面像素 (x, y)（UI 画圈用）
    #   screen_xy  : 掌心锚点屏幕绝对像素 (x, y)（命中窗口 + 位移用）
    #   dt         : 距上一帧秒数
    # ------------------------------------------------------------
    def update(self, n_hands: int, lms, px_xy: Optional[Tuple[float, float]],
               screen_xy: Optional[Tuple[float, float]], dt: float = 0.0) -> MoveInfo:
        info = self.info
        info.move_state = self._move_state
        info.move_hwnd = self._move_hwnd
        info.confirm_sec = self._move_confirm
        info.delta_xy = (0.0, 0.0)
        info.ui_active = False
        info.anchor_px = px_xy
        info.need_redraw = False

        def _hold_during_lost() -> bool:
            """ACTIVE 下条件丢失的统一容忍分支。"""
            if self._move_state == _MS.MOVE_ACTIVE and self._try_tolerate_lost(dt):
                info.ui_active = True
                info.move_state = _MS.MOVE_ACTIVE
                info.need_redraw = True
                return True
            self._reset()
            info.move_state = _MS.IDLE
            return False

        # ---- 非单手场景（无手 / 双手）：不触发；ACTIVE 时容忍短暂丢失 ----
        if int(n_hands) != 1 or lms is None:
            _hold_during_lost()
            return info

        # ---- 单手但非张开：让正常控鼠/点击工作；ACTIVE 时同样先容忍 ----
        open_ok = bool(self._stabilize(bool(self.is_open_palm(lms)), dt))
        if not open_ok:
            _hold_during_lost()
            return info

        # ---- 单手 + 张开：进入/续跑移窗状态机 ----
        self._lost_sec = 0.0
        info.ui_active = True

        # IDLE → CONFIRM
        if self._move_state == _MS.IDLE:
            self._move_state = _MS.MOVE_CONFIRM
            self._move_confirm = 0.0
            info.move_state = _MS.MOVE_CONFIRM
            info.confirm_sec = 0.0
            info.need_redraw = True
            return info

        # CONFIRM：累计墙钟，达标后选中窗口
        if self._move_state == _MS.MOVE_CONFIRM:
            self._move_confirm += float(dt)
            info.confirm_sec = self._move_confirm
            if self._move_confirm >= MOVE_CONFIRM_SEC:
                hwnd = 0
                if screen_xy is not None:
                    try:
                        pt = wintypes.POINT()
                        pt.x = wintypes.LONG(int(round(float(screen_xy[0]))))
                        pt.y = wintypes.LONG(int(round(float(screen_xy[1]))))
                        hwnd_hit = int(WindowFromPoint(pt) or 0)
                        hwnd = self._promote_hit_hwnd(hwnd_hit)
                    except Exception:
                        hwnd = 0
                if hwnd == 0:
                    try:
                        fg = int(GetForegroundWindow() or 0)
                        hwnd = int(self._promote_hit_hwnd(fg) or fg or 0)
                    except Exception:
                        hwnd = 0
                rc = RECT()
                ok = bool(GetWindowRect(wintypes.HWND(hwnd), ctypes.byref(rc)))
                if not ok or rc.right <= rc.left or rc.bottom <= rc.top:
                    self._reset()
                    info.need_redraw = True
                    info.move_state = _MS.IDLE
                    return info
                try:
                    SetForegroundWindow_API(wintypes.HWND(hwnd))
                except Exception:
                    pass
                self._move_hwnd = hwnd
                self._move_left0 = float(rc.left)
                self._move_top0 = float(rc.top)
                self._move_w0 = float(rc.right - rc.left)
                self._move_h0 = float(rc.bottom - rc.top)
                if screen_xy is not None:
                    self._move_ref_sx = float(screen_xy[0])
                    self._move_ref_sy = float(screen_xy[1])
                else:
                    self._move_ref_sx = self._move_left0
                    self._move_ref_sy = self._move_top0
                # 光标立刻锁窗口中心（避免跳动误触）
                try:
                    SetCursorPos(
                        wintypes.INT(int(round(self._move_left0 + self._move_w0 * 0.5))),
                        wintypes.INT(int(round(self._move_top0 + self._move_h0 * 0.5))),
                    )
                except Exception:
                    pass
                self._move_state = _MS.MOVE_ACTIVE
                info.move_state = _MS.MOVE_ACTIVE
                info.move_hwnd = hwnd
                info.window_rect = (int(rc.left), int(rc.top), int(rc.right), int(rc.bottom))
            info.need_redraw = True
            return info

        # ACTIVE：按掌心屏幕位移移动窗口（只改 x/y）
        if self._move_state == _MS.MOVE_ACTIVE and self._move_hwnd != 0:
            cur_sx, cur_sy = self._move_ref_sx, self._move_ref_sy
            if screen_xy is not None:
                cur_sx = float(screen_xy[0])
                cur_sy = float(screen_xy[1])
            dx_screen = cur_sx - self._move_ref_sx
            dy_screen = cur_sy - self._move_ref_sy
            i_left = int(round(self._move_left0 + dx_screen))
            i_top = int(round(self._move_top0 + dy_screen))
            try:
                SetWindowPos(
                    wintypes.HWND(self._move_hwnd),
                    wintypes.HWND(HWND_TOP),
                    wintypes.INT(i_left), wintypes.INT(i_top),
                    wintypes.INT(0), wintypes.INT(0),
                    wintypes.UINT(SWP_MOVE_FLAGS),
                )
            except Exception:
                pass
            rc = RECT()
            if bool(GetWindowRect(wintypes.HWND(self._move_hwnd), ctypes.byref(rc))):
                info.window_rect = (int(rc.left), int(rc.top), int(rc.right), int(rc.bottom))
                # 每帧重锁窗口中心：唯一光标写入源，防误触
                try:
                    SetCursorPos(
                        wintypes.INT(int(round((rc.left + rc.right) * 0.5))),
                        wintypes.INT(int(round((rc.top + rc.bottom) * 0.5))),
                    )
                except Exception:
                    pass
            info.delta_xy = (float(dx_screen), float(dy_screen))
            info.move_hwnd = self._move_hwnd
            info.move_state = _MS.MOVE_ACTIVE
            info.need_redraw = True
            return info

        info.need_redraw = True
        return info
