# -*- coding: utf-8 -*-
"""
双手和弦：左手握拳（修饰门）+ 右手转动 → 鼠标滚轮
================================================================

规则只有四条：
1. 【入队】非 MISS（有数值）的观测点 (t,x,y) 直接入 20 点 FIFO；
   MISS 帧不入队（队列保留，MISS 后三点仍可判）。
2. 【取点】只用队列最后三点 P0→P1→P2，两条边航向角 φ01、φ12。
3. 【方向】仅看夹角符号：S=wrap(φ12−φ01)；屏幕 y 向下，
   S>0=视觉顺时针→下滑(-1)，S<0=逆时针→上滑(+1)。
4. 【出量】q = d角度/dt × SCROLL_K；方向已含在 S 符号里（顺负逆正）。
   两条边的航向 φ01/φ12 分别是两段弧【中点时刻】 (t0+t1)/2、(t1+t2)/2
   的方向，故角度变化 S 的真实时间间隔 = (t2−t0)/2 = 相邻点平均间隔：
   w = S / ((t2−t0)/2)（标准中心差分，非均匀/MISS 间隔同样成立）。

没有静止点替换、没有断档吸附、没有形状门/角度门/速度门、
没有确认计数、没有同号确认、没有跨帧方向记忆。LOCK/FLIP/RELEASE
只是给 UI 的【派生显示】，不参与任何行为。

输入降噪（非判据）：tip 坐标入队前做一阶 EMA（POS_SMOOTH=.65），
抑制 MediaPipe 关键点 1~2px 抖动对航向角的放大（小位移时航向对
x/y 噪声极敏感，1px 噪声可使夹角符号翻转）。平滑后的坐标即"有数值
的点"，不改变三点判据本身。

和弦入口（由 hands_tasks 计算 _chord_scroll）：
  双手同帧 且 标签交换后 "Right"（物理左手）手势 == FIST。
active=False 经 ACTIVE_GRACE_SEC(0.25s) 宽限才 reset（仅防和弦手
瞬时丢帧抖动，与三点判据无关）。

输入坐标：TIP 像素按画面短边归一（16:9 画面 x 合法范围 0~1.78）。
"""
from __future__ import annotations

import math
import time
from collections import deque

# ---------------- 参数 ----------------
PTS_MAX          = 20     # FIFO 点队列容量（用户指定），每点 (t, x, y)
POS_SMOOTH       = 0.12   # 坐标 EMA 平滑系数（输入降噪，非判据）；越小越抗抖
SCROLL_K         = 0.05   # 唯一增益：q(格/观测点) = K·(S/dt)，S/dt 单位 rad/s。
MIN_Q            = 0.15   # 动态 q 最小下限（格/帧，18 wheel 单位 > _WHEEL_SEND_MIN=2）：慢圈也保证单帧可投递
Q_SMOOTH         = 0.4    # q 幅值 EMA 平滑系数（符号即时切换，只平滑速度）；越小越流畅
STATIC_PATH_MIN  = 0.02   # 静止位移门限（归一化坐标，约14px@720p）：三点总位移小于此值视为静止，q=0
                          # 物理含义=每个观测点覆盖 K 秒；实测相机约 20fps
                          # （每点 .05s）→ 一圈总滚量≈2π≈6.28 格，真机标定
ACTIVE_GRACE_SEC = 0.25   # active=False 宽限：宽限内不 reset（和弦手丢帧防抖）
TRAIL_MAX        = 40     # UI 轨迹点数
SPEED_REF        = 2.0 * math.pi * 1.5  # UI 速度条满幅角速度（1.5圈/秒）

_TWO_PI = 2.0 * math.pi


def _wrap_pi(a: float) -> float:
    """航向角差夹到 (-π, π]——跨 ±π 边界必须选连续分支，否则方向翻转。"""
    while a > math.pi:
        a -= _TWO_PI
    while a <= -math.pi:
        a += _TWO_PI
    return a


class CircleScrollDetector:
    """最后三点航向夹角 → 方向看符号、出量=d角度/dt。无判据状态机。"""

    def __init__(self):
        self._trail: deque = deque(maxlen=TRAIL_MAX)
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        """和弦解除/离镜：队列、轨迹、派生显示全部清零。"""
        self._trail.clear()
        self._pts: deque = deque(maxlen=PTS_MAX)   # (t, x, y) 全部非 MISS 点
        self._sx: float | None = None              # EMA x（输入降噪）
        self._sy: float | None = None              # EMA y
        # ---- 仅 UI 派生显示，不参与任何判定 ----
        self._dir = 0
        self._ev = ""
        self._ev_dir = 0
        self._ev_t = 0.0
        self._lost_t: float | None = None
        self._q_ema: float = 0.0   # q 幅值 EMA（只平滑 |q|，符号即时切换）
        # ---- 逐帧诊断（每次由最新三点重算）----
        self._dbg_kind = ""
        self._dbg_q = 0.0
        self._dbg_w = 0.0
        self._dbg_S = 0.0
        self._dbg_cross = 0.0
        self._dbg_dot = 0.0
        self._dbg_path = 0.0
        self._dbg_p0 = ""
        self._dbg_p1 = ""
        self._dbg_p2 = ""
        self._now = 0.0

    # ------------------------------------------------------------------
    def update(self, active: bool, x: float | None, y: float | None,
               now: float | None = None) -> float:
        """喂入一帧，返回本帧滚轮量（notch，+上滑/-下滑，0=无）。

        active=True 且坐标 None = MISS：不入队、q=0、队列原样保留。
        active=False = 和弦解除：宽限后 reset。
        """
        if now is None:
            now = time.monotonic()
        self._now = now
        self._ev = ""
        self._dbg_q = 0.0

        if not active:
            if self._lost_t is None:
                self._lost_t = now
            if now - self._lost_t >= ACTIVE_GRACE_SEC:
                self._dbg_kind = "INACTIVE_RESET"
                self.reset()
                return 0.0
            self._dbg_kind = "INACTIVE_GRACE"
            return 0.0                    # 宽限内冻结
        self._lost_t = None

        # ================= MISS：不入队、不出量 =================
        if x is None or y is None:
            self._dbg_kind = "MISS"
            return 0.0

        # ================= 非 MISS 点：坐标 EMA 后入队（带时间戳）=================
        x, y = float(x), float(y)
        if self._sx is None:
            self._sx, self._sy = x, y
        else:
            self._sx = POS_SMOOTH * x + (1.0 - POS_SMOOTH) * self._sx
            self._sy = POS_SMOOTH * y + (1.0 - POS_SMOOTH) * self._sy
        self._dbg_kind = "FRESH"
        self._pts.append((now, self._sx, self._sy))
        self._trail.append((self._sx, self._sy))

        # ================= 最后三点：方向看符号，出量=d角度/dt =================
        # 用队列最后三个相邻点（不回溯），避免 MISS 导致三点跨越多个画圈周期。
        # MISS 问题在检测层解决，检测器层不做跨圈补偿。
        q = 0.0
        if len(self._pts) >= 3:
            (t0, x0, y0), (t1, x1, y1), (t2, x2, y2) = (
                self._pts[-3], self._pts[-2], self._pts[-1])
            dx1, dy1 = x1 - x0, y1 - y0
            dx2, dy2 = x2 - x1, y2 - y1
            cross = dx1 * dy2 - dy1 * dx2
            dot = dx1 * dx2 + dy1 * dy2
            S = math.atan2(cross, dot)
            dt_mid = 0.5 * (t2 - t0)
            self._dbg_S = S
            self._dbg_cross = cross
            self._dbg_dot = dot
            self._dbg_path = math.hypot(dx1, dy1) + math.hypot(dx2, dy2)
            self._dbg_p0 = f"{x0:.3f},{y0:.3f}"
            self._dbg_p1 = f"{x1:.3f},{y1:.3f}"
            self._dbg_p2 = f"{x2:.3f},{y2:.3f}"
            if dt_mid > 0.0:
                w = S / dt_mid
                self._dbg_w = w
                # 静止检测：三点总位移太小视为手静止，不出量（采样约束，非判据门限）
                if self._dbg_path < STATIC_PATH_MIN:
                    q = 0.0
                else:
                    # 动态 q = -SCROLL_K * w，但下限 MIN_Q，保证单帧投递量 > _WHEEL_SEND_MIN，不卡死
                    dynamic_q = -SCROLL_K * w
                    if abs(dynamic_q) < MIN_Q:
                        q = math.copysign(MIN_Q, dynamic_q)
                    else:
                        q = dynamic_q
                    # q 幅值 EMA：只平滑速度，符号即时切换（方向不延迟）
                    sign = 1.0 if q > 0.0 else (-1.0 if q < 0.0 else 0.0)
                    if sign == 0.0:
                        self._q_ema = 0.0
                    else:
                        self._q_ema = Q_SMOOTH * abs(q) + (1.0 - Q_SMOOTH) * self._q_ema
                        q = sign * self._q_ema

        # ---- 派生 UI 事件：方向只是 q 符号的显示，零行为后果 ----
        td = -1 if q < 0.0 else (1 if q > 0.0 else 0)
        if td != self._dir:
            if td == 0:
                self._ev = "RELEASE"
            elif self._dir == 0:
                self._ev = "LOCK"
            else:
                self._ev = "FLIP"
            self._ev_dir = td
            self._ev_t = float(now)
            self._dir = td

        self._dbg_q = q
        return q

    # ------------------------------------------------------------------
    def metrics(self) -> dict:
        """UI/日志快照（纯观测），键名保持与 hands_tasks 打印兼容。"""
        w = self._dbg_w
        return {
            "state": "LOCKED" if self._dir != 0 else "IDLE",
            "dir": self._dir,
            "m_dir": self._dir,             # UI 轨迹预览色
            "speed": float(abs(w)),
            "omega": float(w),
            "speed_pct": float(min(1.0, abs(w) / SPEED_REF)),
            "moving": bool(self._dir != 0),
            "trail": self._trail,
            "ev": self._ev,
            "ev_dir": self._ev_dir,
            "ev_t": self._ev_t,
            # ---- [CS-DBG] 逐帧诊断字段（旧列位保留，无用项给常量）----
            "kind": self._dbg_kind,
            "q": float(self._dbg_q),
            "keep": False,
            "wnow": float(w),
            "win_n": len(self._pts),
            "win_tspan": (float(self._pts[-1][0] - self._pts[0][0])
                          if len(self._pts) >= 2 else 0.0),
            "fit_R": 0.0,
            "rmse": 0.0,
            "path": float(self._dbg_path),
            "m_delta_deg": math.degrees(abs(self._dbg_S)),
            "s_deg": math.degrees(self._dbg_S),   # 带符号夹角（>0 顺）
            "cross": float(self._dbg_cross),
            "dot": float(self._dbg_dot),
            "p0": self._dbg_p0,
            "p1": self._dbg_p1,
            "p2": self._dbg_p2,
            "td": int(self._dir),
            "cf": 0,
            "stall_age": -1.0,
            "fresh_age": (float(self._now - self._pts[-1][0])
                          if self._pts else 0.0),
        }
