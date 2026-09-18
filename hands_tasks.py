"""
hands_tasks.py  —— 体感鼠标全部业务逻辑的纯封装（零复杂线程/状态，仅四件套函数）
================================================================================

四件套：
  1. parse_args()                → 解析命令行参数（与原 main.py 完全一致）
  2. clamp_args(args)            → 对 args 做硬约束范围裁剪（原地修改）
  3. init_models(args)           → 创建 detector/mouse/tap/stretch 等全部组件
  4. run_loop(components, ...)   → 对应原 main() 里「打开摄像头 → 识别循环 → 清理」的全部逻辑

设计原则（极简）：
  · 没有任何后台线程、服务、锁 —— 调用者想放什么线程就放什么线程
  · 只有 run_loop 是阻塞的；stop_event.set() 会在下一帧开头立即干净退出
  · status_store 是普通 dict，调用者（server.py）随便读，不加任何锁（单写多读=Python GIL 安全）
"""

import argparse
import copy as _cp
import math as _mt
import os
import re
import threading
import time
from collections import deque  # ⭐ 新增：palm_n 滑窗稳定性判断（index_bend 误判过滤）
# ⭐ 2026-09-10 摄像头加固：必须在 import cv2 之前设置——
#   禁用 OpenCV MSMF 后端的硬件 MediaFoundation 变换（部分 UVC 相机因此持续输出黑帧，OpenCV 官方已知问题）
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
import cv2
import numpy as np

from hand_detector import HandDetector, TasksHandDetector, HAND_CONNECTIONS
from gesture_recognizer import recognize_gesture
from mouse_controller import MouseController
from draw_cn_text import put_text_cn
from finger_tap_click_detector import (FingerTapClickDetector, TwoFingerRightClickDetector,
                                       _S as _TS, E as _TEV, TWO_FINGER_TIP_ENTRY_R)
from diagonal_stretch import DiagonalStretchManager, _S as _DS, _MODE as _DM, _MS as _DMS

# ---------------- 常量（原 main.py 顶部，原样搬） ----------------
HAND_COLORS = {"Left": (255, 100, 0), "Right": (0, 140, 255)}
FINGER_NAMES_CN = ["拇", "食", "中", "无", "小"]
COMBOS = []  # 双手组合：用户要求全部置空


# ======================================================================
# 辅助函数（原 main.py 里的 _draw_finger_state / _measure_cn_text_size / _draw_tap_bar）
# ======================================================================
def _draw_finger_state(frame, ext_tuple, ratios, origin, color):
    x0, y0 = origin
    for i, name in enumerate(FINGER_NAMES_CN):
        state = "开" if ext_tuple[i] else "弯"
        ratio_key = {"拇": "thumb_r", "食": "index_r", "中": "middle_r",
                     "无": "ring_r", "小": "pinky_r"}[name]
        val = ratios.get(ratio_key, 0.0)
        text = f"{name}:{state} ({val:.2f})"
        c = (0, 220, 0) if ext_tuple[i] else (0, 0, 220)
        put_text_cn(frame, text, (x0, y0 + i * 22),
                    font_scale=0.55, color=c, thickness=2)
    pinch = ratios.get("pinch_r", None)
    line = y0 + 5 * 22 + 10
    put_text_cn(frame, f"掌宽: {ratios.get('palm_w', 0):.3f}",
                (x0, line), font_scale=0.5, color=color, thickness=2)
    if pinch is not None:
        threshold = 0.28
        pin_c = (0, 255, 220) if pinch < threshold else (200, 200, 200)
        tag = "✓触发" if pinch < threshold else ""
        put_text_cn(frame, f"捏合: {pinch:.3f} {tag}",
                    (x0, line + 20), font_scale=0.5, color=pin_c, thickness=2)


def _measure_cn_text_size(text: str, font_scale: float):
    px = max(10, int(font_scale * 33))
    w_sum = 0
    for c in text:
        w_sum += px if ord(c) > 127 else px // 2
    return w_sum, int(px * 1.25)


def _three_point_angle(p_a: np.ndarray, p_v: np.ndarray, p_b: np.ndarray) -> float:
    """计算 ∠A-V-B 的夹角（以 V 为顶点），返回 0~180 度（纯数值，已做除零保护）。
    向量 VA = A - V, VB = B - V；cosθ = VA·VB / (|VA||VB|)
    用于关节弯曲：伸直≈180°，折 90°≈90°，对折≈0°。
    """
    va = p_a - p_v
    vb = p_b - p_v
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    denom = na * nb
    if denom < 1e-8:
        return 180.0  # 无法判断 → 认为伸直（保守，避免误触）
    cos_v = float(np.dot(va, vb) / denom)
    cos_v = -1.0 if cos_v < -1.0 else (1.0 if cos_v > 1.0 else cos_v)
    return float(_mt.degrees(_mt.acos(cos_v)))


def _index_bend_from_joint_angles(lms, palm_denom: float | None,
                                  dist_bend_fallback: float | None = None,
                                  joint_ids=(5, 6, 7, 8)) -> float:
    """⭐ 2026-08-25 统一弯曲度量 v3（位置无关 + 手势匹配真实点击）
    输出值域：0.20 全弯 ~ 0.88 伸直；越大=越伸直（与旧语义兼容，detector 阈值平滑迁移）。

    joint_ids: (MCP, PIP, DIP, TIP) 关键点序号——食指(5,6,7,8)默认；
               中指传 (9,10,11,12) 复用同一度量（2026-09-16 双指并拢右键手势）。

    ✅ 统一设计 3 大原则（用户要求：不分区补偿、匹配真实点击、位置无关）：
    1) 关节权重匹配真实点击：PIP 0.40 + MCP 0.20 + DIP 0.40
       （人类食指点击动作 = 主要弯 DIP 末端关节，PIP/MCP 几乎不动 → DIP 必须高权重）
    2) angle/dist 融合统一 95% angle + 5% dist：
       （dist_bend 分母 dist(#5,#17)=掌心宽度 天然受 透视/远近/画面边缘clip 影响 = 位置相关变量；
         三点夹角 angle 是纯旋转度量、位置无关 → 让 angle 绝对主导，dist 只做 5% 兜底防崩溃）
    3) 无任何分区 if/else（彻底删除 _top_band_hint 条带切换、顶部特殊权重补偿）
    """
    # 取点（用 x,y 不用 z，MediaPipe z 在 2D 画面里尺度不稳定；投影夹角本身就对深度折叠鲁棒）
    _jm, _jp, _jd, _jt = joint_ids
    p0   = np.array([lms[0].x,  lms[0].y],  dtype=np.float64)   # WRIST 腕部
    p5   = np.array([lms[_jm].x, lms[_jm].y], dtype=np.float64)  # 指根 MCP（食5/中9）
    p6   = np.array([lms[_jp].x, lms[_jp].y], dtype=np.float64)  # PIP 中间关节（食6/中10）
    p7   = np.array([lms[_jd].x, lms[_jd].y], dtype=np.float64)  # DIP 指尖前（食7/中11）
    p8   = np.array([lms[_jt].x, lms[_jt].y], dtype=np.float64)  # TIP 指尖（食8/中12）
    return _bend_from_points(p0, p5, p6, p7, p8, palm_denom, dist_bend_fallback)


def _bend_from_points(p_wrist, p_mcp, p_pip, p_dip, p_tip,
                      palm_denom: float | None,
                      dist_bend_fallback: float | None = None) -> float:
    """⭐ 从显式 5 点计算单指/双指单元的弯曲度（与 _index_bend_from_joint_angles 完全同公式）。
    双指并拢右键用：p_mcp/p_pip/p_dip/p_tip 传食中两指对应关节的中点 → 得到"双指单元"的合成 bend。
    输出值域：0.20 全弯 ~ 0.88 伸直；越大=越伸直。
    """
    p0, p5, p6, p7, p8 = p_wrist, p_mcp, p_pip, p_dip, p_tip

    # (A) PIP 角：∠MCP_5-PIP_6-DIP_7  （辅助，兜底用户偶尔弯 PIP 的动作）
    theta_pip = _three_point_angle(p5, p6, p7)
    # (B) MCP 角：∠WRIST_0-MCP_5-PIP_6  （兜底，避免极端假动作）
    theta_mcp = _three_point_angle(p0, p5, p6)
    # (C) DIP 角：∠PIP_6-DIP_7-TIP_8  （主力！人类真实点击 = 只弯 DIP 末节）
    theta_dip = _three_point_angle(p6, p7, p8)

    # 归一化到 0=伸直(180°) 1=弯成0°
    def _norm_180(t: float) -> float:
        t = 0.0 if t < 0.0 else (180.0 if t > 180.0 else float(t))
        return (180.0 - t) / 180.0

    n_pip = _norm_180(theta_pip)  # 0=伸直 1=完全对折
    n_mcp = _norm_180(theta_mcp)
    n_dip = _norm_180(theta_dip)

    # ✅ 统一原则1：DIP=主(0.60) + PIP=辅(0.25) + MCP=兜底(0.15)
    #   ⭐ 2026-08-26 数值铁证：轻点击(∠DIP=140°弯40°)
    #     旧权重0.40/0.20/0.40 → final放大后 bratio=0.855 > MIN_ABS=0.75 → 必须明显深弯才触发 ❌
    #     新权重0.25/0.15/0.60 → final放大后 bratio=0.740 ≤ MIN_ABS=0.75 → 轻点击也能立即触发 ✅
    #   中等点击(DIP折到120°=弯60°) → n_dip=0.333 → bratio=0.58 ✅
    #   深点击(DIP折到30°=弯150°) → n_dip=0.833 → bratio=0.28 ✅
    #   移动微角度 ±2° → n_dip=0.011 → bratio 变化≈0.007 → DI=0.007 远不够误触 ✅
    W_PIP, W_MCP, W_DIP = 0.25, 0.15, 0.60
    bend_01 = W_PIP * n_pip + W_MCP * n_mcp + W_DIP * n_dip
    bend_01 = 0.0 if bend_01 < 0.0 else (1.0 if bend_01 > 1.0 else float(bend_01))

    # v2 线性映射：angle → 0.20 弯曲 ~ 0.88 伸直（SLOPE 放大，Δ 更明显=差分更灵敏）
    SLOPE_ANGLE = 0.95
    BASELINE, FLOOR = 0.88, 0.20
    angle_abs = BASELINE - bend_01 * SLOPE_ANGLE
    angle_abs = float(max(FLOOR, min(BASELINE + 0.08, angle_abs)))

    # ★ dist_bend_fallback 先归一到 0.20~0.88（和 angle 同尺度）再融合
    DIST_NORM_STRAIGHT = 2.40   # dist_bend 伸直时的典型值（= TIP-MCP / palm_n）
    DIST_NORM_BENT     = 0.90   # dist_bend 完全弯曲时的典型值
    def _norm_dist_bend(db: float) -> float:
        t = (float(db) - DIST_NORM_BENT) / (DIST_NORM_STRAIGHT - DIST_NORM_BENT)
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        return FLOOR + t * (BASELINE - FLOOR)  # db=2.4→t=1→0.88; db=0.9→t=0→0.20 ✅ 语义一致

    # ✅ 统一原则2：95% angle（位置无关） + 5% dist（仅兜底，避免 angle 计算失败时全崩）
    #   - 顶部 clip / 透视缩短 / 掌心远近：只有 dist 受影响，5% 权重影响从旧 40%→最多 ±0.02 final → 可忽略=位置无关
    W_ANGLE, W_DIST = 0.95, 0.05
    if dist_bend_fallback is None:
        final = float(angle_abs)
    else:
        dist_normed = _norm_dist_bend(float(dist_bend_fallback))
        final = W_ANGLE * float(angle_abs) + W_DIST * float(dist_normed)

    # ★ v2 关键：噪声门 + 真弯曲 Δ 放大
    #   · drop < 2.5%（≈ 抖动/掌心翻转伪信号）：完全不放大，保持原融合值=稳
    #   · drop ≥ 2.5%（真的食指弯下去了）：超过门的部分 ×2.0 放大 → 差分/绝对门限更容易触发=灵
    DROP_MULT = 2.0
    DROP_NOISE_GATE = 0.025
    drop = BASELINE - float(final)
    if drop > DROP_NOISE_GATE:
        final = BASELINE - DROP_MULT * (drop - DROP_NOISE_GATE) - DROP_NOISE_GATE
    final = float(max(FLOOR, min(BASELINE + 0.08, final)))

    # palm_denom 归一化的弱修正：如果 palm_denom 离镜头特别远（太小），就把值稍微推回伸直
    # （避免远/近导致绝对尺度偏移影响 detector 学习 baseline）
    if palm_denom is not None and palm_denom > 1e-6:
        scale_fix = 0.98 + 0.04 * min(1.0, palm_denom / 0.25)
        final = final * scale_fix

    return float(round(final, 6))


def _draw_tap_bar(frame, tm: dict, det: FingerTapClickDetector | None = None):
    H, W = frame.shape[:2]
    bar_x0, bar_y0 = W - 320, 36
    bar_w, bar_h = 300, 28

    vis_min, vis_max = 0.20, 1.00
    def _pct(x): return (x - vis_min) / (vis_max - vis_min)
    bend = max(vis_min, min(vis_max, tm["bend"])) if tm else 0.80
    bend_px = int(bar_x0 + bar_w * _pct(bend))

    cv2.rectangle(frame, (bar_x0 - 2, bar_y0 - 2),
                  (bar_x0 + bar_w + 2, bar_y0 + bar_h * 3 + 14),
                  (0, 0, 0), -1)
    cv2.rectangle(frame, (bar_x0, bar_y0), (bar_x0 + bar_w, bar_y0 + bar_h),
                  (60, 60, 60), -1)
    bx = int(bar_x0 + bar_w * _pct(tm["baseline"]))
    cv2.line(frame, (bx, bar_y0), (bx, bar_y0 + bar_h), (255, 255, 255), 1)
    ex = int(bar_x0 + bar_w * _pct(tm["exit_t"]))
    cv2.line(frame, (ex, bar_y0), (ex, bar_y0 + bar_h), (60, 220, 255), 2)
    en = int(bar_x0 + bar_w * _pct(tm["enter_t"]))
    cv2.line(frame, (en, bar_y0), (en, bar_y0 + bar_h), (0, 0, 255), 2)

    st = tm["state"]
    if st in (_TS.BEND_ARMED, _TS.BEND_HOLD, _TS.TAPPED):
        fill_c = (0, 0, 255)
    elif st == _TS.WAIT_DBL:
        fill_c = (255, 0, 255)  # 双击等待窗口=品红（单击挂起中）
    elif st == _TS.PRESSED:
        fill_c = (0, 200, 255)
    elif st == _TS.COOLDOWN:
        fill_c = (140, 140, 140)
    else:
        fill_c = (0, 220, 140)
    cv2.rectangle(frame, (bar_x0, bar_y0), (bend_px, bar_y0 + bar_h), fill_c, -1)
    tri_points = np.array([
        [bend_px, bar_y0 - 4],
        [bend_px - 6, bar_y0 - 11],
        [bend_px + 6, bar_y0 - 11],
    ], np.int32)
    cv2.drawContours(frame, [tri_points], 0, fill_c, -1)

    state_cn_map = {
        _TS.IDLE:       ("伸直待机(基线学习中)", (160, 255, 160)),
        _TS.BEND_ARMED: ("食指弯曲确认中…",      (80, 180, 255)),
        _TS.BEND_HOLD:  (f"弯曲保持 {tm['hold_frames']} 帧",  (0, 180, 255)),
        _TS.TAPPED:     ("✅ TAP 单击触发!",     (60, 60, 255)),
        _TS.WAIT_DBL:   (f"等待第二次点击(双击)… {tm.get('dbl_window_left', 0):.2f}s", (255, 0, 255)),
        _TS.PRESSED:    ("🔒 按住拖拽中",          (0, 200, 255)),
        _TS.COOLDOWN:   (f"冷却 {tm['cooldown_left']} 帧",   (180, 180, 180)),
    }
    st_text, st_color = state_cn_map.get(st, (str(st), (255, 255, 255)))
    line2_y = bar_y0 + bar_h + 6
    put_text_cn(frame, st_text, (bar_x0, line2_y),
                font_scale=0.55, color=st_color, thickness=2)

    line3_y = line2_y + 22
    tap_max   = float(tm.get("TAP_MAX", 18))
    long_max  = float(tm.get("LONG_MAX", 22))
    tot_f     = float(tm.get("total_frames", 0))
    hold_f    = float(tm.get("hold_frames", 0))
    stab_f    = float(tm.get("stable_frames", 0))

    pb_w = int((bar_w - 10) / 2)
    pb_h = 10
    pb1_x = bar_x0
    p1 = 0.0 if tot_f <= 0 else min(1.2, tot_f / tap_max)
    cv2.rectangle(frame, (pb1_x, line3_y), (pb1_x + pb_w, line3_y + pb_h), (60, 60, 60), -1)
    tap_c = (30, 255, 90) if p1 < 0.7 else ((60, 220, 255) if p1 < 1.0 else (0, 0, 255))
    cv2.rectangle(frame, (pb1_x, line3_y), (pb1_x + int(pb_w * min(1.0, p1)), line3_y + pb_h), tap_c, -1)
    put_text_cn(frame, f"短按窗口 {int(tot_f):>3d}/{int(tap_max):>2d}", (pb1_x, line3_y - 14),
                font_scale=0.32, color=tap_c, thickness=1)
    pb2_x = bar_x0 + pb_w + 10
    p2 = 0.0 if hold_f <= 0 else min(1.0, hold_f / long_max)
    cv2.rectangle(frame, (pb2_x, line3_y), (pb2_x + pb_w, line3_y + pb_h), (60, 60, 60), -1)
    long_c = (40, 220, 255) if p2 < 0.5 else ((0, 150, 255) if p2 < 1.0 else (0, 0, 255))
    cv2.rectangle(frame, (pb2_x, line3_y), (pb2_x + int(pb_w * p2), line3_y + pb_h), long_c, -1)
    put_text_cn(frame, f"长按计数 {int(hold_f):>3d}/{int(long_max):>2d}  稳{int(stab_f):>2d}f",
                (pb2_x, line3_y - 14), font_scale=0.32, color=long_c, thickness=1)

    line4_y = line3_y + pb_h + 10
    s4 = f"bend={bend:.2f} 基线={tm['baseline']:.2f}"
    put_text_cn(frame, s4, (bar_x0, line4_y),
                font_scale=0.42, color=(200, 220, 220), thickness=1)
    line5_y = line4_y + 20
    s5 = (f"弯入线={tm['enter_t']:.2f}(红) 伸直线={tm['exit_t']:.2f}(黄)"
          f" 单击×{tm['tap_count']} 长按×{tm['long_count']}")
    put_text_cn(frame, s5, (bar_x0, line5_y),
                font_scale=0.36, color=(160, 180, 200), thickness=1)
    if det is not None:
        line6_y = line5_y + 20
        hp = tm.get("hardness_pct", 60)
        sp = tm.get("speed_pct", 60)
        enter_r = (det.params.get("ENTER_BEND_RATIO", 0.78) * 100) if det else 78.0
        drop    = (det.params.get("MIN_BEND_DROP",      0.12) * 100) if det else 12.0
        s6 = f"硬度 H={hp:>3d}% 速度 S={sp:>3d}%  弯幅≥{100-enter_r:.0f}%+降{drop:.0f}%"
        put_text_cn(frame, s6, (bar_x0, line6_y),
                    font_scale=0.38, color=(120, 220, 255), thickness=1)


# ======================================================================
# ① parse_args()  —— 完全对应原 main.py L195-L229
# ======================================================================
def parse_args(argv=None):
    """解析命令行参数（argv=None 时用 sys.argv；传 list 则用 list，便于 server.py 传入自定义参数）。"""
    parser = argparse.ArgumentParser(description="双手体感鼠标 (DualHand -> Mouse)")
    parser.add_argument("--use-tasks", action="store_true",
                        help="使用 Tasks API + 本地 hand_landmarker.task 模型模式（纯离线，不依赖网络）")
    parser.add_argument("--model", type=str, default=None,
                        help="（仅 --use-tasks 时有效）指定 hand_landmarker.task 的绝对/相对路径")
    parser.add_argument("--max-hands", type=int, default=2, help="最多识别手的数量 (默认 2)")
    parser.add_argument("--anchor", type=str, default="palm", choices=["palm", "index_tip"],
                        help="鼠标光标锚点：palm=掌心 / index_tip=食指TIP")
    parser.add_argument("--cursor-mode", type=str, default="smooth", choices=["smooth", "absolute", "direct"],
                        help="光标移动策略：smooth(AR悬停5层稳) / absolute(指哪打哪3层兼顾稳跟手) / direct⭐严格跟UI掌心绿圈(无阻尼无锁，翻转手掌也不卡)")
    parser.add_argument("--roi-scale", type=float, default=0.70,
                        help="画面→屏幕映射中心区域大小(0.30~1.00)")
    parser.add_argument("--roi-offset-y", type=float, default=0.62,
                        help="ROI长方形在画面高度的上下位置（-1.0=贴顶, 0.0=居中, +0.62=默认偏下12%防顶部clip, +1.0=贴底）")
    parser.add_argument("--roi-safe-shrink", type=float, default=0.82,
                        help="【2026-08-24】在 roi-scale 原始 ROI 长方形基础上的安全子 ROI 等比内缩比例(0.50~1.00)；越小虚线框越小，指尖越不易超画面边。默认 0.82。")
    parser.add_argument("--tap-hardness", type=float, default=0.70,
                        help="点击弯曲幅度门槛 0.00~1.00")
    parser.add_argument("--tap-speed", type=float, default=0.80,
                        help="点击识别响应速度 0.00~1.00")
    return parser.parse_args(argv)


# ======================================================================
# ② clamp_args(args)  —— 完全对应原 main.py L230-L236
# ======================================================================
def clamp_args(args):
    """硬约束 args 数值范围（原地修改）。"""
    if args.roi_scale < 0.30: args.roi_scale = 0.30
    if args.roi_scale > 1.00: args.roi_scale = 1.00
    if args.roi_offset_y < -1.0: args.roi_offset_y = -1.0
    if args.roi_offset_y > +1.0: args.roi_offset_y = +1.0
    if args.roi_safe_shrink < 0.50: args.roi_safe_shrink = 0.50
    if args.roi_safe_shrink > 1.00: args.roi_safe_shrink = 1.00
    if args.tap_hardness < 0.0: args.tap_hardness = 0.0
    if args.tap_hardness > 1.0: args.tap_hardness = 1.0
    if args.tap_speed    < 0.0: args.tap_speed    = 0.0
    if args.tap_speed    > 1.0: args.tap_speed    = 1.0
    return args


# ======================================================================
# ③ init_models(args)  —— 完全对应原 main.py L247-L267（不打开摄像头！）
# ======================================================================
def init_models(args):
    """
    创建所有识别/控制组件（**不打开摄像头**）。

    返回 dict，key 即组件名：
      {args, detector, mouse, tap_detector, stretch, mode_label, mode_color}
    """
    detector_kwargs = dict(max_hands=args.max_hands,
                           detection_confidence=0.65,   # 从 0.7 下调：更容易检出边缘角度的手（配合 EMA 去抖不抖）
                           tracking_confidence=0.55)   # 从 0.6 下调：降低 tracking 门限，跟手更稳
    if args.use_tasks:
        if args.model:
            model_path = args.model
        else:
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "assets", "hand_landmarker.task",
            )
        detector = TasksHandDetector(model_asset_path=model_path, **detector_kwargs)
        mode_label = f"[MODE: 本地模型 Tasks]  {os.path.basename(model_path)}"
        mode_color = (0, 200, 255)
    else:
        detector = HandDetector(**detector_kwargs)
        mode_label = "[MODE: 默认 Solutions]  mediapipe 内置模型"
        mode_color = (180, 255, 120)

    mouse = MouseController(alpha=0.25, cursor_mode=args.cursor_mode, roi_scale=args.roi_scale,
                            roi_offset_y=args.roi_offset_y, roi_safe_shrink=args.roi_safe_shrink)
    tap_detector = FingerTapClickDetector(hardness=args.tap_hardness, speed=args.tap_speed)
    # 食中双指并拢勾按 → 右键（与左键同 hardness/speed 档位手感）
    right_tap_detector = TwoFingerRightClickDetector(hardness=args.tap_hardness, speed=args.tap_speed)
    stretch = DiagonalStretchManager()

    return dict(
        args=args,
        detector=detector,
        mouse=mouse,
        tap_detector=tap_detector,
        right_tap_detector=right_tap_detector,
        stretch=stretch,
        mode_label=mode_label,
        mode_color=mode_color,
    )


# ======================================================================
# 摄像头稳健打开（2026-09-10 加固，解决"冷启动黑屏/卡死，先开 Windows 设置测试后才能用"）
# ======================================================================
_CAM_WARMUP_FRAMES  = 25    # 预热读帧上限（30fps≈0.8s，等自动曝光/白平衡收敛 + USB 管线跑通）
_CAM_MIN_NONBLACK   = 6     # 预热期至少多少帧"非全黑"才判定相机活着
_CAM_BLACK_MEAN_THR = 10.0  # 帧平均亮度 < 此值 = 黑帧（正常室内画面均值通常 40~130；死相机输出≈0）


def _warmup_capture(cap, max_frames=_CAM_WARMUP_FRAMES, min_nonblack=_CAM_MIN_NONBLACK):
    """打开后连续读帧预热：丢黑帧、等自动曝光收敛。
    返回 (ok, n_read, n_nonblack, elapsed_ms, last_mean)。
    两阶段：冷启动相机自动曝光/USB 管线唤醒可能 >25 帧——若读得到帧但全黑，
    再续读一轮（共 ~50 帧 ≈1.6s）；一帧都读不到（设备死/卡）则不等第二轮。"""
    t0 = time.perf_counter()
    n_read = 0
    n_nonblack = 0
    last_mean = 0.0
    for _phase in range(2):
        for _ in range(max_frames):
            try:
                ok_r, frame = cap.read()
            except Exception:
                ok_r, frame = False, None
            if not ok_r or frame is None:
                time.sleep(0.02)
                continue
            n_read += 1
            try:
                last_mean = float(frame.mean())
            except Exception:
                last_mean = 0.0
            if last_mean >= _CAM_BLACK_MEAN_THR:
                n_nonblack += 1
                if n_nonblack >= min_nonblack:
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    return True, n_read, n_nonblack, elapsed_ms, last_mean
        # 一帧都没读到 = 设备死/卡住，不必再等第二轮
        if n_read == 0:
            break
        # 读到帧但全黑 = 可能冷启动曝光未收敛，续读一轮
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return (n_nonblack >= min_nonblack), n_read, n_nonblack, elapsed_ms, last_mean


def _run_in_com_thread(fn):
    """在一次性 MTA 线程里执行 COM 枚举函数，返回其结果（异常返回 []）。
    🔴 关键（2026-09-10 真机实测）：绝不能在"稍后要用 OpenCV MSMF 开相机"的
       线程里调用 CoInitializeEx——该线程无论初始化为 MTA 还是 STA，cap_msmf
       的设备激活都会走跨单元慢路径：VideoCapture 构造 ~18s + 设分辨率 ~58s
       （对比：未初始化 COM 的线程构造只要 0.4s）。因此所有 CoInitializeEx
       只在这个专用线程里做，线程结束前 CoUninitialize、线程随即销毁，
       对调用方线程（run_loop 开相机线程）零 COM 状态污染。"""
    import ctypes
    box = {}

    def _work():
        try:
            ctypes.windll.ole32.CoInitializeEx(None, 0)  # MTA
        except Exception:
            pass
        try:
            box["r"] = fn()
        except Exception as e:
            box["e"] = repr(e)
        finally:
            try:
                ctypes.windll.ole32.CoUninitialize()
            except Exception:
                pass

    th = threading.Thread(target=_work, daemon=True)
    th.start()
    th.join()
    if "e" in box:
        return []
    return box.get("r") or []


def _list_dshow_cameras():
    """枚举 DirectShow 视频输入设备（公开入口，内部在专用 COM 线程执行）。"""
    return _run_in_com_thread(_dshow_enum_impl)


def _dshow_enum_impl():
    """枚举 DirectShow 视频输入设备，返回 [(index, friendly_name, device_path), ...]；失败返回 []。
    纯 ctypes COM（零新依赖）。
    🔴 这是"对标 Windows 设置"的关键：设置按设备身份（USB 符号链接/友好名）枚举、点哪台开哪台；
       OpenCV 只能按枚举索引打开，且 DSHOW/MSMF 枚举顺序不一致、MSMF 打开失败会静默回退默认设备。
       先拿到真实清单再按名字打开，才能保证"选对物理摄像头"。"""
    import ctypes
    from ctypes import wintypes

    class _GUID(ctypes.Structure):
        _fields_ = [("Data1", wintypes.DWORD),
                    ("Data2", wintypes.WORD),
                    ("Data3", wintypes.WORD),
                    ("Data4", ctypes.c_ubyte * 8)]

    class _VARIANT(ctypes.Structure):
        _fields_ = [("vt", wintypes.USHORT),
                    ("_r1", wintypes.USHORT),
                    ("_r2", wintypes.USHORT),
                    ("_r3", wintypes.USHORT),
                    ("bstrVal", ctypes.c_void_p),
                    ("_pad", ctypes.c_void_p)]

    ole32 = ctypes.windll.ole32
    try:
        ole32.CoInitializeEx(None, 0)  # COINIT_MULTITHREADED；已初始化/模式冲突返回值忽略
    except Exception:
        pass
    # 🔴 不要给共享的 ole32 函数设置带局部 _GUID 类的 argtypes：windll 函数对象的
    #    argtypes 是进程级缓存，局部 _GUID 类每次调用都新建，会让另一个枚举函数
    #    （_mf_enum_impl，用自己的局部 _GUID）调用时抛 ArgumentError、被吞成"0 台"。
    _CLSIDFromString = ole32.CLSIDFromString
    _CoCreateInstance = ole32.CoCreateInstance

    def _guid(s):
        g = _GUID()
        _CLSIDFromString(s, ctypes.byref(g))
        return g

    CLSID_SystemDeviceEnum = _guid("{62BE5D10-60EB-11d0-BD3B-00A0C911CE86}")
    CLSID_VideoInputDevice  = _guid("{860BB310-5D01-11d0-BD3B-00A0C911CE86}")
    IID_ICreateDevEnum      = _guid("{29840822-5B84-11D0-BD3B-00A0C911CE86}")
    IID_IEnumMoniker        = _guid("{00000102-0000-0000-C000-000000000046}")
    IID_IPropertyBag        = _guid("{55272A00-42CB-11CE-8135-00AA004BB851}")

    def _vtbl(p, idx):
        tbl = ctypes.cast(p, ctypes.POINTER(ctypes.c_void_p))[0]
        return ctypes.cast(tbl, ctypes.POINTER(ctypes.c_void_p))[idx]

    def _release(p):
        if p:
            try:
                ctypes.WINFUNCTYPE(wintypes.ULONG, ctypes.c_void_p)(_vtbl(p, 2))(p)
            except Exception:
                pass

    dev_enum = ctypes.c_void_p()
    hr = _CoCreateInstance(ctypes.byref(CLSID_SystemDeviceEnum), None, 1 | 4,
                           ctypes.byref(IID_ICreateDevEnum), ctypes.byref(dev_enum))
    if hr != 0 or not dev_enum:
        return []
    mon_enum = ctypes.c_void_p()
    _CreateEnum = ctypes.WINFUNCTYPE(wintypes.LONG, ctypes.c_void_p,
                                     ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p),
                                     wintypes.DWORD)
    hr = _CreateEnum(_vtbl(dev_enum, 3))(dev_enum, ctypes.byref(CLSID_VideoInputDevice),
                                         ctypes.byref(mon_enum), 0)
    _release(dev_enum)
    if hr != 0 or not mon_enum:
        return []

    cams = []
    try:
        _Next = ctypes.WINFUNCTYPE(wintypes.LONG, ctypes.c_void_p,
                                   wintypes.ULONG, ctypes.POINTER(ctypes.c_void_p),
                                   ctypes.POINTER(wintypes.ULONG))
        _BindToStorage = ctypes.WINFUNCTYPE(wintypes.LONG, ctypes.c_void_p,
                                            ctypes.c_void_p, ctypes.c_void_p,
                                            ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p))
        _Read = ctypes.WINFUNCTYPE(wintypes.LONG, ctypes.c_void_p,
                                   wintypes.LPCWSTR, ctypes.POINTER(_VARIANT), ctypes.c_void_p)
        f_next = _Next(_vtbl(mon_enum, 3))
        idx = 0
        while True:
            mon = ctypes.c_void_p()
            fetched = wintypes.ULONG(0)
            if f_next(mon_enum, 1, ctypes.byref(mon), ctypes.byref(fetched)) != 0 \
                    or not fetched.value or not mon:
                break
            name, path = "", ""
            bag = ctypes.c_void_p()
            # IMoniker vtable：IUnknown(0-2) → IPersist::GetClassID(3) →
            # IPersistStream::IsDirty/Load/Save/GetSizeMax(4-7) → IMoniker::BindToObject(8)/BindToStorage(9)
            if _BindToStorage(_vtbl(mon, 9))(mon, None, None,
                                             ctypes.byref(IID_IPropertyBag), ctypes.byref(bag)) == 0 and bag:
                f_read = _Read(_vtbl(bag, 3))
                for key, is_path in (("FriendlyName", False), ("Description", False), ("DevicePath", True)):
                    var = _VARIANT()
                    if f_read(bag, key, ctypes.byref(var), None) == 0 and var.vt == 8 and var.bstrVal:
                        try:
                            val = ctypes.cast(var.bstrVal, wintypes.LPWSTR).value or ""
                        except Exception:
                            val = ""
                        try:
                            ctypes.windll.oleaut32.SysFreeString(var.bstrVal)
                        except Exception:
                            pass
                        if is_path:
                            path = path or val
                        else:
                            name = name or val
                _release(bag)
            _release(mon)
            cams.append((idx, name or f"Camera{idx}", path))
            idx += 1
    finally:
        _release(mon_enum)
    return cams


def _list_mf_cameras():
    """枚举 Media Foundation 视频捕获设备（公开入口，内部在专用 COM 线程执行）。"""
    return _run_in_com_thread(_mf_enum_impl)


def _mf_enum_impl():
    """枚举 Media Foundation 视频捕获设备（实时清单），返回 [(index, friendly_name, symbolic_link), ...]。
    🔴 这就是 Windows 设置里的相机列表，也是 OpenCV MSMF 后端内部使用的同一枚举
       （cap_msmf 用 MFEnumDeviceSources 按序取第 N 个）——因此这里拿到的 index 与
       cv2.VideoCapture(i, CAP_MSMF) 打开的物理设备严格一一对应。
       相比之下 DSHOW 过滤器表在摄像头重插/换 USB 口后会残留陈旧 moniker（isOpened=False）。
    纯 ctypes 调 mfplat.dll，零新依赖。"""
    import ctypes
    from ctypes import wintypes

    class _GUID(ctypes.Structure):
        _fields_ = [("Data1", wintypes.DWORD),
                    ("Data2", wintypes.WORD),
                    ("Data3", wintypes.WORD),
                    ("Data4", ctypes.c_ubyte * 8)]

    def _guid(s):
        g = _GUID()
        ctypes.windll.ole32.CLSIDFromString(s, ctypes.byref(g))
        return g

    # 属性键 GUID（本机 mfidl.h 实测，EXTERNS_GUID 原值；网上旧博客流传的
    # {975E5F5E...}/{CA179C15...} 等是错的，用错会导致 MFEnumDeviceSources
    # 返回 MF_E_ATTRIBUTENOTFOUND(0xC00D36E6)）
    _KEY_SOURCE_TYPE = _guid("{C60AC5FE-252A-478F-A0EF-BC8FA5F7CAD3}")
    _VAL_VIDCAP       = _guid("{8AC3587A-4AE7-42D8-99E0-0A6013EEF90F}")
    _KEY_FRIENDLY     = _guid("{60D0E559-52F8-4FA2-BBCE-ACDB34A8EC01}")
    _KEY_SYMLINK      = _guid("{58F0AAD8-22BF-4F8A-BB3D-D2C4978C6E2F}")

    try:
        mfplat = ctypes.windll.mfplat
        mf = ctypes.windll.mf  # MFEnumDeviceSources 在 mf.dll（不在 mfplat.dll）
    except Exception:
        return []
    # MFStartup 传 MF_VERSION = (MF_SDK_VERSION=2 << 0) | (MF_API_VERSION=0x70 << 16)
    # = 0x00020070。注意：① 0x700002 字节序反了会被拒(0xC00D36E3)；
    # ② 传 0x2 虽然返回 S_OK，但平台实际未完全初始化。
    # 进程内只启动一次（引用计数），且永不 MFShutdown——OpenCV MSMF 后端也要用 MF。
    if not getattr(_list_mf_cameras, "_mf_started", False):
        try:
            if mfplat.MFStartup(0x00020070, 0) == 0:
                _list_mf_cameras._mf_started = True
        except Exception:
            return []

    mfplat.MFCreateAttributes.restype = wintypes.LONG
    mfplat.MFCreateAttributes.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                          ctypes.c_uint32]

    def _vtbl(p, idx):
        tbl = ctypes.cast(p, ctypes.POINTER(ctypes.c_void_p))[0]
        return ctypes.cast(tbl, ctypes.POINTER(ctypes.c_void_p))[idx]

    def _release(p):
        if p:
            try:
                ctypes.WINFUNCTYPE(wintypes.ULONG, ctypes.c_void_p)(_vtbl(p, 2))(p)
            except Exception:
                pass

    attrs = ctypes.c_void_p()
    if mfplat.MFCreateAttributes(ctypes.byref(attrs), 1) != 0 or not attrs:
        return []
    try:
        # IMFAttributes vtable 顺序（本机 mfobjects.h 实测，Get 系列在前 Set 在后，
        # 与很多旧资料写的序号不同）：
        #   GetGUID=10 / GetAllocatedString=13 / DeleteAllItems=20 /
        #   SetUINT32=21 / SetGUID=24 / SetString=25 / GetCount=30
        _SetGUID = ctypes.WINFUNCTYPE(wintypes.LONG, ctypes.c_void_p,
                                      ctypes.POINTER(_GUID), ctypes.POINTER(_GUID))
        if _SetGUID(_vtbl(attrs, 24))(attrs, ctypes.byref(_KEY_SOURCE_TYPE),
                                      ctypes.byref(_VAL_VIDCAP)) != 0:
            return []

        mf.MFEnumDeviceSources.restype = wintypes.LONG
        mf.MFEnumDeviceSources.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)),
            ctypes.POINTER(ctypes.c_uint32)]
        arr = ctypes.POINTER(ctypes.c_void_p)()
        count = ctypes.c_uint32(0)
        if mf.MFEnumDeviceSources(attrs, ctypes.byref(arr), ctypes.byref(count)) != 0:
            return []

        # IMFAttributes::GetAllocatedString = vtable slot 13
        _GetStr = ctypes.WINFUNCTYPE(wintypes.LONG, ctypes.c_void_p,
                                     ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p),
                                     ctypes.POINTER(ctypes.c_uint32))
        _CoTaskMemFree = ctypes.windll.ole32.CoTaskMemFree
        cams = []
        try:
            for i in range(int(count.value)):
                act = arr[i]
                name, link = "", ""
                for key, is_link in ((_KEY_FRIENDLY, False), (_KEY_SYMLINK, True)):
                    out = ctypes.c_void_p()
                    nch = ctypes.c_uint32(0)
                    if _GetStr(_vtbl(act, 13))(act, ctypes.byref(key),
                                               ctypes.byref(out), ctypes.byref(nch)) == 0 and out:
                        try:
                            val = ctypes.cast(out, wintypes.LPWSTR).value or ""
                        except Exception:
                            val = ""
                        _CoTaskMemFree(out)
                        if is_link:
                            link = link or val
                        else:
                            name = name or val
                cams.append((i, name or f"Camera{i}", link))
                _release(act)
        finally:
            try:
                _CoTaskMemFree(arr)
            except Exception:
                pass
        return cams
    finally:
        _release(attrs)


def _open_backend_once(cid, be, be_name, round_idx, target_w, target_h, _mjpg, attempts, dev_name=None):
    """单次指定后端打开 + 协商 + 预热。成功返回 (cap, info_fields)，失败返回 (None, None)。
    注意：OpenCV 5.0 的 DSHOW/MSMF 后端都只接受 index（不支持按名字打开）——设备身份
    核验在 _open_camera_robust 的枚举层完成（MF 枚举顺序与 MSMF 后端严格一致），
    dev_name 仅用于日志展示。"""
    name, be = be_name, be
    cap = None
    try:
        t0 = time.perf_counter()
        _tag = f"index{int(cid)}({dev_name!r})" if dev_name else f"index{int(cid)}"
        cap = cv2.VideoCapture(int(cid), be)
        if cap is None or not cap.isOpened():
            attempts.append(f"[{name}#轮{round_idx}]({_tag}) isOpened=False（设备不存在/被占用）")
            if cap is not None:
                try: cap.release()
                except Exception: pass
            return None, None

        # ---- 低延迟：驱动缓冲队列压到 1 帧（2026-09-15）----
        #   不设置时部分驱动（尤其 MSMF）会缓存 2~N 帧旧帧，cap.read() 拿到的是旧画面 → 鼠标位置整体慢半拍（~33ms+/帧）。
        #   DSHOW 普遍支持该属性；MSMF 部分驱动静默不接受 → set 无效但无害，预热多读几帧也能排空旧帧。
        try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception: pass

        # ---- 媒体类型协商：先 MJPG，再分辨率、帧率 ----
        try: cap.set(cv2.CAP_PROP_FOURCC, _mjpg)
        except Exception: pass
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
        try: cap.set(cv2.CAP_PROP_FPS, 30)
        except Exception: pass
        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        afps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

        # ---- 目标分辨率没协商上 → 降 VGA 重试 ----
        if aw < target_w - 80 or ah < target_h - 80:
            try: cap.set(cv2.CAP_PROP_FOURCC, _mjpg)
            except Exception: pass
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            attempts.append(f"[{name}#轮{round_idx}] 目标{target_w}×{target_h}协商失败，已降 640×480")

        # ---- 预热 + 黑帧检测 ----
        ok_w, n_read, n_nonblack, w_ms, last_mean = _warmup_capture(cap)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        try: _buf_n = int(cap.get(cv2.CAP_PROP_BUFFERSIZE) or 0)
        except Exception: _buf_n = -1
        attempts.append(
            f"[{name}#轮{round_idx}]({_tag}) open=OK {aw}×{ah}@{afps:.0f}fps buf={_buf_n} "
            f"预热:读{n_read}帧/非黑{n_nonblack}/末帧亮度{last_mean:.0f} "
            f"预热{w_ms:.0f}ms/总{elapsed_ms:.0f}ms → {'✔采用' if ok_w else '✗全黑或死帧，弃用'}"
        )
        if ok_w:
            return cap, dict(
                backend=name, backend_id=int(be), width=aw, height=ah, fps=afps,
                warmup_read=n_read, warmup_nonblack=n_nonblack,
                warmup_ms=int(w_ms), open_ms=int(elapsed_ms), round=round_idx,
                index_warning=False, device_name=(dev_name or f"index{int(cid)}"),
            )
        try: cap.release()
        except Exception: pass
        return None, None
    except Exception as e:
        attempts.append(f"[{name}#轮{round_idx}] exception: {e!r}")
        if cap is not None:
            try: cap.release()
            except Exception: pass
        return None, None


def _open_camera_robust(cid, target_w=1280, target_h=720):
    """稳健打开摄像头，返回 (cap, info_dict)；全部失败时 cap=None。

    🔴 对标 Windows 设置（2026-09-10 真机定论）：
      · 设置里的相机走 Media Foundation 实时枚举（MFEnumDeviceSources）+ Frame Server
        管道，重插/换 USB 口都能秒开；OpenCV 的 MSMF 后端内部用的是**同一个枚举**，
        所以 MSMF(index) 的 index 与我们自己枚举 MF 拿到的 index 严格一一对应。
      · DSHOW 过滤器表在相机重插后会残留陈旧 moniker（实测 isOpened=False），
        且枚举顺序与 MF 不一致（实测同机 MF: [0]外接[1]内置；DSHOW: [0]内置[1]外接）。
    🔴 统一索引（2026-09-10 起）：
      · **MF 实时清单是唯一的索引空间**——int cid = MF index、str cid = MF 友好名片段，
        与 Windows 设置 / /cameras 接口返回的 index 完全一致；
      · DSHOW 清单**不再参与定位**：仅在需要 DSHOW 备选后端时，按设备身份
        （VID/PID，其次友好名）把同一物理设备翻译成 DSHOW 自己的本地 index 去打开；
        查不到同名设备就跳过 DSHOW，❌绝不按位置开别的相机；
      · 仅当 MF 枚举整体失败（清单为空）才降级用 DSHOW 清单定位，并显式告警。
    流程：MF 实时清单定位 → MSMF 优先（与设置同管道）两轮预热唤醒，
          失败再用身份映射后的 DSHOW 备选；MJPG 先设、分辨率降级、两阶段预热黑帧检测。
    """
    attempts = []
    try:
        _mjpg = cv2.VideoWriter_fourcc(*"MJPG")
    except Exception:
        _mjpg = 0
    t_all = time.perf_counter()

    # ---- ① 双清单枚举：MF（实时真源） + DSHOW（参考） ----
    mf_cams, ds_cams = [], []
    try:
        mf_cams = _list_mf_cameras()
    except Exception as e:
        attempts.append(f"[枚举] MF 设备枚举异常: {e!r}")
    try:
        ds_cams = _list_dshow_cameras()
    except Exception as e:
        attempts.append(f"[枚举] DSHOW 设备枚举异常: {e!r}")
    if mf_cams:
        attempts.append(f"[枚举] MF(=Windows设置同源) 检测到 {len(mf_cams)} 台相机: "
                        + "；".join(f"[{i}] {n!r}" for i, n, _p in mf_cams))
    else:
        attempts.append("[枚举] MF 未枚举到相机（该管道不可用）")
    if ds_cams:
        attempts.append(f"[枚举] DSHOW 检测到 {len(ds_cams)} 台相机: "
                        + "；".join(f"[{i}] {n!r}" for i, n, _p in ds_cams))

    def _vidpid(path):
        """从设备路径/符号链接提取 'vid_xxxx&pid_yyyy'（小写），用于跨后端物理身份匹配。"""
        m = re.search(r"vid_[0-9a-f]{4}&pid_[0-9a-f]{4}", (path or ""), re.IGNORECASE)
        return m.group(0).lower() if m else None

    def _ds_idx_of_identity(dev_name, dev_path):
        """把 MF 选定的物理设备翻译成 DSHOW 本地 index：VID/PID 严格匹配优先，
        其次友好名精确/包含匹配；查不到返回 None（DSHOW 备选后端跳过，绝不按位置乱开）。"""
        target_vp = _vidpid(dev_path)
        if target_vp:
            for i, n, p in ds_cams:
                if _vidpid(p) == target_vp:
                    return i
        low = (dev_name or "").lower()
        for i, n, _p in ds_cams:
            if low and low == (n or "").lower():
                return i
        for i, n, _p in ds_cams:
            if low and low in (n or "").lower():
                return i
        return None

    # ---- ② 定位目标设备：MF 实时清单 = 唯一索引空间 ----
    mf_idx, ds_idx, dev_name, dev_path = None, None, None, None
    degraded = not mf_cams  # MF 管道整体不可用 → 才降级用 DSHOW 清单
    if degraded and ds_cams:
        attempts.append("[选择] ⚠ MF 实时枚举为空（管道不可用）→ 降级模式："
                        "临时按 DSHOW 清单定位（index 以 DSHOW 清单为准），建议重启服务恢复 MF。")
    pool = ds_cams if degraded else mf_cams

    if isinstance(cid, str):
        key = cid.lower().strip()
        hit = [c for c in pool if key in (c[1] or "").lower()]
        if not hit:
            extra = ""
            if not degraded:
                ds_stale = [c for c in ds_cams if key in (c[1] or "").lower()]
                if ds_stale:
                    extra = ("；该名字只出现在 DSHOW 陈旧清单中——相机未真正在线"
                             "（重插后 DSHOW moniker 残留），请重插相机并先在 Windows 设置里确认")
            attempts.append(f"❌ 按名字 {cid!r} 未匹配到在线相机（实时清单见上{extra}）。")
            return None, dict(attempts_summary=attempts)
        dev_name, dev_path = hit[0][1], hit[0][2]
        if degraded:
            ds_idx = hit[0][0]
        else:
            mf_idx = hit[0][0]
    else:
        cid_int = int(cid)
        if cid_int >= len(pool):
            src = "MF 实时清单" if not degraded else "DSHOW 清单（降级模式）"
            attempts.append(
                f"❌ camera_id={cid_int} 越界：{src} 只有 {len(pool)} 台"
                + ("：" + "；".join(f"[{i}] {n!r}" for i, n, _p in pool) if pool else "（为空）")
                + f"。请用 /cameras 查看实时清单，或把 config.json 的 default_camera_id "
                f"改为清单中的 index 或相机名字片段（如 \"HD\"）。❌ 不静默回退到其他相机。"
            )
            return None, dict(attempts_summary=attempts)
        dev_name, dev_path = pool[cid_int][1], pool[cid_int][2]
        if degraded:
            ds_idx = cid_int
        else:
            mf_idx = cid_int

    # 正常模式：DSHOW 仅按物理身份映射为备选后端的本地 index
    if not degraded:
        ds_idx = _ds_idx_of_identity(dev_name, dev_path)
    attempts.append(
        f"[选择] 目标设备 {dev_name!r}：统一 MF index={mf_idx}"
        + (f"；DSHOW 备选后端映射到同身份设备 index={ds_idx}"
           if ds_idx is not None else "；DSHOW 清单无同身份设备（备选后端跳过）")
    )

    # ---- ③ 按计划打开：MSMF 优先（设置同管道），DSHOW 仅备选；各两轮预热唤醒 ----
    plans = []
    if mf_idx is not None:
        plans.append(("MSMF", cv2.CAP_MSMF, mf_idx))
    if ds_idx is not None:
        plans.append(("DSHOW", cv2.CAP_DSHOW, ds_idx))
    if not plans:
        attempts.append(f"❌ 相机 {dev_name!r} 无法定位到任何可打开的后端"
                        f"（MF 清单为空且 DSHOW 清单也无此设备）。"
                        f"请拔插相机后先在 Windows 设置里确认能看到它，再重试；或用相机名字选择。")
        return None, dict(attempts_summary=attempts)

    for be_name, be, idx in plans:
        for round_idx in (1, 2):
            cap, info = _open_backend_once(idx, be, be_name, round_idx,
                                           target_w, target_h, _mjpg, attempts, dev_name=dev_name)
            if cap is not None:
                info["attempts_summary"] = attempts
                info["open_ms"] = int((time.perf_counter() - t_all) * 1000.0)
                info["index_warning"] = False  # 已通过 MF/DSHOW 双清单身份锚定
                return cap, info
            if round_idx == 1:
                time.sleep(0.3)  # 给驱动 teardown/唤醒缓冲，再重开

    attempts.append(
        f"❌ 相机 {dev_name!r} 经 MSMF/DSHOW 两轮尝试仍打不开或持续黑帧——"
        f"若 Windows 设置里能正常显示：请先关闭所有占用相机的程序（含本服务旧进程）后重试；"
        f"若设置里也是黑的：检查镜头遮挡/IR 红外相机/线缆。"
    )
    return None, dict(attempts_summary=attempts)


# ======================================================================
# ④ run_loop(components, camera_id, stop_event, preview, status_store)
#    完全对应原 main.py L239 (cap = VideoCapture) → L908 (finally destroyAllWindows)
# ======================================================================
def run_loop(components, camera_id=0, stop_event=None, preview=True, status_store=None):
    """
    阻塞式识别主循环。

    参数：
      components   : init_models() 返回的 dict（每次 /run 建议传新的组件，避免复用 closed detector）
      camera_id    : 摄像头选择，默认 0；int = MF 实时清单 index（与 Windows 设置同源，
                     /cameras 接口可查），str = 相机友好名片段（如 "HD"，不受枚举顺序影响）
      stop_event   : threading.Event；每帧开头先检查；set() 时立即干净退出（可 None = 纯脚本模式，靠 q 键退出）
      preview      : True=开 cv2.imshow 预览窗口；False=纯后台识别（不画窗）
      status_store : dict，写入实时状态（server.py 的 /status 接口直接读它）；None 时内部自动建一个（没用）

    写入 status_store 的字段（都是最简单的标量，读写无锁）：
      running            : bool   —— 是否处于主循环中（正常跑）
      fps                : float  —— 最近 ~0.5s 的 FPS（EMA 平滑）
      current_camera_id  : int | str — 实际使用的相机选择（MF index 或名字片段）
      start_time_ms      : int    —— 进入循环的时间戳（ms）
      uptime_ms          : int    —— 已运行毫秒数
      last_left_gesture  : str or None
      last_right_gesture : str or None
      stretch_mode       : str    —— "IDLE" / "ARMED_CONFIRM" / "SCALING" / "MOVE_CONFIRM" / "MOVE_ACTIVE"
      error              : str or None —— 异常信息（正常为 None）
      preview_active     : bool
    """
    args            = components["args"]
    detector        = components["detector"]
    mouse           = components["mouse"]
    tap_detector    = components["tap_detector"]
    right_tap_detector = components["right_tap_detector"]
    stretch         = components["stretch"]
    mode_label      = components["mode_label"]
    mode_color      = components["mode_color"]

    # ---------- 初始化 status_store ----------
    if status_store is None:
        status_store = {}
    start_ms = int(time.time() * 1000)
    status_store.clear()
    status_store.update(dict(
        running=False, fps=0.0, current_camera_id=camera_id,
        start_time_ms=start_ms, uptime_ms=0,
        last_left_gesture=None, last_right_gesture=None,
        stretch_mode="IDLE", error=None,
        preview_active=bool(preview),
    ))

    combo_hint = ""
    combo_hint_counter = 0
    # ⭐ 上一帧 stretch 状态（move_only 最前置在本帧 stretch.update 之前执行，只能读上一帧结果；
    #   首帧用 None 哨兵）
    stretch_info = None
    _fps_smooth = None
    _fps_last_t = time.perf_counter()
    _fps_frame_cnt = 0
    _fps_display = 0.0
    _fps_cam_read_ms = 0.0
    _fps_infer_ms = 0.0
    _fps_draw_ms = 0.0
    _t1 = _t2 = _t3 = 0.0

    # ================================================================
    # 🔥 速度 & 稳定性优化状态（EMA / Majority Vote / 自适应跳帧）
    # ================================================================
    # (1) landmarks EMA 平滑：每只手 21×3(x,y,z) 个点做指数滑动平均
    #     ✅ 2026-08-20 调参：α 从 0.45 → 0.85
    #       之前 α=0.45（55% 历史权重）→ 手掌/食指图标严重拖尾，用户抱怨"手已经移了10cm，图标还在后面追"。
    #       现在 α=0.85（15% 历史权重）→ 跟手优先（几乎 1:1 跟随），把"平滑去噪"工作留给 mouse_controller 内部的
    #       「中值杀尖峰 + Sigmoid 微移阻尼 + 静止方差锁」3 层流水线处理（那边只针对鼠标坐标，不影响UI绘制）。
    #       另外：掌心一次性位移 > _EMA_RESET_TOL（归一化）时，直接清空该手 EMA 缓存，强制硬跟手（彻底消除拖尾）。
    _EMA_ALPHA = 0.85
    _EMA_RESET_TOL = 0.015                 # 掌心单帧归一化位移超过此值 → 清 EMA 缓存（≈1280×720下≈10像素）
    _ema_landmarks: dict[str, list] = {}   # key = Left/Right, value = 21 个 [x,y,z] 列表
    _ema_miss_count: dict[str, int] = {}   # 连续丢失计数，超过阈值清空 EMA

    # (2) 手势标签 majority-vote 去抖：最近 N 帧投票取众数
    #     ✅ 2026-08-20 调参：窗口从 5 → 3
    #       之前 5 帧会让手势切换滞后 ~133ms（30fps），和EMA拖尾叠加后用户感觉"我手明明是OPEN，拉伸连线3秒才出来"。
    #       3 帧折中：单帧噪声（比如半帧弯曲误判）仍能被 2/3 多数票杀掉，但响应延迟从 133ms → 66ms。
    _MV_WINDOW = 3
    _mv_history: dict[str, list] = {}      # key = Left/Right, value = list[str] (gesture names)
    _mv_final: dict[str, str] = {}

    # (3) 自适应跳帧：
    #   · 连续 ≥_SKIP_MIN_STABLE 帧都稳定检测到手 + 掌心位移 < _SKIP_PX_TOL*min(w,h)
    #     → 下一帧**跳过推理**，直接复用上次结果 + EMA 预测延续
    #   · 好处：推理频率接近减半，FPS 明显提升；代价：延迟 +1 帧（16~20ms，人眼无感）
    #   · 任何手丢失 / 手势切换 / 掌心大跳 → 立即退出跳帧模式
    #     ✅ 2026-08-20 调参（解决拖尾的第三个因素）：
    #       SKIP_MIN_STABLE 6 → 12（之前连续6帧就跳=400ms就开始跳，手一动还在继续跳→放大拖尾；现在12帧=800ms才触发，更保守）
    #       SKIP_PX_TOL 0.012 → 0.006（约 4.3 像素，之前 8.6 像素=手抖算稳定也跳→实际慢动作时跳帧+EMA双重拖尾；现在只有真正完全不动才跳）
    _SKIP_MIN_STABLE = 12
    _SKIP_PX_TOL = 0.006                    # 归一化位移：0.006 ≈ 1280×720 下短边720 → ≈ 4.3 像素（真正静止才认定为稳定）
    _skip_stable_cnt = 0                    # 连续稳定帧计数
    _skip_last_hands_data = None            # 上次的 hands_data（深拷贝后备用）
    _skip_last_palms: dict[str, tuple] = {} # 上次掌心（归一化）：判断是否稳定
    _skip_skip_next = False                 # 下帧是否真的跳过推理
    _skip_was_skipped = False               # 刚刚发生了跳帧（写 status_store 用）
    _skip_total_frames = 0                  # 累计主循环帧数
    _skip_skipped_frames = 0                # 累计跳过推理帧数

    cap = None
    try:
        # ---------- 打开摄像头（2026-09-10 加固：MF 实时枚举统一索引 + MSMF 优先 +
        #            DSHOW 按身份映射备选 + MJPG + 两阶段预热黑帧检测） ----------
        _cid = camera_id  # int = MF 实时清单 index；str = 相机友好名片段（如 "HD"）
        print(f"[hands_tasks.run_loop] ⏳ 正在打开摄像头 camera_id={_cid!r}"
              f"（MF 实时枚举定位，MSMF 同 Windows 设置管道；冷启动最长约 8 秒…）", flush=True)
        cap, _cap_info = _open_camera_robust(_cid)
        if cap is None:
            _probe_log = _cap_info.get("attempts_summary", [])
            msg = f"无法打开摄像头 camera_id={_cid!r}，尝试记录: {'; '.join(_probe_log)}"
            if status_store is not None:
                status_store["error"] = msg
                try:
                    status_store["_diag_backend_probe"] = dict(
                        attempts_summary=_probe_log, camera_id=_cid,
                        success_attempt=0, success_backend=None)
                except Exception:
                    pass
            raise RuntimeError(msg)
        print(f"[hands_tasks.run_loop] ✅ 摄像头 {_cap_info.get('device_name', _cid)!r}"
              f"（camera_id={_cid!r}）打开成功"
              f"（后端={_cap_info['backend']} {_cap_info['width']}×{_cap_info['height']}@{_cap_info['fps']:.0f}fps，"
              f"预热非黑帧={_cap_info['warmup_nonblack']}/{_cap_info['warmup_read']}，"
              f"第{_cap_info['round']}轮命中，总耗时={_cap_info['open_ms']}ms）")
        for _pline in _cap_info.get("attempts_summary", []):
            print(f"    {_pline}")
        # 把探测结果写到 status_store（/status 接口可查看排障）
        if status_store is not None:
            try:
                status_store["_diag_backend_probe"] = dict(
                    attempts_summary=_cap_info.get("attempts_summary", []),
                    camera_id=_cid,
                    success_attempt=_cap_info.get("round", 1),
                    success_backend=_cap_info.get("backend"),
                    total_elapsed_ms=_cap_info.get("open_ms", 0),
                    actual_frame_w=_cap_info.get("width"),
                    actual_frame_h=_cap_info.get("height"),
                    actual_fps=_cap_info.get("fps"),
                    warmup_read=_cap_info.get("warmup_read"),
                    warmup_nonblack=_cap_info.get("warmup_nonblack"),
                    index_warning=bool(_cap_info.get("index_warning", False)),
                )
            except Exception:
                pass

        status_store["running"] = True

        # ================== 主循环（与原 main.py 唯一差别：加 stop_event 每帧退出） ==================
        while True:
            # 🔴 新增：stop_event 命中 → 立即干净退出
            if stop_event is not None and stop_event.is_set():
                break

            _t0 = time.perf_counter()
            ok, frame = cap.read()
            _t1 = time.perf_counter()
            if not ok:
                print("[hands_tasks.run_loop] 摄像头读取失败，退出")
                break

            # ---------- 单点镜像 ----------
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # 把摄像头实际宽高通知 MouseController：_norm_to_screen 才能和预览绿框 100% 对齐
            mouse.set_frame_size(w, h)

            # ================================================================
            # 🔥 自适应跳帧：稳定时隔帧跳过推理 → 推理 FPS 翻倍
            # ================================================================
            _skip_total_frames += 1
            _frame_inferred_this_time = True
            if _skip_skip_next and _skip_last_hands_data is not None:
                # 跳！直接复用上次 hands_data（稍后 landmarks 会被 EMA 再平滑 + 重绘）
                _skip_skipped_frames += 1
                _skip_was_skipped = True
                _frame_inferred_this_time = False
                hands_data = _cp.deepcopy(_skip_last_hands_data)
                # 跳帧也必须画关节连线（用户感知不到推理被跳过）
                for _h in hands_data:
                    _pts = [(int(float(_l.x) * w), int(float(_l.y) * h)) for _l in _h["landmarks"]]
                    for (_px, _py) in _pts:
                        cv2.circle(frame, (_px, _py), 3, (255, 0, 255), 2)
                    for _a, _b in HAND_CONNECTIONS:
                        if _a < len(_pts) and _b < len(_pts):
                            cv2.line(frame, _pts[_a], _pts[_b], (0, 255, 0), 2)
            else:
                _skip_was_skipped = False
                # ---------- 正常：双手检测 ----------
                hands_data, frame = detector.detect(frame)
                _skip_last_hands_data = _cp.deepcopy(hands_data)

            # ================================================================
            # 🔥 landmarks EMA 指数滑动平均：去关节抖动，提升小范围瞄准稳定性
            # ================================================================
            from hand_detector import _LmWrapper
            _cur_labels_in_frame: set = set()
            for _hand in hands_data:
                _lab = _hand["label"]
                _cur_labels_in_frame.add(_lab)
                _cur_arr = [[float(_p.x), float(_p.y), float(_p.z)] for _p in _hand["landmarks"]]

                # ✅ 掌心大位移硬重置（彻底消除快速移动时的EMA拖尾）：
                #   先计算当前掌心（关键点0 + 关键点9 的中点，和 cursor anchor 完全一致）
                #   如果这一帧掌心相对上一帧跳了 > _EMA_RESET_TOL（≈10px归一化），说明是真实快速移动，
                #   直接丢掉历史EMA（不拖尾巴），强制用当前值（然后下一帧再从这个点开始EMA）。
                _cx0_cur = 0.5 * (_cur_arr[0][0] + _cur_arr[9][0])
                _cy0_cur = 0.5 * (_cur_arr[0][1] + _cur_arr[9][1])
                _reset_this_hand = False
                if _lab in _ema_landmarks:
                    _last_arr = _ema_landmarks[_lab]
                    _cx0_prev = 0.5 * (_last_arr[0][0] + _last_arr[9][0])
                    _cy0_prev = 0.5 * (_last_arr[0][1] + _last_arr[9][1])
                    if abs(_cx0_cur - _cx0_prev) > _EMA_RESET_TOL or abs(_cy0_cur - _cy0_prev) > _EMA_RESET_TOL:
                        _reset_this_hand = True

                if _lab in _ema_landmarks and not _reset_this_hand:
                    _last_arr = _ema_landmarks[_lab]
                    _smoothed = [
                        [_EMA_ALPHA * _cur_arr[i][0] + (1 - _EMA_ALPHA) * _last_arr[i][0],
                         _EMA_ALPHA * _cur_arr[i][1] + (1 - _EMA_ALPHA) * _last_arr[i][1],
                         _EMA_ALPHA * _cur_arr[i][2] + (1 - _EMA_ALPHA) * _last_arr[i][2]]
                        for i in range(21)
                    ]
                else:
                    # 首次出现 / 丢失后重新出现 / 掌心大位移硬重置 → 用当前原数值（不拖历史尾巴）
                    if _reset_this_hand and _lab in _ema_landmarks:
                        pass  # 马上会用 _cur_arr 覆盖掉
                    _smoothed = _cur_arr
                _ema_landmarks[_lab] = _smoothed
                _ema_miss_count[_lab] = 0
                # 把平滑后的 wrapped 回填到 hand["landmarks"]（后续所有业务都用平滑值）
                _hand["landmarks"] = [_LmWrapper(_v[0], _v[1], _v[2]) for _v in _smoothed]
            # 丢失的手：miss_count 累计，超时清空
            for _lab in list(_ema_landmarks.keys()):
                if _lab not in _cur_labels_in_frame:
                    _ema_miss_count[_lab] = _ema_miss_count.get(_lab, 0) + 1
                    if _ema_miss_count[_lab] > 8:
                        _ema_landmarks.pop(_lab, None)
                        _ema_miss_count.pop(_lab, None)

            hand_results = {}

            # ======================================================================
            # Pass A：只收集 hand_results + 画「手标签/手指状态/指尖小圆」
            # ======================================================================
            # 🔥 手势 Majority-Vote 去抖：最近 5 帧投票取众数，避免手势标签 1 帧跳变
            _cur_palms_now: dict[str, tuple] = {}   # 记录本帧掌心（给下一帧跳帧稳定性判断用）
            _gestures_changed_any = False
            for hand in hands_data:
                label = hand["label"]
                lms = hand["landmarks"]
                gesture_raw, ext, ratios = recognize_gesture(lms)
                # --- Majority vote ---
                if label not in _mv_history:
                    _mv_history[label] = []
                hist = _mv_history[label]
                hist.append(gesture_raw)
                if len(hist) > _MV_WINDOW:
                    hist = hist[-_MV_WINDOW:]
                    _mv_history[label] = hist
                # 统计众数；平票时选**最新出现**的那个（保持响应性）
                counts: dict = {}
                for g in hist:
                    counts[g] = counts.get(g, 0) + 1
                max_c = max(counts.values())
                # 倒序遍历，取最后出现的 max_c 对应的 gesture
                gesture_final = gesture_raw
                for g in reversed(hist):
                    if counts.get(g, 0) == max_c:
                        gesture_final = g
                        break
                if label in _mv_final and _mv_final[label] != gesture_final:
                    _gestures_changed_any = True
                _mv_final[label] = gesture_final
                # 用投票后的手势（不改变 ext/ratios，保持原始弯曲度反馈）
                hand_results[label] = (gesture_final, ext, ratios, lms)
                # 记录掌心（归一化）
                _palm_x, _palm_y = HandDetector.cursor_anchor_from_landmarks(lms, "palm")
                _cur_palms_now[label] = (float(_palm_x), float(_palm_y))

                color = HAND_COLORS.get(label, (128, 128, 128))
                tip_px, tip_py = HandDetector.landmark_to_pixel(lms[8], w, h)
                cv2.circle(frame, (tip_px, tip_py), 11, color, -1)
                cv2.circle(frame, (tip_px, tip_py), 11, (255, 255, 255), 2)

                user_label = "右手" if label == "Left" else "左手"
                top_y = 38
                hand_title = f"[{user_label}] {gesture_final}"
                # 说明：Pass A 先执行（_stable_this 还没算 → 所以此处 B-2 跳帧标志尚未覆盖为 True/False），
                #       为避免 NameError，手标签中文每帧都画（这部分占比小，约 1~2ms）；
                #       B-2 跳帧大头（Tap条 / 拉伸卡片 / 顶部 / 底部帮助栏）在后面有 _stable_this 值后再生效，能省 8~12ms。
                if user_label == "左手":
                    put_text_cn(frame, hand_title, (12, top_y),
                                font_scale=0.85, color=color, thickness=2)
                    _draw_finger_state(frame, ext, ratios, (12, 65), color)
                else:
                    put_text_cn(frame, hand_title, (w - 260, top_y),
                                font_scale=0.85, color=color, thickness=2)
                    _draw_finger_state(frame, ext, ratios, (w - 260, 65), color)

            # ================================================================
            # 🔥 自适应跳帧：稳定性判定，决定下一帧是否跳过
            # ================================================================
            _stable_this = False
            if hands_data and (not _gestures_changed_any):
                # (a) 手数量与 labels 完全一致
                _prev_labels = set(_skip_last_palms.keys())
                _cur_labels = _cur_labels_in_frame
                if _prev_labels and _prev_labels == _cur_labels:
                    # (b) 所有手掌心位移都 < 容差
                    _all_small = True
                    for _lab in _cur_labels:
                        if _lab not in _skip_last_palms:
                            _all_small = False; break
                        _px0, _py0 = _skip_last_palms[_lab]
                        _px1, _py1 = _cur_palms_now[_lab]
                        if abs(_px1 - _px0) > _SKIP_PX_TOL or abs(_py1 - _py0) > _SKIP_PX_TOL:
                            _all_small = False; break
                    if _all_small:
                        _stable_this = True
            if _stable_this:
                _skip_stable_cnt += 1
            else:
                _skip_stable_cnt = 0
            # 隔帧跳：连续稳定 6 帧 → 下帧可跳过；但跳过之后下下帧必须检测（避免连跳失控）
            _skip_skip_next = (
                _skip_stable_cnt >= _SKIP_MIN_STABLE
                and (not _skip_was_skipped)
            )
            _skip_last_palms = dict(_cur_palms_now)

            # ======================================================================
            # 2026-08-22 回滚 B-2 中文跳帧：用户反馈"隔帧绘制仍然持续严重闪烁，影响观感"
            # 已验证：即便黑底+中文共跳帧，cv2.imshow 每帧输出画面仍会被底层缓冲翻转导致"肉眼可见的闪烁"
            # → 恢复最原始方式：所有中文 put_text_cn / 黑底 / 背景矩形 每帧必画，完全不跳帧
            #   代价：稳定时 CPU 多消耗 ~8~12ms（FPS 降 ~3-5）；收益：UI 100% 不闪烁
            # ======================================================================
            _should_draw_cn_text_now = True  # 冗余兼容，后续会完全删除使用处的 if 条件（保持向下兼容）

            # ======================================================================
            # Pass B：基于**完整的 hand_results** 统一计算 全局门控
            #          （原 main.py 里这段放在 for 里=顺序错误，现在统一算一次，100% 顺序正确）
            # ======================================================================
            # (B1) 全局 disable：影响「Tap-Click」和「block_mouse_move_for_stretch」
            left_gesture_global   = hand_results.get("Left",  (None,))[0]
            right_gesture_global  = hand_results.get("Right", (None,))[0]
            left_pinch_global  = bool(left_gesture_global  is not None and left_gesture_global  != "OPEN")
            right_pinch_global = bool(right_gesture_global is not None and right_gesture_global != "OPEN")
            left_open_global   = bool(left_gesture_global  == "OPEN")
            right_open_global  = bool(right_gesture_global == "OPEN")
            any_fist_global = bool(left_gesture_global == "FIST" or right_gesture_global == "FIST")
            left_near_fist_global  = bool(("Left"  in hand_results) and sum(1 for x in hand_results["Left"][1]  if not x) >= 4)
            right_near_fist_global = bool(("Right" in hand_results) and sum(1 for x in hand_results["Right"][1] if not x) >= 4)
            any_near_fist_global   = left_near_fist_global or right_near_fist_global
            single_hand_exists = bool(("Left" in hand_results) ^ ("Right" in hand_results))
            single_open_global = bool(single_hand_exists and ((left_open_global and ("Left" in hand_results)) or (right_open_global and ("Right" in hand_results))))
            both_hands_global     = ("Left" in hand_results) and ("Right" in hand_results)
            pure_two_hand_stretch = bool(both_hands_global and left_pinch_global and right_pinch_global)
            pure_two_hand_move    = bool(both_hands_global and left_open_global  and right_open_global)
            disable_click_for_stretch_global = (
                both_hands_global
                or pure_two_hand_stretch
                or pure_two_hand_move
                # ⚠️ 2026-08-26 两次修复（点击完全失效根因最终版）：
                #   V1 删除 stretch.state/_move_state != IDLE：单手移动时 move_state 恒非IDLE=正常行为，不能禁点击
                #   V2 删除 any_fist_global / any_near_fist_global：
                #        near_fist_bend_count 统计「弯曲指节数≥4」会误判 POINT/OTHER 手势(仅食指伸=其他4指弯=0)
                #        → ext=(0,1,0,0,0) → count=4/5 → any_nfist=True → 每帧reset detector → bratio恒0/DI恒0
                #   仅保留纯双手语义手势条件(both_hands/pure_stretch/pure_move)：拉伸场景禁点击100%覆盖，单手点击通畅✅
            )
            block_mouse_move_for_stretch = disable_click_for_stretch_global

            # ⭐ 2026-09-17 修复「五指张开移窗时，光标在窗口中心↔掌心位置来回闪动」：
            #   MOVE_ACTIVE 期间 diagonal_stretch 每帧 SetCursorPos 锁窗口中心（防误触），
            #   而掌心跟随 move_only 又每帧把光标写回掌心映射点——两个写入源交替=闪动。
            #   锁定期本帧一切掌心光标写入全部跳过，光标只允许窗口中心锁这一个写入源。
            #   注：此处读的是【上一帧】move_state（stretch.update 在后面 L1708 才跑），
            #   进入/退出 MOVE_ACTIVE 各有 1 帧交接，帧内同点 SetCursorPos 幂等、肉眼不可见。
            _cursor_locked_by_window_move = bool(
                stretch_info is not None and stretch_info.move_state == _DMS.MOVE_ACTIVE
            )

            # ======================================================================
            # ⭐ B-1 优化：鼠标移动急活【最前置】
            #   目标：彻底消除"慢活（stretch.update/Tap 累积/每手循环 near_fist 计数/拉伸窗口确认）阻塞 move_only → 鼠标钉住 + 之后大跳"。
            #   位置：全局门控算完（any_fist_global 已可用）→ 任何慢活（Pass C 每手局部处理、Tap、stretch.update、绘制）之前。
            #   等价语义：和 Pass C 右手分支完全一致（cursor_anchor_norm + gesture_hint=FIST/gesture），只是更早；
            #             通过 _eval_anchor_once 的同帧缓存，即使后面原代码再 move_only 一次，也只是 pyautogui.moveTo(同坐标) 幂等，不会额外重算平滑。
            # ======================================================================
            _pre_move_only_done_right = False
            if _cursor_locked_by_window_move:
                # 移窗锁定期：本帧掌心 move_only 禁止执行；同时置位"已处理"，
                # 阻止 Pass C 兜底再调一次 moveTo
                _pre_move_only_done_right = True
            elif "Left" in hand_results:
                try:
                    _gesture, _ext, _ratios, _lms = hand_results["Left"]
                    _cursor_anchor_norm = HandDetector.cursor_anchor_from_landmarks(_lms, args.anchor)
                    _move_hint = "FIST" if any_fist_global else _gesture
                    mouse.move_only(_cursor_anchor_norm, gesture_hint=_move_hint)
                    _pre_move_only_done_right = True
                except Exception:
                    _pre_move_only_done_right = False

            # (B2) 双手/纯双手判定（每只手局部用，值一样，放外面统一算）
            both_hands_now = both_hands_global
            is_pure_two_hand_mode = pure_two_hand_stretch

            # ======================================================================
            # Pass C：对每只手做 鼠标映射 + 画「光标锚点 绿/橙圆」
            #          （因为门控都已算好了，这一段完全顺序正确，没有未定义变量）
            # ======================================================================
            stretch_click_gestures = ("PEACE", "PINCH", "ROCK", "THUMB")
            for label, (gesture, ext, _ratios, lms) in hand_results.items():
                # --- 这只手自己的 safe_gesture 判定（与原逻辑一致）---
                disable_click_for_stretch = (
                    both_hands_now
                    or stretch.state != _DS.IDLE
                    or is_pure_two_hand_mode
                )
                safe_gesture = gesture
                if disable_click_for_stretch and gesture in stretch_click_gestures:
                    safe_gesture = "POINT"
                bent_count_this = sum(1 for x in ext if not x)
                near_fist_this_hand = (bent_count_this >= 4)
                if near_fist_this_hand and gesture in stretch_click_gestures:
                    safe_gesture = "POINT"

                # --- 鼠标事件：⭐ 2026-08-22 严格『先移动、后判断点击』---
                #   为什么之前不对齐（无规律卡顿）：
                #     原 if (not block_mouse_move_for_stretch) and ... 把『移动』和『点击』两个语义绑在一起判断：
                #       只要手掌翻转时被 MediaPipe 偶判成 OPEN（single_open_global=True）/ 出现双手 / 拉伸状态非 IDLE
                #       → block_mouse_move_for_stretch=True → 整个分支跳过（不调 mouse.handle）
                #       → 但绿圈仍用最新 cursor_anchor_norm 画在画面上 → 视觉脱节：绿圈动了，鼠标钉死
                #   修复原则（用户说：手势识别不影响鼠标移动——先看鼠标位置，再判断手势指令）：
                #     移动：只要 label=="Left"（即这帧有右手掌心），就无条件先调 mouse.move_only() → 和绿圈 1:1 对齐
                #           例外：全局有 FIST → 握拳=取消指令，不动；FIST 之外全部强制跟绿圈同步
                #     点击：继续保留原有的 block_mouse_move_for_stretch / stretch.state 门控（双手/拉伸/OPEN 等只禁止点击，不禁止移动）
                if label == "Left":
                    # ↓↓ 先取锚点（画绿圈 + 移动鼠标用的**同一份**坐标，绝对保证对齐）↓↓
                    cursor_anchor_norm = HandDetector.cursor_anchor_from_landmarks(lms, args.anchor)
                    # move_only 例外判 FIST（全局 FIST 任何手都取消移动）
                    move_gesture_hint = "FIST" if any_fist_global else gesture
                    # ⭐ B-1 优化：若【全局门控后 move_only 最前置】已经跑过 → 这里不再重复 moveTo（同帧缓存已保证 smoother 状态只更新一次）
                    #            只有前置段异常（try 失败）时才兜底再跑一次，保证向后兼容。
                    if (not _pre_move_only_done_right) and (not _cursor_locked_by_window_move):
                        mouse.move_only(cursor_anchor_norm, gesture_hint=move_gesture_hint)
                    # 门控只拦"点击/拖拽语义"（调 handle 走点击分支）——但该帧鼠标已经 move_only 移动过了，不会再卡
                    # ⭐ 移窗锁定帧额外跳过 handle：OPEN 稳定分支内部也会 moveTo，否则与窗口中心锁打架闪动
                    if ((not _cursor_locked_by_window_move)
                            and (not block_mouse_move_for_stretch)
                            and (not is_pure_two_hand_mode)
                            and stretch.state != _DS.SCALING):
                        fingertip_norm = (lms[8].x, lms[8].y)
                        mouse.handle(safe_gesture, fingertip_norm, cursor_anchor_norm=cursor_anchor_norm)
                    # ↓↓ 画锚点（绿圈=掌心，橙圈=食指TIP）——和 move_only 用的 cursor_anchor_norm 完全同一份 ↓↓
                    mx, my = HandDetector.landmark_to_pixel(lms[8], w, h)
                    cv2.circle(frame, (mx, my), 7, (0, 140, 255), 2)
                    ax = int(cursor_anchor_norm[0] * w)
                    ay = int(cursor_anchor_norm[1] * h)
                    cv2.circle(frame, (ax, ay), 12, (255, 255, 255), 2)
                    cv2.circle(frame, (ax, ay), 9, (0, 230, 80), -1)

            # ---------- 食指 Tap-Click（仅右手） ----------
            index_bend: float | None = None
            two_finger_bend: float | None = None  # 双指右键：双指单元合成伸直度（四关节中点，0.20弯~0.88直）
            two_finger_together: bool | None = None  # 双指右键：食/中四排关节是否都并拢（None=本帧不可判定）
            two_finger_gap: float | None = None      # 双指右键：四排关节距 max（诊断/判定共用）
            two_finger_gap_parts: tuple | None = None  # (MCP,PIP,DIP,TIP) 四排指距比值，仅诊断日志用
            right_screen_pt = None
            if "Left" in hand_results:
                _, _, _, lms = hand_results["Left"]
                tip = np.array([lms[8].x, lms[8].y])
                m5  = np.array([lms[5].x, lms[5].y])
                m17 = np.array([lms[17].x, lms[17].y])
                palm_n_raw = float(np.linalg.norm(m5 - m17))
                tip_m5 = float(np.linalg.norm(tip - m5))

                # ======================================================================
                # ⭐ 2026-08-22 修复"点击经常误判"：palm_n 稳定性过滤（根因：dist(MCP5,MCP17) 随掌心翻转/远近剧烈波动 →
                #   即使食指没动，index_bend = dist(8,5)/palm_n 也会假"弯→直" → 被 TAP 状态机误判为点击）。
                # ⭐ 2026-08-25 统一原则3：分区跳过逻辑彻底删除。三档阈值全局放宽（7/10/8 → 15/20/16）适配顶部 clip 抖动：
                #     顶部 #5/#17 clip 导致 palm_n_raw 抖 8%~15% 是正常物理现象（非"掌心翻转假动作"），
                #     旧阈值 7% 把正常顶部帧全判为不稳定 → DI_down恒0=点不出来；放宽至 15%/20%/16% 后：
                #       正常顶部 clip 抖 12% → 通过；真实掌心翻转抖 25%+ → 仍能正确拦截。
                #   算法：
                #     1) EMA 平滑单帧 palm_n（抗瞬时尖峰）
                #     2) 最近 8 帧 palm_n 进 deque 滑窗，算 mean_palm（近期手掌大小基准）
                #     3) 不稳定判定（满足任一条 → 这帧 index_bend 视为噪声，传 None 给 tap_detector → 状态机不更新）：
                #        · 当前 palm_n_raw  vs  EMA(palm_n)   偏差 > 15%
                #        · 当前 palm_n_raw  vs  8 帧均值       偏差 > 20%
                #        · 8 帧滑窗标准差 / 滑窗均值 > 16%       （近期手掌大小一直在变化）
                # ======================================================================
                _palm_que: deque = getattr(run_loop, '_tap_palm_n_queue', None)
                _palm_ema: float | None = getattr(run_loop, '_tap_palm_n_ema', None)
                if _palm_que is None:
                    _palm_que = deque(maxlen=8)
                    run_loop._tap_palm_n_queue = _palm_que
                if palm_n_raw > 1e-6:
                    _palm_que.append(palm_n_raw)
                    if _palm_ema is None:
                        _palm_ema = palm_n_raw
                    else:
                        _palm_ema = 0.70 * palm_n_raw + 0.30 * _palm_ema
                    run_loop._tap_palm_n_ema = _palm_ema  # 保存到函数属性（静态）

                    palm_stable = True
                    q_mean = float(sum(_palm_que) / len(_palm_que))
                    # 偏差 1：当前 vs EMA
                    if abs(palm_n_raw - _palm_ema) / max(_palm_ema, 1e-6) > 0.15:
                        palm_stable = False
                    # 偏差 2：当前 vs 滑窗均值
                    elif abs(palm_n_raw - q_mean) / max(q_mean, 1e-6) > 0.20:
                        palm_stable = False
                    elif len(_palm_que) >= 4:
                        # 偏差 3：滑窗本身波动大（最近几帧手掌一直在动/翻）
                        arr = np.array(_palm_que, dtype=np.float64)
                        q_std = float(np.std(arr))
                        if q_std / max(q_mean, 1e-6) > 0.16:
                            palm_stable = False

                    # ⭐ 掌心归一化基准稳定 → 才算 index_bend（否则传 None → tap_detector 本帧完全不学习不流转）
                    if palm_stable:
                        # 额外修正：分母 palm_n 用 EMA 平滑后的，而不是 raw 单帧，进一步降低单帧抖动
                        denom = float(_palm_ema)
                        if denom > 1e-6:
                            # ------------------------------------------------------------------
                            # ⭐ 2026-08-24 根因修复：正面弯曲识别率极低 → 升级弯曲度量
                            #   旧度量 tip_m5/denom（dist(TIP_8-MCP_5)/palm_n）：
                            #     侧面视角：食指弯曲投影在画面上，tip_m5 剧烈缩短 → 识别 OK
                            #     正面视角：弯曲向镜头深度折叠，TIP_8 离 MCP_5 在 x,y 投影上几乎没动 →
                            #               tip_m5/denom ≈ 常数 → 差分和绝对门限都没信号 → 怎么弯都不触发。
                            #   新度量 _index_bend_from_joint_angles（v3 统一版）：
                            #     全局关节权 PIP0.4 + MCP0.2 + DIP0.4 匹配真实点击(DIP动)；
                            #     全局 95%angle+5%dist 位置无关；无分区切换；
                            #     正/侧/顶/底任何位置，只要真折了DIP就能识别。
                            # ------------------------------------------------------------------
                            dist_bend_old = tip_m5 / denom
                            index_bend = _index_bend_from_joint_angles(
                                lms, palm_denom=denom, dist_bend_fallback=dist_bend_old)
                            # ----------------------------------------------------------
                            # ⭐ 2026-09-17 食中双指并拢右键 v2：不再分别算食指/中指 bend
                            #   用户要求：前置条件已是并拢 → 食中两指对应关节(PIP/PIP, DIP/DIP, TIP/TIP)
                            #   应保持在容忍距离内；弯曲度只算"双指单元"一个合成值（四关节中点代入同公式）。
                            #   同步性由四排关节距离直接保证：一指先直 → TIP 排距离超阈 → together 失效 → 作废。
                            # ----------------------------------------------------------
                            try:
                                _p5  = np.array([lms[5].x,  lms[5].y],  dtype=np.float64)
                                _p9  = np.array([lms[9].x,  lms[9].y],  dtype=np.float64)
                                _p6  = np.array([lms[6].x,  lms[6].y],  dtype=np.float64)
                                _p10 = np.array([lms[10].x, lms[10].y], dtype=np.float64)
                                _p7  = np.array([lms[7].x,  lms[7].y],  dtype=np.float64)
                                _p11 = np.array([lms[11].x, lms[11].y], dtype=np.float64)
                                _p8  = np.array([lms[8].x,  lms[8].y],  dtype=np.float64)
                                _p12 = np.array([lms[12].x, lms[12].y], dtype=np.float64)

                                # 四排食-中指距（归一化到掌心宽）：MCP/PIP/DIP/TIP
                                _g_mcp = float(np.linalg.norm(_p5  - _p9))
                                _g_pip = float(np.linalg.norm(_p6  - _p10))
                                _g_dip = float(np.linalg.norm(_p7  - _p11))
                                _g_tip = float(np.linalg.norm(_p8  - _p12))

                                # ⭐ 并拢判据（2026-09-17 v7，分阶段阈值，全部 3D 米制 world 坐标）：
                                #   算 MCP/PIP/DIP/TIP 四排食-中指距 ÷ 同坐标系掌宽 dist(5,17)。
                                #   together 只表达【起势口径】：TIP ≤ TIP_ENTRY_R(0.80)。
                                #   · 起势必须 TIP 先并拢（IDLE 手指尚高、无遮挡，数据可信）；
                                #   · 深弯后 TIP 会因遮挡炸到 1.3+（真机实测），手势中是否保持并拢
                                #     由检测器自己改看 DIP ≤ DIP_HOLD_R(1.30)（parts 已传给它），
                                #     hands_tasks 不在这里按 TIP 拦深弯帧；
                                #   · 另外检测器有"先并拢后勾"顺序门（高 bend 连续 3 帧），
                                #     挡住单指点击"弯到底才碰巧并拢"导致的左键被抢杀。
                                _wlms = hand.get("world_landmarks")
                                _gap_space = "3D"
                                if _wlms is not None and len(_wlms) >= 18:
                                    _w = np.array([[float(p.x), float(p.y), float(p.z)]
                                                   for p in _wlms], dtype=np.float64)
                                    def _wd3(a: int, b: int) -> float:
                                        return float(np.linalg.norm(_w[a] - _w[b]))
                                    _wden = _wd3(5, 17) or 1e-6          # 米制掌宽（与 2D denom 同语义点对）
                                    _r_mcp  = _wd3(5, 9)  / _wden
                                    _r_pip  = _wd3(6, 10) / _wden
                                    _r_dip  = _wd3(7, 11) / _wden
                                    _r_tip  = _wd3(8, 12) / _wden
                                else:
                                    # 兜底：老检测器无 world 点时退回 2D（2D 下 MCP/PIP 最稳，用指根）
                                    _gap_space = "2D!"
                                    _r_mcp, _r_pip = _g_mcp / denom, _g_pip / denom
                                    _r_dip, _r_tip = _r_pip, _r_pip
                                _sync_spread = _r_tip - _r_pip
                                two_finger_gap = float(_r_tip)
                                two_finger_gap_parts = (round(_r_mcp, 3), round(_r_pip, 3),
                                                        round(_r_dip, 3), round(_r_tip, 3),
                                                        round(_sync_spread, 3))
                                # 起势口径只看 TIP；手势中 DIP 门控在检测器内部（TWO_FINGER_DIP_HOLD_R）
                                two_finger_together = bool(_r_tip <= TWO_FINGER_TIP_ENTRY_R)
                                run_loop._cached_two_finger_gap = round(float(_r_tip), 3)
                                run_loop._cached_two_finger_gap_space = _gap_space

                                # 双指单元合成 bend：四关节取中点，代入与单指完全相同的 _bend_from_points
                                _mcp_mid = (_p5  + _p9)  * 0.5
                                _pip_mid = (_p6  + _p10) * 0.5
                                _dip_mid = (_p7  + _p11) * 0.5
                                _tip_mid = (_p8  + _p12) * 0.5
                                _p0 = np.array([lms[0].x, lms[0].y], dtype=np.float64)
                                _dist_fb = float(np.linalg.norm(_tip_mid - _mcp_mid)) / denom
                                two_finger_bend = _bend_from_points(
                                    _p0, _mcp_mid, _pip_mid, _dip_mid, _tip_mid,
                                    palm_denom=denom, dist_bend_fallback=_dist_fb)
                            except Exception:
                                two_finger_bend = None
                                two_finger_together = None
                                two_finger_gap = None
                                two_finger_gap_parts = None
                    # ⭐ 统一后仍保留缓存供日志只读（IN_TB列显示位置，PIPΔ°列显示DIP/PIP角度变化量，不参与判断逻辑）
                    try:
                        _anx, _any = HandDetector.cursor_anchor_from_landmarks(lms, args.anchor)
                        _rx1, _ry1, _rx2, _ry2 = MouseController.compute_roi_bbox_px(
                            w, h, args.roi_scale, args.roi_offset_y, roi_safe_shrink=args.roi_safe_shrink)
                        _axp = float(_anx) * float(w)
                        _ayp = float(_any) * float(h)
                        _TOP = 0.25
                        _by_top = float(_ry1)
                        _by_bot = float(_ry1) + _TOP * max(1.0, float(_ry2 - _ry1))
                        _inx = (_axp >= float(_rx1) - 6.0) and (_axp <= float(_rx2) + 6.0)
                        _iny = (_ayp >= _by_top - 4.0) and (_ayp <= _by_bot + 4.0)
                        run_loop._cached_in_top_band = bool(_inx and _iny)
                    except Exception:
                        run_loop._cached_in_top_band = False
                    run_loop._cached_palm_stable_actual = bool(palm_stable)
                    _pip_ema_q: deque = getattr(run_loop, '_tap_pip_ema_que', None)
                    _dip_ema_q: deque = getattr(run_loop, '_tap_dip_ema_que', None)
                    if _pip_ema_q is None:
                        _pip_ema_q = deque(maxlen=5)
                        run_loop._tap_pip_ema_que = _pip_ema_q
                    if _dip_ema_q is None:
                        _dip_ema_q = deque(maxlen=5)
                        run_loop._tap_dip_ema_que = _dip_ema_q
                    try:
                        p5q, p6q, p7q = np.array([lms[5].x,lms[5].y]), np.array([lms[6].x,lms[6].y]), np.array([lms[7].x,lms[7].y])
                        pip_now = _three_point_angle(p5q, p6q, p7q)
                        _pip_ema_q.append(pip_now)
                        p6q2, p7q2, p8q2 = p6q, p7q, np.array([lms[8].x,lms[8].y])
                        dip_now = _three_point_angle(p6q2, p7q2, p8q2)
                        _dip_ema_q.append(dip_now)
                        if len(_pip_ema_q) >= 1:
                            pip_ema = float(sum(_pip_ema_q)/len(_pip_ema_q))
                            pip_chg = round(pip_now - pip_ema, 1)
                        else:
                            pip_chg = 0.0
                        if len(_dip_ema_q) >= 1:
                            dip_ema = float(sum(_dip_ema_q)/len(_dip_ema_q))
                            dip_chg = round(dip_now - dip_ema, 1)
                        else:
                            dip_chg = 0.0
                        run_loop._cached_pip_now = round(pip_now,1)
                        run_loop._cached_pip_chg = pip_chg
                        run_loop._cached_dip_now = round(dip_now,1)
                        run_loop._cached_dip_chg = dip_chg
                        del p5q,p6q,p7q,p6q2,p7q2,p8q2
                    except Exception:
                        run_loop._cached_pip_now = None
                        run_loop._cached_pip_chg = 0.0
                        run_loop._cached_dip_now = None
                        run_loop._cached_dip_chg = 0.0
                else:
                    # palm_n_raw 太小（手离镜头特别远/截断）→ 直接丢弃
                    index_bend = None
                    two_finger_bend = None
                    two_finger_together = None
                    two_finger_gap = None
                    two_finger_gap_parts = None

                # ⭐ 2026-08-22 核心修复：Tap-Click 段直接复用 move_only() 同帧计算出来的"缓存平滑后坐标"
                #   之前的问题：
                #     move_only() 调 _smooth → 更新 6 层缓冲/锁/EMA/方差
                #     handle() 又调 _smooth → 二次更新（同帧 anchor_norm 没变但 smoother 内部状态乱了——尤其是 ARSmoother.absolute 的方差锁误锁）
                #     这里 Tap-Click 段又调 _norm_to_screen + _smooth → 第三次！→ smooth 值严重偏离绿圈对应的位置
                #     → 冻结 freeze_for_click() 和 tap_event=1/2/3/4 时 click/long-press drag 都点到了错位的位置 → 用户感知"点不到 / 完全混乱"
                anchor_norm = HandDetector.cursor_anchor_from_landmarks(lms, args.anchor)
                # 走 _eval_anchor_once：如果 move_only() 已经在本帧跑过同 anchor_norm → 缓存命中，smoother 不再二次更新
                _, _, sx, sy, _, _ = mouse._eval_anchor_once(anchor_norm, gesture_hint=None)
                right_screen_pt = (float(sx), float(sy))

            if disable_click_for_stretch_global:
                tap_detector.reset()
                right_tap_detector.reset()
                tap_event = 0
                right_tap_event = 0
            else:
                # ⭐ 2026-09-17 双指并拢右键 v2：单 bend 输入（双指单元合成值）+ together 门控
                right_tap_event = right_tap_detector.update(two_finger_bend,
                                                            two_finger_together, two_finger_gap,
                                                            two_finger_gap_parts)
                if right_tap_detector.freeze_left:
                    # 双指手势占用期 + 右键触发后的冷却尾帧都冻结左指：
                    #   触发帧起双指回弹的 bend 值若漏喂左指，会被它当成独立食指勾按（虚构 Tap → 误双击）
                    if right_tap_detector.armed_this_frame:
                        tap_detector.cancel_in_progress()
                    tap_event = 0
                else:
                    tap_event = tap_detector.update(index_bend)

            # ======================================================================
            # ⭐⭐ 方案 A 取证日志（ROI 顶部 25% 条带）
            #   2026-08-25 致命修复：用户右手=MediaPipe label"Left"(镜像语义)，之前写"Right"=100%永不触发→无日志
            #   保证：
            #     · 每 60 帧打 1 行【心跳】= 无论条带是否命中，必打，证明日志代码执行中
            #     · 掌心进 TOP 25% 条带 → 每 2 帧打 1 行详细表 + 首次表头
            #     · 所有 print 强制 flush（Flask stdout 缓冲默认关闭 flushing）
            # ======================================================================
            try:
                import sys as _sys
                _log_cnt = getattr(run_loop, '_tap_log_frame_cnt', -1)
                _log_header_printed = getattr(run_loop, '_tap_log_header_printed', False)
                _log_cnt = 0 if _log_cnt < 0 else (_log_cnt + 1)
                setattr(run_loop, '_tap_log_frame_cnt', _log_cnt)
                # ---------- (1) 每 60 帧 1 行心跳日志（证明代码在执行，不会“完全无日志”）----------
                if _log_cnt % 60 == 0:
                    rx1, ry1, rx2, ry2 = MouseController.compute_roi_bbox_px(
                        w, h, args.roi_scale, args.roi_offset_y, roi_safe_shrink=args.roi_safe_shrink)
                    TOP_BAND_FACT = 0.25
                    band_y_top = float(ry1); band_y_bot = float(ry1) + TOP_BAND_FACT * max(1.0, float(ry2-ry1))
                    labels = sorted(hand_results.keys())
                    anchor_info = "no Right-Left label"
                    for _lbl in ("Left", "Right"):
                        if _lbl in hand_results:
                            _, _, _, _l = hand_results[_lbl]
                            nx, ny = HandDetector.cursor_anchor_from_landmarks(_l, args.anchor)
                            ay = float(ny) * float(h); ax = float(nx) * float(w)
                            in_band = (ay >= band_y_top - 2.0) and (ay <= band_y_bot + 2.0)
                            anchor_info = f"{_lbl}: anchor({ax:5.1f},{ay:5.1f}) inTopBand={'Y' if in_band else 'N'} | {anchor_info}"
                    # 用户要求删除 [HEARTBEAT-TAPLOG] 心跳兜底日志（2026-08-26）
                    # print(f"[HEARTBEAT-TAPLOG] frame={_log_cnt:>4} labels={labels} safe_rect_y=[{band_y_top:5.1f}~{band_y_bot:5.1f}~{ry2:5.1f}] | {anchor_info}",
                    #       file=_sys.stdout, flush=True)
                # (2) 详细逐帧 TAP 表格日志 │368 │ Y ... 已直接删除（2026-08-26，用户要求）
            except Exception as _tap_log_e:
                import traceback as _tb
                print(f"  [ERROR] top-band tap log 异常：{_tap_log_e!r}\n{_tb.format_exc()}", flush=True)

            if not disable_click_for_stretch_global and stretch.state != _DS.SCALING:
                # 1=单击(窗口超时补发) 2=长按按下 5=双击(快速原子) 6=慢节奏补救双击 7=双指右键：都要求鼠标钉在当前位置完成动作
                if tap_event in (1, 2, 5, 6) or right_tap_event == _TEV.RIGHT_TAP:
                    # freeze_for_click 冻结位置 = 同帧 move_only 输出位置（smoother 已被冻结，后续几帧都返回相同像素点）
                    mouse._smoother.freeze_for_click()
                if tap_event != 0:
                    # handle_finger_tap_event 内部 click / drag / mouseDown 用的就是对齐后的 right_screen_pt
                    mouse.handle_finger_tap_event(tap_event, right_screen_pt)
                if right_tap_event != 0:
                    # 双指并拢勾按回弹 → 右键单击
                    mouse.handle_finger_tap_event(right_tap_event, right_screen_pt)

            # ---------- 双手拉伸 / 移动 ----------
            left_px: tuple | None = None
            right_px: tuple | None = None
            left_palm_screen: tuple | None = None
            right_palm_screen: tuple | None = None
            left_lms_main = None
            right_lms_main = None
            if "Left" in hand_results:
                _, _, _, lms = hand_results["Left"]
                left_lms_main = lms
                nx, ny = HandDetector.cursor_anchor_from_landmarks(lms, "palm")
                left_px = (float(nx) * w, float(ny) * h)
                try:
                    sx, sy = mouse._norm_to_screen(float(nx), float(ny))
                    left_palm_screen = (float(sx), float(sy))
                except Exception:
                    left_palm_screen = None
            if "Right" in hand_results:
                _, _, _, lms = hand_results["Right"]
                right_lms_main = lms
                nx, ny = HandDetector.cursor_anchor_from_landmarks(lms, "palm")
                right_px = (float(nx) * w, float(ny) * h)
                try:
                    sx, sy = mouse._norm_to_screen(float(nx), float(ny))
                    right_palm_screen = (float(sx), float(sy))
                except Exception:
                    right_palm_screen = None
            stretch_info = stretch.update(left_px, right_px, (w, h),
                                          left_lms_main, right_lms_main,
                                          left_palm_screen, right_palm_screen)

            _t2 = time.perf_counter()

            # ---------- 绘制：Tap 条 ----------
            tm = tap_detector.metrics()
            # 回滚 B-2：每帧必画，不再跳帧（避免闪烁）
            _draw_tap_bar(frame, tm, tap_detector)

            # ---------- 绘制：双指并拢右键手势状态（仅在手势进行/触发瞬间显示，平时不占画面）----------
            try:
                _rmt = right_tap_detector.metrics()
                if right_tap_event == _TEV.RIGHT_TAP:
                    setattr(run_loop, '_right_flash_frames', 15)
                _flash = int(getattr(run_loop, '_right_flash_frames', 0))
                if _flash > 0:
                    setattr(run_loop, '_right_flash_frames', _flash - 1)
                if _rmt["active"] or _flash > 0:
                    _rstate = _rmt["state"]
                    if _flash > 0:
                        _rtxt, _rcol = "✅ 右键单击!", (60, 60, 255)
                    elif _rstate in (_TS.BEND_ARMED, _TS.BEND_HOLD):
                        _rtxt, _rcol = "双指并拢勾按中…(右键)", (0, 160, 255)
                    elif _rstate == _TS.PRESSED:
                        _rtxt, _rcol = "双指长按(不触发)", (0, 200, 255)
                    else:
                        _rtxt, _rcol = "双指右键冷却…", (150, 150, 150)
                    put_text_cn(frame, _rtxt, (w - 320, 150),
                                font_scale=0.55, color=_rcol, thickness=2)
            except Exception:
                pass

            # ---------- 绘制：SCALE 拉伸 UI ----------
            mode_now = getattr(stretch_info, 'mode', 'SCALE')
            if mode_now == _DM.SCALE and (stretch_info.need_frame_redraw or stretch_info.ui_pinch_active):
                p1, p2 = stretch_info.p1, stretch_info.p2
                if p1 is not None and p2 is not None:
                    x1, y1 = int(p1[0]), int(p1[1])
                    x2, y2 = int(p2[0]), int(p2[1])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    import math as _mt_tmp
                    dx_abs = abs(x2 - x1); dy_abs = abs(y2 - y1)
                    theta_deg = abs(_mt_tmp.degrees(_mt_tmp.atan2(float(dy_abs), float(max(dx_abs, 1)))))
                    if theta_deg <= 20.0:
                        pre_mode = "水平(拉宽/拉窄)"; pre_c = (40, 180, 255)
                    elif theta_deg >= 70.0:
                        pre_mode = "垂直(拉高/拉低)"; pre_c = (80, 255, 150)
                    else:
                        pre_mode = "对角(等比缩放)"; pre_c = (220, 120, 255)
                    def _draw_pinch_dot(lms, px, py, color):
                        if lms is None: return
                        t4 = (int(float(lms[4].x) * w), int(float(lms[4].y) * h))
                        t8 = (int(float(lms[8].x) * w), int(float(lms[8].y) * h))
                        cv2.line(frame, t4, t8, color, 2)
                        cv2.circle(frame, t4, 7, color, -1)
                        cv2.circle(frame, t8, 7, color, -1)
                        # 回滚 B-2：每帧必画"捏 XX%"文字，避免闪烁
                        put_text_cn(frame,
                                    f"捏: {min(99, int(stretch_info.left_pinch_r * 100))}%" if px < w // 2 else f"捏: {min(99, int(stretch_info.right_pinch_r * 100))}%",
                                    (max(4, px - 32), max(4, py + 32)), font_scale=0.5, color=color, thickness=2)
                    if (stretch_info.state == _DS.ARMED_CONFIRM) or (stretch_info.ui_pinch_active and stretch_info.state == _DS.IDLE):
                        color = (255, 210, 80)
                        thick = 2
                        cv2.circle(frame, (x1, y1), 22, color, 3)
                        cv2.circle(frame, (x2, y2), 22, color, 3)
                        cv2.line(frame, (x1, y1), (x2, y2), color, 2)
                        _draw_pinch_dot(left_lms_main,  x1, y1, color)
                        _draw_pinch_dot(right_lms_main, x2, y2, color)
                        hold_n = stretch.min_hold_frames
                        pct = stretch_info.confirm_frames / max(1, hold_n)
                        pct = 1.0 if pct > 1.0 else pct
                        prog_w = 340
                        bx = cx - prog_w // 2
                        by = cy - 62
                        # 回滚 B-2：黑色大底 + 中文 每帧必画，完全不跳帧
                        cv2.rectangle(frame, (bx - 6, by - 24), (bx + prog_w + 6, by + 48), (0, 0, 0), -1)
                        cv2.rectangle(frame, (bx, by), (bx + prog_w, by + 12), (60, 60, 60), -1)
                        if stretch_info.state == _DS.IDLE and stretch_info.ui_pinch_active:
                            fill_ratio = min(1.0, theta_deg / 90.0)
                            cv2.rectangle(frame, (bx, by), (bx + int(prog_w * fill_ratio), by + 12), pre_c, -1)
                            # 回滚 B-2：中文每帧必画
                            put_text_cn(frame,
                                        f"预判定【{pre_mode}】  θ={theta_deg:.0f}° ｜ 双手≥120px 并保持3帧即可激活",
                                        (bx, by + 20), font_scale=0.58, color=(255, 255, 255), thickness=2)
                        else:
                            cv2.rectangle(frame, (bx, by), (bx + int(prog_w * pct), by + 12), color, -1)
                            # 回滚 B-2：中文每帧必画
                            put_text_cn(frame,
                                        f"选中确认中… {stretch_info.confirm_frames}/{hold_n}({int(pct*100)}%)  θ={theta_deg:.0f}° → {pre_mode}",
                                        (bx, by + 20), font_scale=0.6, color=(255, 255, 255), thickness=2)
                    elif stretch_info.state == _DS.SCALING:
                        color = (80, 230, 255)
                        cv2.circle(frame, (x1, y1), 22, color, 4)
                        cv2.circle(frame, (x2, y2), 22, color, 4)
                        cv2.line(frame, (x1, y1), (x2, y2), color, 3)
                        _draw_pinch_dot(left_lms_main,  x1, y1, color)
                        _draw_pinch_dot(right_lms_main, x2, y2, color)
                        ratio = stretch_info.ratio
                        mode = getattr(stretch_info, 'stretch_mode', 'DIAGONAL')
                        if mode == "HORIZONTAL":
                            mode_title = "🤏水平拉伸"
                            sign = "拉宽" if ratio > 1.01 else ("拉窄" if ratio < 0.99 else "保持宽度")
                        elif mode == "VERTICAL":
                            mode_title = "🤏垂直拉伸"
                            sign = "拉高" if ratio > 1.01 else ("拉低" if ratio < 0.99 else "保持高度")
                        else:
                            mode_title = "🤏对角拉伸"
                            sign = "放大" if ratio > 1.01 else ("缩小" if ratio < 0.99 else "保持")
                        ratio_c = (60, 255, 90) if ratio > 1.01 else ((60, 180, 255) if ratio < 0.99 else (200, 200, 200))
                        lines = [
                            (f"{mode_title} → {sign}", (255, 255, 255), 0.75),
                            (f"X {ratio:.3f}  比例", ratio_c, 0.95),
                            (f"窗口: HWND 0x{int(stretch_info.hwnd):08X}   {stretch_info.win_w}×{stretch_info.win_h}",
                             (230, 230, 230), 0.55),
                        ]
                        card_w, card_h = 520, 100
                        bx = cx - card_w // 2
                        by = cy - card_h // 2 - 16
                        # 回滚 B-2：卡片黑底 + 边框 + 3 行中文 每帧必画（卡片外框颜色每帧画）
                        cv2.rectangle(frame, (bx, by), (bx + card_w, by + card_h), (0, 0, 0), -1)
                        cv2.rectangle(frame, (bx, by), (bx + card_w, by + card_h), color, 2)
                        put_text_cn(frame, lines[0][0], (bx + 24, by + 14), font_scale=lines[0][2], color=lines[0][1], thickness=2)
                        put_text_cn(frame, lines[1][0], (bx + 24, by + 44), font_scale=lines[1][2], color=lines[1][1], thickness=3)
                        put_text_cn(frame, lines[2][0], (bx + 24, by + 82), font_scale=lines[2][2], color=lines[2][1], thickness=2)

            # ---------- 绘制：MOVE 移动 UI ----------
            if mode_now == _DM.MOVE and (stretch_info.move_need_frame_redraw or stretch_info.ui_move_active):
                hands_mode_now = getattr(stretch_info, 'hands_mode', 'DUAL')
                p1_m, p2_m = None, None
                llms_ui = left_lms_main
                rlms_ui = right_lms_main
                if "Left" in hand_results:
                    _, _, _, llms_ui = hand_results["Left"]
                    try:
                        nlx, nly = HandDetector.cursor_anchor_from_landmarks(llms_ui, "palm")
                        p1_m = (float(nlx) * w, float(nly) * h)
                    except Exception:
                        p1_m = stretch_info.p1
                if "Right" in hand_results:
                    _, _, _, rlms_ui = hand_results["Right"]
                    try:
                        nrx, nry = HandDetector.cursor_anchor_from_landmarks(rlms_ui, "palm")
                        p2_m = (float(nrx) * w, float(nry) * h)
                    except Exception:
                        p2_m = stretch_info.p2
                if p1_m is None: p1_m = stretch_info.p1
                if p2_m is None: p2_m = stretch_info.p2

                mv_color = (90, 255, 120)
                def _draw_open_palm_marks(lms, px, py):
                    if lms is None: return
                    for tip_i in (4, 8, 12, 16, 20):
                        try:
                            tx = int(float(lms[tip_i].x) * w)
                            ty = int(float(lms[tip_i].y) * h)
                            cv2.circle(frame, (tx, ty), 8, mv_color, -1)
                        except Exception:
                            pass
                    # 回滚 B-2：掌心"张开"文字 每帧必画
                    put_text_cn(frame, "张开",
                                (max(4, px - 26), max(4, py + 38)), font_scale=0.55, color=mv_color, thickness=2)

                if hands_mode_now != "SINGLE" and p1_m is not None and p2_m is not None:
                    x1, y1 = int(p1_m[0]), int(p1_m[1])
                    x2, y2 = int(p2_m[0]), int(p2_m[1])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    cv2.circle(frame, (x1, y1), 24, mv_color, 3)
                    cv2.circle(frame, (x2, y2), 24, mv_color, 3)
                    cv2.line(frame, (x1, y1), (x2, y2), mv_color, 2)
                    _draw_open_palm_marks(llms_ui,  x1, y1)
                    _draw_open_palm_marks(rlms_ui, x2, y2)
                    if stretch_info.move_state == _DMS.MOVE_CONFIRM or (stretch_info.ui_move_active and stretch_info.move_state == _DMS.IDLE):
                        hold_n = 3
                        pct = stretch_info.move_confirm_frames / hold_n
                        pct = 1.0 if pct > 1.0 else pct
                        prog_w = 400
                        bx = cx - prog_w // 2
                        by = cy - 66
                        # 回滚 B-2：黑色大底 + 中文 每帧必画
                        cv2.rectangle(frame, (bx - 6, by - 30), (bx + prog_w + 6, by + 60), (0, 0, 0), -1)
                        cv2.rectangle(frame, (bx, by), (bx + prog_w, by + 14), (50, 50, 50), -1)
                        if stretch_info.ui_move_active and stretch_info.move_state == _DMS.IDLE:
                            cv2.rectangle(frame, (bx, by), (bx + int(prog_w * 0.02), by + 14), mv_color, -1)
                            # 回滚 B-2：中文每帧必画
                            put_text_cn(frame,
                                        f"✋预判定【移动窗口(双手)】左手掌心指向要移动的窗口 ｜ 保持五指张开3帧即可选中",
                                        (bx, by + 24), font_scale=0.55, color=(255, 255, 255), thickness=2)
                        else:
                            cv2.rectangle(frame, (bx, by), (bx + int(prog_w * pct), by + 14), mv_color, -1)
                            # 回滚 B-2：中文每帧必画
                            put_text_cn(frame,
                                        f"✋选中窗口确认中… {stretch_info.move_confirm_frames}/{hold_n}({int(pct*100)}%)  → 左手掌心下窗口=目标",
                                        (bx, by + 24), font_scale=0.58, color=(255, 255, 255), thickness=2)
                    elif stretch_info.move_state == _DMS.MOVE_ACTIVE:
                        dxv, dyv = stretch_info.move_delta_xy
                        dx_int, dy_int = int(round(float(dxv))), int(round(float(dyv)))
                        rect = stretch_info.move_window_rect
                        if rect is not None and len(rect) == 4:
                            ww = int(rect[2] - rect[0]); wh = int(rect[3] - rect[1])
                        else:
                            ww = 0; wh = 0
                        lines = [
                            (f"✋五指张开(双手) → 移动窗口中 Δx={dx_int:+d}  Δy={dy_int:+d}", (255, 255, 255), 0.72),
                            (f"方向：{'右' if dx_int > 4 else ('左' if dx_int < -4 else '—')}  {'下' if dy_int > 4 else ('上' if dy_int < -4 else '—')}  ｜ 光标已锁定窗口中心",
                             mv_color, 0.55),
                            (f"窗口: HWND 0x{int(stretch_info.move_hwnd):08X}  {ww}×{wh}",
                             (230, 230, 230), 0.52),
                        ]
                        card_w, card_h = 620, 110
                        bx = cx - card_w // 2
                        by = cy - card_h // 2 - 20
                        # 回滚 B-2：卡片黑底 + 边框 + 3 行中文 每帧必画
                        cv2.rectangle(frame, (bx, by), (bx + card_w, by + card_h), (0, 0, 0), -1)
                        cv2.rectangle(frame, (bx, by), (bx + card_w, by + card_h), mv_color, 2)
                        put_text_cn(frame, lines[0][0], (bx + 22, by + 14), font_scale=lines[0][2], color=lines[0][1], thickness=2)
                        put_text_cn(frame, lines[1][0], (bx + 22, by + 46), font_scale=lines[1][2], color=lines[1][1], thickness=2)
                        put_text_cn(frame, lines[2][0], (bx + 22, by + 86), font_scale=lines[2][2], color=lines[2][1], thickness=2)

                if hands_mode_now == "SINGLE" and p1_m is not None:
                    x1, y1 = int(p1_m[0]), int(p1_m[1])
                    cx = x1
                    cy = y1
                    cv2.circle(frame, (x1, y1), 26, mv_color, 3)
                    _draw_open_palm_marks(llms_ui if ("Left" in hand_results) else rlms_ui, x1, y1)
                    if stretch_info.move_state == _DMS.MOVE_CONFIRM or (stretch_info.ui_move_active and stretch_info.move_state == _DMS.IDLE):
                        hold_n = 3
                        pct = stretch_info.move_confirm_frames / hold_n
                        pct = 1.0 if pct > 1.0 else pct
                        prog_w = 420
                        bx = x1 + 40 if cx < w // 2 else (x1 - 40 - prog_w)
                        by = y1 - 66
                        # 回滚 B-2：黑色大底 + 中文 每帧必画
                        cv2.rectangle(frame, (bx - 6, by - 30), (bx + prog_w + 6, by + 60), (0, 0, 0), -1)
                        cv2.rectangle(frame, (bx, by), (bx + prog_w, by + 14), (50, 50, 50), -1)
                        if stretch_info.ui_move_active and stretch_info.move_state == _DMS.IDLE:
                            cv2.rectangle(frame, (bx, by), (bx + int(prog_w * 0.02), by + 14), mv_color, -1)
                            # 回滚 B-2：中文每帧必画
                            put_text_cn(frame,
                                        f"✋预判定【移动窗口(单手)】掌心指向要移动的窗口 ｜ 保持五指张开3帧即可选中",
                                        (bx, by + 24), font_scale=0.55, color=(255, 255, 255), thickness=2)
                        else:
                            cv2.rectangle(frame, (bx, by), (bx + int(prog_w * pct), by + 14), mv_color, -1)
                            # 回滚 B-2：中文每帧必画
                            put_text_cn(frame,
                                        f"✋选中窗口确认中(单手)… {stretch_info.move_confirm_frames}/{hold_n}({int(pct*100)}%)",
                                        (bx, by + 24), font_scale=0.58, color=(255, 255, 255), thickness=2)
                    elif stretch_info.move_state == _DMS.MOVE_ACTIVE:
                        dxv, dyv = stretch_info.move_delta_xy
                        dx_int, dy_int = int(round(float(dxv))), int(round(float(dyv)))
                        rect = stretch_info.move_window_rect
                        if rect is not None and len(rect) == 4:
                            ww = int(rect[2] - rect[0]); wh = int(rect[3] - rect[1])
                        else:
                            ww = 0; wh = 0
                        lines = [
                            (f"✋五指张开(单手) → 移动窗口中 Δx={dx_int:+d}  Δy={dy_int:+d}", (255, 255, 255), 0.72),
                            (f"方向：{'右' if dx_int > 4 else ('左' if dx_int < -4 else '—')}  {'下' if dy_int > 4 else ('上' if dy_int < -4 else '—')}  ｜ 光标已锁定窗口中心",
                             mv_color, 0.55),
                            (f"窗口: HWND 0x{int(stretch_info.move_hwnd):08X}  {ww}×{wh}",
                             (230, 230, 230), 0.52),
                        ]
                        card_w, card_h = 620, 110
                        bx = x1 + 40 if cx < w // 2 else (x1 - 40 - card_w)
                        by = y1 - card_h // 2
                        # 回滚 B-2：卡片黑底 + 边框 + 3 行中文 每帧必画
                        cv2.rectangle(frame, (bx, by), (bx + card_w, by + card_h), (0, 0, 0), -1)
                        cv2.rectangle(frame, (bx, by), (bx + card_w, by + card_h), mv_color, 2)
                        put_text_cn(frame, lines[0][0], (bx + 22, by + 14), font_scale=lines[0][2], color=lines[0][1], thickness=2)
                        put_text_cn(frame, lines[1][0], (bx + 22, by + 46), font_scale=lines[1][2], color=lines[1][1], thickness=2)
                        put_text_cn(frame, lines[2][0], (bx + 22, by + 86), font_scale=lines[2][2], color=lines[2][1], thickness=2)

            # ---------- 左上角：拉伸/移动状态提示 ----------
            sst = stretch_info.state
            msst = stretch_info.move_state
            hands_mode_now = getattr(stretch_info, 'hands_mode', 'DUAL')
            if msst == _DMS.MOVE_CONFIRM or (stretch_info.ui_move_active and msst == _DMS.IDLE):
                s_color = (90, 255, 120)
                if hands_mode_now == "SINGLE":
                    s_text = f"✋ 单手五指张开-移动窗口 确认中 {stretch_info.move_confirm_frames}/3"
                else:
                    s_text = f"✋ 双手五指张开-移动窗口 确认中 {stretch_info.move_confirm_frames}/3"
            elif msst == _DMS.MOVE_ACTIVE:
                s_color = (90, 255, 120)
                ddx, ddy = stretch_info.move_delta_xy
                if hands_mode_now == "SINGLE":
                    s_text = f"✋ 单手移动窗口中 Δx={int(round(float(ddx))):+d} Δy={int(round(float(ddy))):+d}  光标锁窗口中心"
                else:
                    s_text = f"✋ 移动窗口中 Δx={int(round(float(ddx))):+d} Δy={int(round(float(ddy))):+d}  光标锁窗口中心"
            elif sst == _DS.ARMED_CONFIRM:
                s_color = (255, 210, 80)
                s_text = f"🤏 双手捏合拉伸-选中确认 {stretch_info.confirm_frames}/{stretch.min_hold_frames}"
            elif sst == _DS.SCALING:
                s_color = (80, 230, 255)
                s_text = f"🤏 双手捏合拉伸-窗口缩放中 X {stretch_info.ratio:.3f}  {stretch_info.win_w}×{stretch_info.win_h}"
            elif stretch_info.ui_pinch_active and sst == _DS.IDLE:
                s_color = (255, 210, 80)
                s_text = f"🤏 双手捏合姿态识别到(非五指张开) → 双手拉开≥120px 并保持3帧激活缩放"
            elif stretch_info.ui_move_active and msst == _DMS.IDLE:
                s_color = (90, 255, 120)
                s_text = f"✋ 双手五指张开姿态识别到 → 左手掌心指向窗口 保持3帧激活移动"
            else:
                s_color = None
                s_text = ""
            if s_text:
                # 回滚 B-2：背景黑块 + 中文 每帧必画
                cv2.rectangle(frame, (8, 90), (8 + 470, 90 + 28), (0, 0, 0), -1)
                put_text_cn(frame, s_text, (14, 94), font_scale=0.6, color=s_color, thickness=2)

            # ---------- 组合手势提示 ----------
            current_combo = ""
            if "Left" in hand_results and "Right" in hand_results:
                lg = hand_results["Left"][0]
                rg = hand_results["Right"][0]
                for (lp, rp), desc in COMBOS:
                    if lg == lp and rg == rp:
                        current_combo = desc
                        break
            if current_combo:
                combo_hint = current_combo
                combo_hint_counter = 60
            elif combo_hint_counter > 0:
                combo_hint_counter -= 1
            if combo_hint_counter > 0:
                tw, th = _measure_cn_text_size(combo_hint, 0.8)
                tx = (w - tw) // 2
                ty = h - 30
                # 回滚 B-2：背景黑块 + 中文 每帧必画
                cv2.rectangle(frame, (tx - 12, ty - th - 14),
                              (tx + tw + 12, ty + 12),
                              (0, 0, 0), -1)
                put_text_cn(frame, combo_hint, (tx, ty),
                            font_scale=0.8, color=(0, 230, 230), thickness=2)

            # ---------- 顶部模式标签 ----------
            anchor_text = "掌心" if args.anchor == "palm" else "食指尖"
            if args.cursor_mode == "absolute":
                mode_text = "绝对指哪打哪"
            elif args.cursor_mode == "direct":
                mode_text = "⭐严格跟绿圈（翻转不卡）"
            else:
                mode_text = "AR 悬停平滑"
            roi_pct = int(round(args.roi_scale * 100))
            safe_pct = int(round(args.roi_safe_shrink * 100))
            # 回滚 B-2：顶部黑底 + 中文 每帧必画
            cv2.rectangle(frame, (0, 0), (w, 30), (0, 0, 0), -1)
            put_text_cn(frame, mode_label, (10, 6),
                        font_scale=0.6, color=mode_color, thickness=2)
            label_color = (255, 220, 0) if args.cursor_mode == "direct" else (0, 230, 80)
            put_text_cn(frame, f"🖱️{mode_text} | 锚点: {anchor_text} | 原始ROI: {roi_pct}% × 安全内缩: {safe_pct}%  🟢=光标  🟠=食指TIP",
                        (w - 780, 6), font_scale=0.6, color=label_color, thickness=2)
            # ---------- ROI 矩形（与 MouseController._norm_to_screen 100% 对齐：统一调 compute_roi_bbox_px）----------
            x1, y1, x2, y2 = MouseController.compute_roi_bbox_px(w, h, args.roi_scale, args.roi_offset_y,
                                                                 roi_safe_shrink=args.roi_safe_shrink)
            _dash_len = 12
            _gap = 8
            _step = _dash_len + _gap
            for x in range(x1, x2, _step):
                xe = min(x + _dash_len, x2)
                cv2.line(frame, (x, y1), (xe, y1), (0, 220, 120), 2)
            for x in range(x1, x2, _step):
                xe = min(x + _dash_len, x2)
                cv2.line(frame, (x, y2), (xe, y2), (0, 220, 120), 2)
            for y in range(y1, y2, _step):
                ye = min(y + _dash_len, y2)
                cv2.line(frame, (x1, y), (x1, ye), (0, 220, 120), 2)
            for y in range(y1, y2, _step):
                ye = min(y + _dash_len, y2)
                cv2.line(frame, (x2, y), (x2, ye), (0, 220, 120), 2)

            # ---------- 底部提示 ----------
            help_line = "右手=控鼠 | 🟢掌心移动=鼠标(食指点击不漂) | 食指轻弯=单击 | 保弯=拖 | 🤏V=单 | Rock=右 | 拇=双 | 拳=拖 | 🤏双手非张开=捏合拉伸→窗口放大缩小 | ✋单手五指张开=掌心下窗口拖动 | ✋双手五指张开=双手配合拖动窗口(光标锁窗口中心防误触) | q退"
            # 回滚 B-2：底部帮助栏 每帧必画
            put_text_cn(frame, help_line, (12, h - 28),
                        font_scale=0.55, color=(220, 220, 220), thickness=2)
            _t3 = time.perf_counter()

            # ---------- FPS ----------
            _fps_frame_cnt += 1
            _now = time.perf_counter()
            _dt = _now - _fps_last_t
            if _dt >= 0.5:
                _fps_inst = _fps_frame_cnt / max(_dt, 1e-6)
                if _fps_smooth is None:
                    _fps_smooth = _fps_inst
                else:
                    _fps_smooth = 0.25 * _fps_inst + 0.75 * _fps_smooth
                _fps_display = _fps_smooth
                _fps_cam_read_ms = (_t1 - _t0) * 1000.0
                _fps_infer_ms    = (_t2 - _t1) * 1000.0
                _fps_draw_ms     = (_t3 - _t2) * 1000.0
                _fps_frame_cnt = 0
                _fps_last_t = _now
            fps_txt = f"FPS {_fps_display:5.1f}  采集{_fps_cam_read_ms:5.1f}ms 识别{_fps_infer_ms:5.1f}ms 绘制{_fps_draw_ms:5.1f}ms"
            # 附加跳帧率提示（让用户知道优化在生效）
            if _skip_total_frames >= 10:
                _skip_pct = 100.0 * _skip_skipped_frames / max(_skip_total_frames, 1)
                if _skip_pct >= 15.0:
                    fps_txt += f"  跳帧{_skip_pct:.0f}%⚡"
            fps_color = (0, 230, 80) if _fps_display >= 25 else (80, 180, 255) if _fps_display >= 18 else (80, 80, 255)
            # 回滚 B-2：FPS 文字背景黑块 + 中文 每帧必画
            cv2.rectangle(frame, (w - 448, 30), (w - 4, 60), (0, 0, 0), -1)
            put_text_cn(frame, fps_txt, (w - 440, 36),
                        font_scale=0.55, color=fps_color, thickness=2)

            # ---------- 🔴 更新 status_store（纯标量，单写多读，GIL 安全）----------
            lg = hand_results.get("Left",  (None, None, None, None))[0]
            rg = hand_results.get("Right", (None, None, None, None))[0]
            if msst == _DMS.MOVE_CONFIRM:    smode = "MOVE_CONFIRM"
            elif msst == _DMS.MOVE_ACTIVE:  smode = "MOVE_ACTIVE"
            elif sst == _DS.ARMED_CONFIRM:  smode = "ARMED_CONFIRM"
            elif sst == _DS.SCALING:        smode = "SCALING"
            else:                           smode = "IDLE"
            _opt_skip_pct = float(100.0 * _skip_skipped_frames / max(_skip_total_frames, 1))
            status_store.update(dict(
                fps=float(_fps_display),
                uptime_ms=int(time.time() * 1000) - start_ms,
                last_left_gesture=lg,
                last_right_gesture=rg,
                stretch_mode=smode,
                preview_active=bool(preview),
                # 优化诊断字段
                opt_infer_size=480,
                opt_ema_alpha=float(_EMA_ALPHA),
                opt_mv_window=int(_MV_WINDOW),
                opt_skip_total=int(_skip_total_frames),
                opt_skip_skipped=int(_skip_skipped_frames),
                opt_skip_pct=float(_opt_skip_pct),
            ))

            # ---------- imshow / waitKey（仅 preview=True） ----------
            if preview:
                cv2.imshow("DualHand -> Mouse (双手体感鼠标)", frame)
                # waitKey(1) 必须在创建窗口的同一线程调用，所以这里直接调
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # ⚠️ 2026-08-26 核心修复（无预览=不能控制鼠标根因）：
                #   无任何 imshow / namedWindow 的情况下，Windows 版 OpenCV 的 cv2.waitKey(1)
                #   因为 HighGUI 消息泵没有窗口可挂，会进入「死等键盘事件」阻塞（不是1ms立即返回）
                #   → 主循环每帧卡死在这里，move_only() / handle_finger_tap_event() 虽然已经执行完，
                #     但下一轮循环永远到不了 → 用户体感「鼠标完全不动」。
                #   修复：纯后台模式直接用 time.sleep(0.001)（1ms 轻量级调度让步，纯时间，不进 HighGUI）。
                time.sleep(0.001)
                # 无预览模式额外心跳（~每 2 秒 1 次，与输入无关）：证明循环在跑，方便排查静默状态
                if (_log_cnt % 120 == 0) and (_log_cnt > 0):
                    print(f"[RUN-OK bg-mode] frame={_log_cnt:>5} FPS={_fps_display:5.1f} preview=OFF gesture_left={hand_results.get('Left',(None,))[0]} move_ok={right_screen_pt is not None} tap_event={tap_event}",
                          flush=True)

    # ================== finally：100% 与原 main.py 行为一致 ==================
    except Exception as e:
        import traceback
        msg = f"[hands_tasks.run_loop] 异常: {e}\n{traceback.format_exc()}"
        print(msg)
        status_store["error"] = str(e)
    finally:
        status_store["running"] = False
        try:
            mouse.reset()
        except Exception:
            pass
        try:
            detector.close()
        except Exception:
            pass
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        # 🔴 注意：destroyAllWindows 应该在调用了 imshow 的同一线程调（即这里 run_loop 本身）
        if preview:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
