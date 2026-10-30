"""
hands_tasks.py  —— 体感鼠标全部业务逻辑的纯封装（零复杂线程/状态，仅四件套函数）
================================================================================

四件套：
  1. parse_args()                → 解析命令行参数（与原 main.py 完全一致）
  2. clamp_args(args)            → 对 args 做硬约束范围裁剪（原地修改）
  3. init_models(args)           → 创建 detector/mouse/tap/right_tap 等全部组件
  4. run_loop(components, ...)   → 对应原 main() 里「打开摄像头 → 识别循环 → 清理」的全部逻辑

设计原则（极简）：
  · 没有任何后台线程、服务、锁 —— 调用者想放什么线程就放什么线程
  · 只有 run_loop 是阻塞的；stop_event.set() 会在下一帧开头立即干净退出
  · status_store 是普通 dict，调用者（server.py）随便读，不加任何锁（单写多读=Python GIL 安全）
"""

import argparse
import copy as _cp
import ctypes
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
import pyautogui

from hand_detector import HandDetector, TasksHandDetector, HAND_CONNECTIONS
from gesture_recognizer import recognize_gesture, GestureTimeStabilizer
from mouse_controller import MouseController
from draw_cn_text import put_text_cn
from finger_tap_click_detector import (FingerTapClickDetector, ThumbMiddleRightClickDetector,
                                       TwoFingerRightClickDetector, _S as _TS, E as _TEV,
                                       TWO_FINGER_TIP_ENTRY_R,
                                       RF_OBSERVE_ONLY, PINCH_ABS_MAX, PINCH_RELEASE_ABS)
from window_mover import (WindowMover, MOVE_CONFIRM_SEC, _MS as _WMS,
                          SetWindowPos, HWND_TOP, SWP_NOZORDER, SWP_NOACTIVATE,
                          WindowFromPoint, GetAncestor, GA_ROOT,
                          is_gameviewer_cursor_window)
from window_mover import RECT, GetForegroundWindow, GetWindowRect
from circle_scroll_detector import CircleScrollDetector

# ---------------- 常量（原 main.py 顶部，原样搬） ----------------
HAND_COLORS = {"Left": (255, 100, 0), "Right": (0, 140, 255)}
FINGER_NAMES_CN = ["拇", "食", "中", "无", "小"]
COMBOS = []  # 双手组合：用户要求全部置空

# ⭐ 2026-10-29 捏合 vs 卷指 鉴别门（v2 双特征 AND）：
#   真机教训：单用 PIP 角误伤真捏合（实测真捏合段 4/5 None 帧来自遮蔽——
#   真捏合食指为够到拇指尖 PIP 也会弯到 <135°，与"弯食指"区间重叠）。
#   改为双特征同时成立才遮蔽（任一不成立=放行）：
#     ① 食指 PIP 角 ∠(5,6,7) < _PINCH_CURL_PIP_DEG（关节明显折叠）
#     ② d85 = dist(TIP8,MCP5)/掌宽 < _PINCH_CURL_D85_MAX（指尖卷回食指根附近；
#        对捏时食指整体仍伸展、d85≈0.6+；卷指时指尖扫回掌侧、d85≈0.2~0.45）
#   触发遮蔽时 [CURL] 埋点打印实测 PIP/d85，供下一轮按真机数据精调阈值。
#   真机实测（[CURL] 埋点）：真捏合 PIP=116~128°、d85=0.46~0.47 → d85 门从
#   0.50 降到 0.40（放行真捏合留余量；深卷指 d85 预计 0.2~0.4 仍被拦）。
#   若"弯食指"回潮（假点击出现但 [CURL] 无打印），按当时实测再收。
_PINCH_CURL_PIP_DEG = 135.0
_PINCH_CURL_D85_MAX = 0.40


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


# ======================================================================
# ⭐ 同视角：左右手身份时序稳定器（2026-09-30 新增稳定性兜底）
# ======================================================================
# 为什么需要：
#   面对面视角手掌基本正对镜头，MediaPipe 的 Left/Right 分类稳定；同视角（头戴）
#   常见手背入镜，捏合/勾按时手掌还会绕视线轴旋转，此时 MP handedness 会在
#   Left/Right 之间【逐帧横跳】。而下游 landmarks-EMA、手势多数投票、双手全局门控、
#   双手拉伸全部以 label 为字典 key —— 一次横跳就等于"双手身份互换"：
#   平滑缓存串手、控鼠手瞬间切到另一只手、双手拉伸左右锚点对调。
# 策略（位置锁定为主、标签滞回为辅）：
#   1) 检测手与在册轨迹按掌心归一化距离做【全局最优一一匹配】（最多 2 手，枚举配法，
#      避免两只手最近邻贪心被同时带偏）；
#   2) 匹配上的手沿用轨迹锁定身份，本帧 MP 标签给反了直接忽略；
#   3) MP 连续 SWITCH_FRAMES 帧给相反标签才承认真换手（人转身/双手真实换位），
#      单帧/数帧横跳被滞回门挡住；
#   4) 双手位置重叠难分（两配法距离接近）时，给"标签一致"的配法小幅奖励 → 退回尊重 MP；
#   5) 未匹配检测=新轨迹（用 MP 标签初始化；与在册身份撞车时取另一标签，保证不重名）；
#      未匹配轨迹保留 MAX_MISS 帧可被找回，超期注销。
# 🔴 单一入口原则：只允许在 run_loop 的检测出口（标签交换之后、EMA/跳帧缓存之前）
#    调用一次，下游所有模块只消费稳定后的 label，禁止在别处自行翻转/猜身份。
class HandIdentityStabilizer:
    MATCH_TOL = 0.15        # 掌心归一化距离容差（短边720≈108px；30fps 正常移动邻帧位移远小于此）
    LABEL_BONUS = 0.05      # 配法代价中"MP标签与轨迹一致"的奖励；双手重叠难分时让位给 MP
    SWITCH_FRAMES = 8       # MP 连续 N 个【样本】给反标签才允许翻转身份（样本数条件）
    SWITCH_SEC = 0.27       # 且连续墙钟时长达到 0.27s（≈原 8 帧 @30fps；帧率变化时物理手感不变）
    MAX_MISS = 8            # 轨迹丢失保留【样本】数（与 landmarks-EMA 的 miss 清理阈值一致）
    MAX_MISS_SEC = 0.27     # 轨迹丢失保留【墙钟时长】（双条件同时满足才注销）

    def __init__(self):
        # 每条轨迹：label=锁定身份；x,y=最近掌心（归一化）；
        #          miss_n/miss_sec=连续丢失样本数/墙钟秒数（双条件到期才注销）
        #          disagree_n/disagree_sec=匹配在手但 MP 连续给反标签的样本数/墙钟秒数
        self._tracks: list[dict] = []

    def reset(self):
        self._tracks = []

    @staticmethod
    def _palm_of(hand: dict) -> tuple[float, float]:
        # ⭐ 2026-10-01 跟踪点与光标锚点 mcp 保持同一真相：四指根 MCP(5,9,13,17) 质心。
        #   旧 (腕0+中指根9)/2 在同视角手腕出画面时 #0 是模型猜的 → 身份轨迹跟着漂。
        lms = hand["landmarks"]
        return (0.25 * (float(lms[5].x) + float(lms[9].x)
                          + float(lms[13].x) + float(lms[17].x)),
                0.25 * (float(lms[5].y) + float(lms[9].y)
                          + float(lms[13].y) + float(lms[17].y)))

    def apply(self, hands_data: list[dict], dt: float = 0.0) -> None:
        """就地把 hands_data 中每只手的 label 改写为时序稳定后的身份。

        Args:
            dt: 距上一帧墙钟秒数（防误触用，与样本计数组成双条件）
        """
        # ---- 无手：全部轨迹累计丢失（样本+墙钟双条件到期注销） ----
        if not hands_data:
            for tr in self._tracks:
                tr["miss_n"] += 1
                tr["miss_sec"] += float(dt)
            self._tracks = [t for t in self._tracks
                            if t["miss_n"] <= self.MAX_MISS and t["miss_sec"] < self.MAX_MISS_SEC]
            return

        dets = [{"hand": h, "mp": h.get("label"),
                 "x": self._palm_of(h)[0], "y": self._palm_of(h)[1],
                 "matched": False} for h in hands_data]
        n_det, n_trk = len(dets), len(self._tracks)

        # ---- 枚举所有"检测↔轨迹"配对方式（d≤2, t≤2，手写穷举；允许不配=新轨迹）----
        candidates = [[]]
        if n_det >= 1:
            for ti in range(n_trk):
                candidates.append([(0, ti)])
        if n_det >= 2:
            for di in (0, 1):
                for ti in range(n_trk):
                    candidates.append([(di, ti)])
            if n_trk >= 2:
                candidates.append([(0, 0), (1, 1)])
                candidates.append([(0, 1), (1, 0)])

        # 选法：① 匹配对数最多优先；② 同对数下总代价最小
        #   代价 = Σ(掌心距离 − 标签一致奖励)，任一对距离超 MATCH_TOL 的配法作废
        best_key, best_pairs = None, []
        for pairs in candidates:
            used_di, used_ti = set(), set()
            cost, valid = 0.0, True
            for di, ti in pairs:
                if di in used_di or ti in used_ti:
                    valid = False
                    break
                used_di.add(di)
                used_ti.add(ti)
                tr = self._tracks[ti]
                dist = _mt.hypot(dets[di]["x"] - tr["x"], dets[di]["y"] - tr["y"])
                if dist > self.MATCH_TOL:
                    valid = False
                    break
                if dets[di]["mp"] == tr["label"]:
                    dist -= self.LABEL_BONUS
                cost += dist
            if not valid:
                continue
            key = (-len(pairs), cost)
            if best_key is None or key < best_key:
                best_key, best_pairs = key, pairs

        # ---- 应用匹配结果：锁定身份 + 标签滞回 ----
        matched_ti: set[int] = set()
        used_labels: set[str] = set()
        for di, ti in best_pairs:
            d, tr = dets[di], self._tracks[ti]
            d["matched"] = True
            matched_ti.add(ti)
            tr["x"], tr["y"], tr["miss_n"], tr["miss_sec"] = d["x"], d["y"], 0, 0.0
            if d["mp"] == tr["label"]:
                tr["disagree_n"] = 0
                tr["disagree_sec"] = 0.0
            elif len(dets) == 2:
                # 仅【双手同帧】才累计反标签账：真实换位只可能发生在双手同框时。
                # 单只手在镜时 MP handedness 无另一只手做参照，会逐帧 Left/Right
                # 横跳（单手帧 MP 标签无参照）——此时翻转身份没有
                # 任何对照意义，只会让下游 label 键整体跳变（控鼠手/和弦修饰手
                # 错位、和弦永久无法再武装）。单手帧反标签一律忽略：账不累计也
                # 不清零，位置匹配已把手与轨迹绑死。
                tr["disagree_n"] += 1
                tr["disagree_sec"] += float(dt)
                # ✨ 双条件：样本数 + 墙钟时长都达标，才承认双手真实换位
                if tr["disagree_n"] >= self.SWITCH_FRAMES and tr["disagree_sec"] >= self.SWITCH_SEC:
                    # MP 持续多帧/足够久给反标签 → 承认双手真实换位，翻转锁定身份
                    tr["label"] = "Right" if tr["label"] == "Left" else "Left"
                    tr["disagree_n"] = 0
                    tr["disagree_sec"] = 0.0
            d["final_label"] = tr["label"]
            used_labels.add(tr["label"])

        # ---- 未匹配轨迹：累计丢失（样本+墙钟）；未匹配检测：建立新轨迹 ----
        for ti, tr in enumerate(self._tracks):
            if ti not in matched_ti:
                tr["miss_n"] += 1
                tr["miss_sec"] += float(dt)
        self._tracks = [t for t in self._tracks
                          if t["miss_n"] <= self.MAX_MISS and t["miss_sec"] < self.MAX_MISS_SEC]

        unmatched = [d for d in dets if not d["matched"]]
        # 冷启动双手同框且 MP 给出【相同标签】：按同视角画面解剖先验裁决——
        # 非镜像画面中用户左手必在画面左侧（x 小 = Right），右手在右侧（x 大 = Left）。
        # 手臂交叉是瞬态且 MP 一般不会给同标签，下一帧起位置锁定+滞回会接管。
        if len(unmatched) >= 2 and len({d["mp"] for d in unmatched}) == 1:
            unmatched.sort(key=lambda d: d["x"])
            for d, lab in zip(unmatched, ("Right", "Left")):
                d["final_label"] = lab
                used_labels.add(lab)
                self._tracks.append(
                    {"label": lab, "x": d["x"], "y": d["y"],
                     "miss_n": 0, "miss_sec": 0.0, "disagree_n": 0, "disagree_sec": 0.0})
            unmatched = []
        for d in unmatched:
            # 新轨迹身份用【解剖位置先验】而非 MP 标签：非镜像头戴画面中
            # 用户左手恒在画面左半（x≤.5=Right）、右手在右半（x>.5=Left）。
            # 单手在镜（另一只手出画/刚双丢轨迹注销后归建）
            # 时 MP 标签大面积误标——物理右手在 x1.2 被连标 Left（交换后 Right），
            # 信 MP 归建会让右手占 Right 键：和弦修饰位被非拳占据→永久无法武装、
            # 控鼠手在标签横跳中误控鼠还误触点击。与上方冷启动双同标裁决同源。
            lab = "Right" if d["x"] <= 0.5 else "Left"
            if lab in used_labels:
                # 与在册身份撞车（单手快速移动未配上等）→ 取另一未占用身份，
                # 宁可瞬时归错也绝不重名（重名=下游字典键覆盖，灾难更大）；
                # 后续双手同帧位置匹配 + 重名裁决会自愈。
                other = "Right" if lab == "Left" else "Left"
                if other not in used_labels:
                    lab = other
            d["final_label"] = lab
            used_labels.add(lab)
            self._tracks.append(
                {"label": lab, "x": d["x"], "y": d["y"],
                 "miss_n": 0, "miss_sec": 0.0, "disagree_n": 0, "disagree_sec": 0.0})

        # ---- 双手标签唯一性保证 ----
        # 画圈转掌时 MediaPipe 可连续数十帧给两只手【相同】原始标签（多为双 Left，
        # 分数 0.6~0.98，与置信度阈值无关）→ 交换后双 Right。此时上方"标签滞回
        # 翻转"只翻一只：右手轨迹 Left→Right 与左手轨迹重名，两条轨迹同标签 →
        # 下游 hand_results/EMA/_gest_stab 全部以 label 为字典键，后者覆盖前者，
        # 画圈手(Left 键)整段消失 = 假 MISS（真机日志 f254~303：n=2 共38帧却
        # 全 MISS）。n=2 且最终标签不构成一 R 一 L 时，按非镜像解剖先验裁决：
        # 画面 x 小=用户左手=Right，x 大=用户右手=Left，并同步修正轨迹身份、
        # 清滞回账（手臂真实交叉为瞬态，下帧位置匹配重新接管，代价仅一帧）。
        if len(dets) == 2 and len({d.get("final_label") for d in dets}) != 2:
            by_x = sorted(dets, key=lambda z: z["x"])
            _want = ((by_x[0], "Right"), (by_x[1], "Left"))
            _det_to_track = {id(dets[di]): ti for di, ti in best_pairs}
            for _d, _lab in _want:
                _d["final_label"] = _lab
                _ti = _det_to_track.get(id(_d))
                if _ti is not None:
                    _tr = self._tracks[_ti]
                    _tr["label"] = _lab
                    _tr["disagree_n"] = 0
                    _tr["disagree_sec"] = 0.0

        # ---- 单一出口：改写回 hands_data，下游只消费这份身份 ----
        for d in dets:
            d["hand"]["label"] = d["final_label"]


# ======================================================================
# 握拳语义架构（拍板）：单手握拳【没有任何含义】
#   - 不取消点击、不冻结光标、不强制抬键、不产生任何动作；单手握拳在交互上完全中性。
#   - FIST 标签（gesture_recognizer）保留，仅作为【双手组合手势】的旁挂/修饰角色
#     （未来 FIST 修饰手 + POINT 画圈手滚轮和弦）；单手路径 FIST 按中性 OTHER 处理。
#   - 历史上的三代"单手握拳全局 guard"全部删除，不得恢复：
#       v3 index_curl 单指标：遮挡时 TIP 推断点贴掌心，100% 漏判/横跳；
#       v4 四指卷心比 r（对掌心/对掌弓质心）：点击瞬间食指 r(0.84~1.29) 与握拳
#          (0.76~1.10) 原理性重叠，暗光 TIP 抖动乱跳，真机点击段 100% 误杀；
#       v5 食指近节角 guard（27~51° vs 56~80°，阈53）：几何本身可分，但作为
#          【全局取消/冻结 + world 缺失宁拦】在遮挡场景仍是致命打击（点击被杀光、
#          光标钉死、长按被强抬）。
#   - 3D world 角度与弯度仅用于观测日志，不参与左键长按、点击或释放判定。
# ======================================================================


# 四指 (MCP, PIP, TIP)：食 5/6/8、中 9/10/12、无名 13/14/16、小 17/18/20
_FOLD_FINGER_TRIPLES = ((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20))


def _four_finger_mcp_angles(pts3d):
    """四指近节骨(MCP→PIP) 与掌轴(wrist0→MCP9) 的 3D world 夹角（度）。

    返回 (食,中,无名,小)；点异常返回 None。
    起唯一消费者 = 纯观测日志；不参与长按、点击或释放判定。
    只用 MCP/PIP/wrist/MCP9 等遮挡下仍稳定的点，不依赖 TIP。
    """
    try:
        u = pts3d[9] - pts3d[0]
        un = float(np.linalg.norm(u))
        if un < 1e-9:
            return None
        u = u / un
        out = []
        for m, pp, _t in _FOLD_FINGER_TRIPLES:
            v = pts3d[pp] - pts3d[m]
            vn = float(np.linalg.norm(v))
            if vn < 1e-9:
                return None
            cos_a = float(np.clip(np.dot(v / vn, u), -1.0, 1.0))
            out.append(float(np.degrees(np.arccos(cos_a))))
        return tuple(out)
    except Exception:
        return None


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


def _bend3d_probe(lms2d, wlms, palm_2d: float,
                  joint_ids=(5, 6, 7, 8)) -> dict | None:
    """⭐ 2026-10-04 [BEND3D] 纯观测探针：3D world 弯度 + 透视压缩标志。
    【判定链路绝不使用本函数任何输出】，仅供 [TAP-SEG]/[2F-SEG]/[BEND3D-JIT] 埋点。

    - bend：与 2D 完全同公式同映射(0.20弯~0.88直)，但关节向量取 world (x,y,z) 真实夹角，
      无 5% 距离兜底(palm 无关)、无掌宽 scale_fix；噪声门/Δ放大保留 → 两口径数值可直接比较。
    - dip/pip：3D 关节夹角(度，伸直≈180)，看朝镜头勾时 3D 是否照样折下去。
    - comp：画面投影 |DIP-TIP| / 2D 掌宽。侧对镜头伸直≈0.4~0.6；指尖朝镜头深度折叠时塌缩。
    - phal_mm：world 末节指骨长度 |DIP-TIP|（毫米），检验个人指骨长度帧间稳定性。
    - tip3：world TIP 坐标，供调用方算逐帧跳变（模型猜测点的时间连续性）。
    """
    try:
        _jm, _jp, _jd, _jt = joint_ids
        w0 = np.array([float(wlms[0].x),  float(wlms[0].y),  float(wlms[0].z)])
        wm = np.array([float(wlms[_jm].x), float(wlms[_jm].y), float(wlms[_jm].z)])
        wp = np.array([float(wlms[_jp].x), float(wlms[_jp].y), float(wlms[_jp].z)])
        wd = np.array([float(wlms[_jd].x), float(wlms[_jd].y), float(wlms[_jd].z)])
        wt = np.array([float(wlms[_jt].x), float(wlms[_jt].y), float(wlms[_jt].z)])
        bend = _bend_from_points(w0, wm, wp, wd, wt,
                                 palm_denom=None, dist_bend_fallback=None)
        dip_deg = _three_point_angle(wp, wd, wt)
        pip_deg = _three_point_angle(wm, wp, wd)
        comp = None
        if palm_2d is not None and palm_2d > 1e-6:
            _v = np.array([lms2d[_jd].x - lms2d[_jt].x,
                           lms2d[_jd].y - lms2d[_jt].y], dtype=np.float64)
            comp = float(np.linalg.norm(_v) / palm_2d)
        phal_mm = float(np.linalg.norm(wt - wd) * 1000.0)
        return {"bend": float(bend), "dip": float(dip_deg), "pip": float(pip_deg),
                "comp": comp, "phal_mm": phal_mm, "tip3": wt}
    except Exception:
        return None


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
        # ⭐ 2026-10-29 新手势：文案从"食指弯曲"改为"拇食捏合"（状态机不变，仅信号源换成捏合比）
        _TS.IDLE:       ("拇食张开待机(基线学习中)", (160, 255, 160)),
        _TS.BEND_ARMED: ("拇食捏合确认中…",      (80, 180, 255)),
        _TS.BEND_HOLD:  (f"捏合保持 {tm.get('hold_sec', 0):.2f}s",  (0, 180, 255)),
        _TS.TAPPED:     ("✅ TAP 单击触发!",     (60, 60, 255)),
        _TS.WAIT_DBL:   (f"等待第二次点击(双击)… {tm.get('dbl_window_left', 0):.2f}s", (255, 0, 255)),
        _TS.PRESSED:    ("🔒 按住拖拽中",          (0, 200, 255)),
        _TS.COOLDOWN:   (f"冷却 {tm.get('cooldown_left', 0):.2f}s",   (180, 180, 180)),
    }
    st_text, st_color = state_cn_map.get(st, (str(st), (255, 255, 255)))
    line2_y = bar_y0 + bar_h + 6
    put_text_cn(frame, st_text, (bar_x0, line2_y),
                font_scale=0.55, color=st_color, thickness=2)

    line3_y = line2_y + 22
    # ⭐ 墙钟化（2026-09-18）：进度/文案全部秒；stable 仍是连续稳定样本帧计数
    tap_max_s  = float(tm.get("TAP_MAX_SEC", 0.8))
    long_max_s = float(tm.get("LONG_PRESS_SEC", 0.733))
    tot_s      = float(tm.get("total_sec", 0))
    hold_s     = float(tm.get("hold_sec", 0))
    stab_f     = float(tm.get("stable_frames", 0))

    pb_w = int((bar_w - 10) / 2)
    pb_h = 10
    pb1_x = bar_x0
    p1 = 0.0 if tot_s <= 0 else min(1.2, tot_s / tap_max_s)
    cv2.rectangle(frame, (pb1_x, line3_y), (pb1_x + pb_w, line3_y + pb_h), (60, 60, 60), -1)
    tap_c = (30, 255, 90) if p1 < 0.7 else ((60, 220, 255) if p1 < 1.0 else (0, 0, 255))
    cv2.rectangle(frame, (pb1_x, line3_y), (pb1_x + int(pb_w * min(1.0, p1)), line3_y + pb_h), tap_c, -1)
    put_text_cn(frame, f"短按窗 {tot_s * 1000:>4.0f}/{tap_max_s * 1000:.0f}ms", (pb1_x, line3_y - 14),
                font_scale=0.32, color=tap_c, thickness=1)
    pb2_x = bar_x0 + pb_w + 10
    p2 = 0.0 if hold_s <= 0 else min(1.0, hold_s / long_max_s)
    cv2.rectangle(frame, (pb2_x, line3_y), (pb2_x + pb_w, line3_y + pb_h), (60, 60, 60), -1)
    long_c = (40, 220, 255) if p2 < 0.5 else ((0, 150, 255) if p2 < 1.0 else (0, 0, 255))
    cv2.rectangle(frame, (pb2_x, line3_y), (pb2_x + int(pb_w * p2), line3_y + pb_h), long_c, -1)
    put_text_cn(frame, f"长按 {hold_s * 1000:>4.0f}/{long_max_s * 1000:.0f}ms 稳{int(stab_f):>2d}f",
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
    parser.add_argument("--anchor", type=str, default="palm",
                        choices=["mcp", "palm", "index_tip"],
                        help="鼠标光标锚点：palm=掌心(默认,勾按/OK/移窗锚点统一避免漂移) / "
                             "mcp=四指根质心 / index_tip=食指TIP")
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
      {args, detector, mouse, tap_detector, right_tap_detector, window_mover,
       mode_label, mode_color}
    """
    detector_kwargs = dict(max_hands=args.max_hands,
                           # ⭐ 2026-09-21 黑帧根因治理：video 模式下 tracking 置信度低于门限时 MediaPipe
                           #   会【退回全帧掌检测器】。食指深折自遮挡掌心时 tracking 置信度在 0.5 附近抖动，
                           #   旧门限 0.55 → 频繁退回重检；掌检测器对非张开手掌命中率低 → 连串整手丢失
                           #   （实测长按 9.5s 内 70% 黑帧、最长连续 9.75s），重检成功的头几帧还吐过渡
                           #   垃圾关键点。下调 tracking 到 0.40 让时序跟踪器持续锁定（跟踪模式用上一帧
                           #   ROI，对指关节自遮挡天然鲁棒）；detection 0.55 仅在真丢手重检时使用。
                           # 右手画圈手检测分偏低（0.58~0.84）频繁 MISS：
                           #   det 0.55→0.45 让低分手也能重检；track 0.40→0.30 让时序
                           #   跟踪更持久。左手握拳分 0.99 不受影响。
                           detection_confidence=0.45,
                           tracking_confidence=0.30)
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
    # 拇指-中指 TIP 接近回弹 → 右键
    right_tap_detector = ThumbMiddleRightClickDetector(hardness=args.tap_hardness, speed=args.tap_speed)
    # 单手五指张开 → 拖移掌心下窗口（2026-10-14 从旧拉伸模块仅恢复 SINGLE MOVE）
    window_mover = WindowMover()
    # 双手和弦滚轮：左手握拳 + 右手食指画圈（单手路径完全不读它）
    circle_scroll = CircleScrollDetector()

    return dict(
        args=args,
        detector=detector,
        mouse=mouse,
        tap_detector=tap_detector,
        right_tap_detector=right_tap_detector,
        window_mover=window_mover,
        circle_scroll=circle_scroll,
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
      error              : str or None —— 异常信息（正常为 None）
      preview_active     : bool
    """
    args            = components["args"]
    detector        = components["detector"]
    mouse           = components["mouse"]
    tap_detector    = components["tap_detector"]
    right_tap_detector = components["right_tap_detector"]
    window_mover    = components["window_mover"]
    circle_scroll   = components["circle_scroll"]
    mode_label      = components["mode_label"]
    mode_color      = components["mode_color"]

    # 画圈滚轮和弦迟滞态每次启动清空（函数属性跨调用会残留，防止单手误成立）
    run_loop._chord_scroll_hold = False
    # OK 调整窗口大小状态每次启动清空
    run_loop._ok_resize_started = False
    run_loop._ok_sm_x = run_loop._ok_sm_y = 0.0       # 拖动 EMA 平滑点
    run_loop._ok_acc_x = run_loop._ok_acc_y = 0.0  # moveRel 亚像素累积
    run_loop._ok_miss_t0 = None   # 拖动中丢手宽限计时；None=手在镜
    # GameViewer 交互分支：光标所在窗口是否 gameviewer.exe（每帧重算）
    run_loop._is_gv = False
    run_loop._gv_press = False    # GV 下五指张开/OK 的左键长按态

    # ---------- 初始化 status_store ----------
    if status_store is None:
        status_store = {}
    start_ms = int(time.time() * 1000)
    status_store.clear()
    status_store.update(dict(
        running=False, fps=0.0, current_camera_id=camera_id,
        start_time_ms=start_ms, uptime_ms=0,
        last_left_gesture=None, last_right_gesture=None,
        move_mode="IDLE",
        chord_scroll=False,
        error=None,
        preview_active=bool(preview),
    ))

    combo_hint = ""
    combo_hint_counter = 0
    # 单手张开移窗：上一帧结果（Pass B 光标锁在 mover.update 之前读取，滞后 1 帧，
    # 进入/退出 ACTIVE 各有 1 帧交接，帧内同点 SetCursorPos 幂等、肉眼不可见）
    mover_info = None
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
    # ⭐⭐ 2026-10-02 墙钟化（MOUSE-STAT 基线：静瞄 p50 1.4~1.9px / 大跳 max 83px 穿透）：
    #   固定 α=0.85 在 18fps 实测帧率下 τ 仅 ~27ms ≈ 无平滑；改为 α=1-exp(-dt/τ)，τ=60ms：
    #   30fps→α≈0.43，18fps→α≈0.60，物理平滑强度帧率无关。
    #   · 跳帧复用的是【同一旧观测】→ α=0（不拿旧数据再平均一次，消除前冲-回拉锯齿）
    #   · 硬重置改为【双帧确认】：连续两帧掌位跳 >tol 才真重置（真快速移动第2帧仍确认，延迟≤1帧）；
    #     单帧飞点不重置 EMA 状态，配合鼠标端中值窗被彻底滤除
    _LM_EMA_TAU_SEC = 0.060
    _EMA_RESET_TOL = 0.015                 # 掌心单帧归一化位移超过此值 → 清 EMA 缓存（≈1280×720下≈10像素）

    def _lm_ema_alpha(dt: float) -> float:
        _d = dt if dt > 0.0 else 1.0 / 30.0   # 首帧/异常 dt 回退 30fps
        if _d > 0.5:
            _d = 0.5
        return 1.0 - _mt.exp(-_d / _LM_EMA_TAU_SEC)

    _ema_landmarks: dict[str, list] = {}   # key = Left/Right, value = 21 个 [x,y,z] 列表
    _ema_miss_count: dict[str, int] = {}   # 连续丢失计数，超过阈值清空 EMA
    _ema_reset_pending: dict[str, bool] = {}  # 上一帧是否已出现掌位大跳（双帧确认硬重置）
    # ⭐ 同视角身份时序稳定器（2026-09-30）：手背入镜/手掌旋转时锁定左右手身份，
    #   防止 MP handedness 单帧横跳 → EMA 串手 / 控鼠手瞬切 / 双手拉伸左右对调
    _hand_id_stab = HandIdentityStabilizer()

    # (2) 手势标签去抖：帧投票 → 时钟区间稳定器（用户拍板：不按帧、
    #     按墙钟区间判断）。每只手一个 GestureTimeStabilizer：候选新手势需持续满
    #     0.10s（离开 FIST 需 0.17s）才翻转，自适应跳帧下去抖时长仍确定。
    #     旧 3 帧多数投票（_MV_WINDOW/_mv_history）已删除。
    _gest_stab: dict[str, GestureTimeStabilizer] = {}
    _mv_final: dict[str, str] = {}   # 仅用于本帧手势翻转检测（上一帧稳定值）

    # [FIST-CHK] 左手（物理左手=交换后标签 "Right"）握拳判定观测：
    #   事件即时（入镜/离镜、raw FIST 翻转）+ 0.5s 心跳，不逐帧刷屏。
    _fist_chk_t0 = 0.0            # 上次心跳墙钟
    _fist_chk_present = False     # 上一帧左手是否在镜
    _fist_chk_stable_fist: bool | None = None  # 上一帧左手稳定输出是否 FIST（None=未在镜）

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

    # 画圈滚轮统计：和弦期逐帧追踪 + 20s 汇总（纯观测）
    _cs_stat_t0 = time.monotonic()
    _cs_rs = {"held": 0, "stale": 0, "fresh": 0, "miss": 0, "inactive": 0,
              "LOCK": 0, "FLIP": 0, "RELEASE": 0, "qp": 0.0, "qn": 0.0}

    # ⭐ 2026-10-02 丢手诊断（纯观测，零行为变更）：
    #   每 20s 一行 [DET-STAT]，把"丢手"拆成可归因的计数：
    #   · 检测层（detector.take_det_diag）：实际推理数 / 80ms回调超时 / 有结果但空手 / 延迟
    #   主循环 bend=None 归因：no_left=本帧无控鼠手标签（检测空手/跟踪丢失/跳帧前已丢）
    #                           palm=掌宽≈0（手太远/关键点残缺）
    #   （起 fist/fist_wm 归因随握拳全局 guard 一并删除）
    #   · 跳帧率：_skip_was_skipped 帧占比（跳帧复用不产生新检测，高跳帧率+高超时=推理拥塞线索）
    #   · 帧亮度：黑帧（均值<_CAM_BLACK_MEAN_THR，与预热同口径）计数 + 窗口 min/avg，
    #             用于区分"检测空手"是手真不在还是相机阶段性吐黑/暗帧（read 成功但画面黑）
    _DET_STAT_SEC = 20.0
    _det_stat_t0 = time.monotonic()
    _det_rs = {"frames": 0, "skip": 0,
               "no_left": 0, "palm": 0, "curl": 0, "other": 0,
               # 亮度（与预热同口径：全帧 BGR 均值；黑帧=_CAM_BLACK_MEAN_THR）
               # black_run/black_run_max：黑帧连续段长度（区分"零星黑帧"与"相机持续吐黑=假死"）
               "black": 0, "b_sum": 0.0, "b_min": None,
               "black_run": 0, "black_run_max": 0,
               # 2026-10-02 卡顿定位：cap.read() 单次耗时 max + >150ms 慢读次数；waitKey max
               "read_max_ms": 0.0, "read_slow": 0, "wk_max_ms": 0.0}
    # 2026-10-03 [TAP-SEG] 段生命周期埋点：ARM 时刻屏幕坐标 + 段内 bend=None 原因累计
    _seg_arm_xy = None
    _seg_none_rs: dict = {}
    # 2026-10-03 [2F-SEG] 右键段埋点：ARM 时刻坐标 + 食/中指独立弯度段内极值
    _rf_arm_xy = None
    _rf_arm_idx = None
    _rf_arm_mid = None
    _rf_idx_min = _rf_idx_max = None
    _rf_mid_min = _rf_mid_max = None
    # TIP/DIP 实测比值段内极值 + ARM 前滚动窗（看指尖"实测多开"、资格何时取得）
    _rf_tip_min = _rf_tip_max = None
    _rf_dip_min = _rf_dip_max = None
    _rf_pre_tip: list = []       # ARM 前最近 ~12 个有效帧 (TIP比值, 空间)
    _rf_arm_pretip: list = []    # 本次 ARM 快照
    _rf_pre_2d = 0
    # 2026-10-04 [2F-OBS] 左键段内的指尖距/中弯原始数据（验证"单指点击时两尖相隔≥1指节"）：
    #   左键 ARM 当帧快照 + 段内极值，随每条 [TAP-SEG] 打印
    _lseg_tip_arm = None
    _lseg_tip_min = None
    _lseg_mid_arm = None
    _lseg_mid_min = _lseg_mid_max = None
    # 2026-10-04 [BEND3D] 左键段双口径（2D vs 3D world）纯观测：
    #   b3d 段内谷底、3D DIP 角谷底、压缩比 min、2D-3D 最大差（发生时的压缩比）、
    #   world TIP 逐帧跳变 max、末节指骨长度
    # [LONG3D] mcp_min=食近节角段内【最大】值（握拳方向角变大，取 max 看最接近/越界）
    _lseg_b2d_arm = None
    _lseg_b3d_arm = None
    _lseg_b3d_min = None
    _lseg_comp_min = None
    _lseg_dip3d_min = None
    _lseg_mcp_max = None
    _lseg_diff_max = None
    _lseg_diff_comp = None
    _lseg_jump_max = None
    _lseg_phal = None
    # [BEND3D] 右键段 3D 双指单元合成弯度（与 2D r 同映射）
    _rf_b3d_arm = None
    _rf_b3d_min = None
    # [BEND3D-JIT] 悬停双口径噪声：仅 IDLE 帧入队，约每 2s 打一行 mean±std
    _jit3d_q: deque = deque(maxlen=64)
    _jit3d_sec = 0.0

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
        _bg_heartbeat_cnt = 0   # 无预览模式心跳帧计数（~每120帧打印一次 RUN-OK）
        # ⭐ 2026-10-01 摄像头读帧两级容错（修复"运行中偶发 1 帧 cap.read()=False 整个服务退出"）：
        #   第一级：偶发空帧 → 等 20ms 重试，读到好帧立即清零（USB 驱动瞬时卡顿下帧自恢复，
        #          下游本就有"黑帧冻结/跳帧复用"语义，用户无感）；
        #   第二级：连续失败满 5s → 判定掉线/被别的程序抢占，release 旧句柄并走完整的
        #          _open_camera_robust 枚举+预热自动重开（USB 重插也能自愈），成功继续运行；
        #          重开仍失败才退出，并把枚举尝试记录打全（那才是真故障）。
        _CAM_READ_RETRY_SEC = 5.0
        _CAM_READ_FAIL_SLEEP = 0.02
        _cam_read_fail_since: float | None = None
        _cam_reopen_count = 0
        # ✨ 2026-10-01 全项目墙钟化：主循环统一时间源（monotonic），下传给所有"帧数→秒数"状态机
        _last_frame_wall = time.monotonic()
        _frame_dt = 0.0   # 上一帧→当前帧的墙钟间隔（首帧0；含异常保护，永不抛错）

        # ✨ 2026-10-02 卡顿看门狗（纯观测）：主循环若在任一同步调用
        #   （cap.read / UIA 右键探测 / HighGUI waitKey）上阻塞 >1s，
        #   独立线程立刻打印【阻塞阶段】；恢复时打印本次卡顿总时长。
        #   背景：21:22 会话某 20s 窗口仅跑 59 帧（正常 360~435）且无空帧日志，
        #   说明线程卡在某个"既不返回也不报错"的调用里，20s 聚合统计无法区分卡点。
        # ⭐ 2026-10-30 [WATCHDOG] 卡顿诊断日志——与捏合手势观察无关，按用户要求注释
        #   （_hb_wall/_hb_stage 定义保留：主循环内各处仍在写阶段标记，纯内存无输出）。
        _hb_wall = [time.monotonic()]
        _hb_stage = ["init"]
        # def _watchdog_loop():
        #     _stall_since = None
        #     _stall_stage = None
        #     _last_holler = 0.0
        #     while stop_event is None or not stop_event.is_set():
        #         time.sleep(0.5)
        #         try:
        #             _age = time.monotonic() - _hb_wall[0]
        #             if _age >= 1.0:
        #                 _now = time.monotonic()
        #                 if _stall_since is None:
        #                     _stall_since = _now - _age  # 近似真实卡死起点（心跳最后更新时刻）
        #                     _stall_stage = _hb_stage[0]
        #                     print(f"[WATCHDOG] ⚠ 主循环卡顿已{_age:.1f}s，阻塞阶段={_stall_stage}",
        #                           flush=True)
        #                     _last_holler = _now
        #                 elif _now - _last_holler >= 2.0:
        #                     print(f"[WATCHDOG] ⚠ 卡顿持续{_now-_stall_since:.1f}s，"
        #                           f"阶段={_hb_stage[0]}", flush=True)
        #                     _last_holler = _now
        #             elif _stall_since is not None:
        #                 print(f"[WATCHDOG] ✅ 恢复（本次卡 {time.monotonic()-_stall_since:.1f}s，"
        #                       f"卡顿时阶段={_stall_stage}，恢复时阶段={_hb_stage[0]}）", flush=True)
        #                 _stall_since = None
        #                 _stall_stage = None
        #         except Exception:
        #             pass
        # threading.Thread(target=_watchdog_loop, name="loop-watchdog", daemon=True).start()

        # ⭐ 2026-10-29 [2F-OBS] 右键观测模式提示——与捏合手势观察无关，按用户要求注释。
        # if RF_OBSERVE_ONLY:
        #     print("=" * 64, flush=True)
        #     print("[2F-OBS] 右键观测模式已开启：右键不会弹出、不会抢占左键（左键恢复正常点击）",
        #           flush=True)
        #     print("[2F-OBS] 请按顺序做动作：① 正常单指点击 10 次（含慢双击几组）；"
        #           "② 刻意做 5 次双指并拢勾按（右键姿势）；然后 /stop", flush=True)
        #     print("[2F-OBS] 日志看 [TAP-SEG] 末尾 TIP= 与 [2F-SEG] 的 TIP段内，"
        #           "用于按真机数据定指尖距门限", flush=True)
        #     print("=" * 64, flush=True)

        while True:
            # 🔴 新增：stop_event 命中 → 立即干净退出
            if stop_event is not None and stop_event.is_set():
                break

            # ✨ 墙钟 dt：所有状态机共用的唯一时间源（异常保护：任何负值/跳变都夹回 0）
            try:
                _wall_now = time.monotonic()
                _frame_dt = float(_wall_now) - float(_last_frame_wall)
                if _frame_dt < 0.0 or _frame_dt > 0.5:
                    _frame_dt = 0.0
                _last_frame_wall = _wall_now
            except Exception:
                _frame_dt = 0.0

            _bg_heartbeat_cnt += 1
            _hb_wall[0] = time.monotonic()  # 看门狗心跳：走到此处=上一帧全部工作已完成
            _hb_stage[0] = "cap.read"
            _t0 = time.perf_counter()
            ok, frame = cap.read()
            _t1 = time.perf_counter()
            try:  # 读帧耗时统计（含失败路径；read 若硬阻塞看门狗会先报阶段=cap.read）
                _rms = (_t1 - _t0) * 1000.0
                if _rms > _det_rs["read_max_ms"]:
                    _det_rs["read_max_ms"] = _rms
                if _rms > 150.0:
                    _det_rs["read_slow"] += 1
            except Exception:
                pass
            if not ok or frame is None:
                # ---- 第一级：偶发空帧，短等重试 ----
                if _cam_read_fail_since is None:
                    _cam_read_fail_since = _t0
                    print("[hands_tasks.run_loop] ⚠ 摄像头空帧，重试中（偶发不退出）…", flush=True)
                _fail_elapsed = _t0 - _cam_read_fail_since
                if _fail_elapsed < _CAM_READ_RETRY_SEC:
                    time.sleep(_CAM_READ_FAIL_SLEEP)
                    continue
                # ---- 第二级：连续 5s 失败 → 自动重开相机 ----
                print(f"[hands_tasks.run_loop] ⚠ 连续 {_fail_elapsed:.1f}s 读帧失败，"
                      f"尝试自动重开摄像头 camera_id={_cid!r}…", flush=True)
                try:
                    cap.release()
                except Exception:
                    pass
                _new_cap, _new_info = _open_camera_robust(
                    int(_cid),
                    int(_cap_info.get("width", 1280) or 1280),
                    int(_cap_info.get("height", 720) or 720))
                if _new_cap is not None:
                    cap = _new_cap
                    if isinstance(_new_info, dict):
                        _cap_info = _new_info
                    _cam_reopen_count += 1
                    _cam_read_fail_since = None
                    print(f"[hands_tasks.run_loop] ✅ 摄像头自动重开成功"
                          f"（第{_cam_reopen_count}次，后端={_cap_info.get('backend')}），继续运行",
                          flush=True)
                    continue
                _attempts = _new_info.get("attempts_summary", []) if isinstance(_new_info, dict) else []
                print(f"[hands_tasks.run_loop] ❌ 连续 {_fail_elapsed:.1f}s 读帧失败，自动重开也失败，退出。"
                      f"尝试记录:\n" + ";\n".join(f"    {_p}" for _p in _attempts),
                      flush=True)
                break
            # 读到好帧：连续失败计时立即清零（偶发失败不跨帧累计）
            _cam_read_fail_since = None
            _hb_stage[0] = "business"

            # ---- 诊断：运行期帧亮度（与预热同口径；纯观测，不影响任何判定）----
            try:
                _frm_mean = float(frame.mean())
                _det_rs["b_sum"] += _frm_mean
                _det_rs["b_min"] = (_frm_mean if _det_rs["b_min"] is None
                                    else min(_det_rs["b_min"], _frm_mean))
                if _frm_mean < _CAM_BLACK_MEAN_THR:
                    _det_rs["black"] += 1
                    _det_rs["black_run"] += 1
                    if _det_rs["black_run"] > _det_rs["black_run_max"]:
                        _det_rs["black_run_max"] = _det_rs["black_run"]
                else:
                    _det_rs["black_run"] = 0
            except Exception:
                pass

            # ---------- 第一人称同视角：画面【不】做水平镜像 ----------
            # 头戴相机视线与人眼一致：手向右动 → 画面 x 向右增，光标自然同向。
            # 旧面对面视角在此 cv2.flip(frame, 1)（自拍镜像），同视角改造后永久移除。
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
                # 骨架统一在下方 EMA 之后绘制（跳帧/正常帧同一出口，所见即平滑值）
            else:
                _skip_was_skipped = False
                # ---------- 正常：双手检测 ----------
                hands_data, frame = detector.detect(frame)
                # ---- [MP-RAW] 检测出口原始观测（标签交换/身份稳定前）----
                # 目的：一锤定音区分"模型层只返回了一只手"与"稳定器/标签交换吞手"。
                # 仅在物理左手(交换前 MP label=Left)在场或和弦保持中打印，避免污染日常日志。
                # 坐标按画面短边归一（与滚轮检测器口径一致）；palm=MCP四指根质心，tip=TIP8。
                # ⭐ 2026-10-29 [MP-RAW] 检测出口原始观测日志——与捏合手势观察无关，按用户要求注释。
                # try:
                #     if (not _skip_was_skipped and (
                #             getattr(run_loop, "_chord_scroll_hold", False)
                #             or any(_h.get("label") == "Left" for _h in hands_data))):
                #         _short0 = float(min(w, h))
                #         _parts = []
                #         for _h in sorted(hands_data, key=lambda z: z.get("label", "")):
                #             _l = _h["landmarks"]
                #             _px = 0.25 * (float(_l[5].x) + float(_l[9].x)
                #                           + float(_l[13].x) + float(_l[17].x)) * w / _short0
                #             _py = 0.25 * (float(_l[5].y) + float(_l[9].y)
                #                           + float(_l[13].y) + float(_l[17].y)) * h / _short0
                #             _parts.append(
                #                 f"{_h.get('label')}?s={float(_h.get('score', 0) or 0):.2f}"
                #                 f" palm={_px:.3f},{_py:.3f}"
                #                 f" tip={float(_l[8].x) * w / _short0:.3f},"
                #                 f"{float(_l[8].y) * h / _short0:.3f}")
                #         print(f"[MP-RAW] f={_bg_heartbeat_cnt} n={len(hands_data)} "
                #               + (" | ".join(_parts) if _parts else "(空手)"),
                #               flush=True)
                # except Exception:
                #     pass
                # ⭐ 同视角标签归一化：MediaPipe 左右手标签假定输入为【镜像自拍图】
                #   （旧面对面流程靠 cv2.flip 满足该假定，用户右手原始标签恰为 "Left"）。
                #   同视角输入不镜像，按官方规则在应用层交换 handedness：
                #   交换后沿用全项目既有约定 —— label "Left" = 用户右手 = 控鼠手，
                #   下游 EMA / 手势投票 / 鼠标 / 点击 / 拉伸 全部无需改动。
                #   🔴 只能在【新检测结果】处交换一次：跳帧复用的 _skip_last_hands_data
                #      已在下方 deepcopy 前完成交换，重复交换会双手身份互换。
                # ⭐⭐⭐ 2026-10-29 手性判别——真机三会话教训（结论：单手帧不赌模型、赌用户习惯）：
                #   三重积 s=(P2-P0)·((P5-P0)×(P17-P0)) 在【真实刚体】上旋转不变，
                #   但 MediaPipe 的 world 坐标是【单目深度估计】：翻面（掌↔背）时
                #   z 深度符号歧义会输出镜像骨架，s 随之变号。真机实锤：同一只用户
                #   右手，s 一个会话恒负、一个会话恒正、还出现过会话内变号 →
                #   s 不能作单手帧身份判据（MP 外观分类在手背朝镜头时同样 0.51~0.95
                #   横跳）。单手帧改为固定认"用户右手"（见 for 循环内分支）——依据
                #   用户固定使用模式：右手控鼠，左手仅在"✊握拳+右手转动=滚轮"和弦
                #   时入镜（双手帧）。三重积仅保留给双手帧（区分谁左谁右，配身份
                #   稳定器"撞车取另一标签"兜底；其翻面误判只影响和弦，待实测再调）。
                _chir_dbg = []
                for _h in hands_data:
                    if len(hands_data) == 1:
                        # 单手在镜 = 用户右手（控鼠手）。不跑三重积（翻面会变号，
                        # 真机已实锤）、不跑 MP 外观（横跳），直接锁定语义。
                        _h["label"] = "Left"
                        _chir_dbg.append("Left(单手=右手)")
                        continue
                    _wl_chir = _h.get("world_landmarks")
                    _s_chir = 0.0
                    if _wl_chir is not None and len(_wl_chir) >= 18:
                        try:
                            _p0 = np.array([_wl_chir[0].x, _wl_chir[0].y, _wl_chir[0].z], dtype=np.float64)
                            _p2 = np.array([_wl_chir[2].x, _wl_chir[2].y, _wl_chir[2].z], dtype=np.float64)
                            _p5 = np.array([_wl_chir[5].x, _wl_chir[5].y, _wl_chir[5].z], dtype=np.float64)
                            _p17 = np.array([_wl_chir[17].x, _wl_chir[17].y, _wl_chir[17].z], dtype=np.float64)
                            _n_chir = np.cross(_p5 - _p0, _p17 - _p0)
                            if float(np.linalg.norm(_n_chir)) > 1e-9:
                                _s_chir = float(np.dot(_p2 - _p0, _n_chir))
                        except Exception:
                            _s_chir = 0.0
                    if abs(_s_chir) > 1e-8:
                        # 物理右手 s<0 → "Left"（控鼠手）；物理左手 s>0 → "Right"
                        _h["label"] = "Left" if _s_chir < 0 else "Right"
                        _chir_dbg.append(f"{_h['label']}(s={_s_chir:+.2e})")
                    else:
                        # 回退：旧同视角镜像假定交换（world 缺失时行为不变）
                        if _h.get("label") == "Left":
                            _h["label"] = "Right"
                        elif _h.get("label") == "Right":
                            _h["label"] = "Left"
                        _chir_dbg.append(f"{_h['label']}(回退)")
                # ⭐ 2026-10-29 [CHIR] 手性判定对照日志——与捏合手势观察无关，按用户要求注释。
                # if (not _skip_was_skipped and hands_data
                #         and _bg_heartbeat_cnt % 30 == 0):
                #     print(f"[CHIR] f={_bg_heartbeat_cnt} "
                #           + " | ".join(_chir_dbg),
                #           flush=True)
                # ⭐ 身份时序稳定（单一入口，紧挨标签交换）：以掌心位置跨帧锁定身份，
                #   手背入镜/转掌造成的 MP 标签单帧横跳在此被消化为【不变】。
                #   必须在 deepcopy 之前：跳帧复用的帧携带的就是已稳定身份，不重复 apply。
                _hand_id_stab.apply(hands_data, _frame_dt)
                # 单手模式固定为用户右手控鼠：稳定器可能沿用双手阶段的 Right 轨迹标签，
                # 造成下游 hand_results 没有 Left，进而完全跳过鼠标移动。
                if len(hands_data) == 1:
                    hands_data[0]["label"] = "Left"
                _skip_last_hands_data = _cp.deepcopy(hands_data)

            # ================================================================
            # 🔥 landmarks EMA 指数滑动平均：去关节抖动，提升小范围瞄准稳定性
            # ================================================================
            from hand_detector import _LmWrapper
            # 本帧 EMA 增益：跳帧复用同一旧观测 → α=0（状态外推而非再平均）
            _ema_a = 0.0 if _skip_was_skipped else _lm_ema_alpha(_frame_dt)
            _cur_labels_in_frame: set = set()
            for _hand in hands_data:
                _lab = _hand["label"]
                _cur_labels_in_frame.add(_lab)
                _cur_arr = [[float(_p.x), float(_p.y), float(_p.z)] for _p in _hand["landmarks"]]

                # ✅ 掌位大位移硬重置（彻底消除快速移动时的EMA拖尾）：
                #   先计算当前掌位（⭐ 2026-10-01 起与 cursor anchor 同真相：四指根
                #   MCP 5/9/13/17 质心，即 args.anchor=mcp；旧 0+9 中点在手腕出画时漂移）
                #   如果这一帧掌位相对上一帧跳了 > _EMA_RESET_TOL（≈10px归一化），说明是真实快速移动，
                #   直接丢掉历史EMA（不拖尾巴），强制用当前值（然后下一帧再从这个点开始EMA）。
                _cx0_cur = 0.25 * (_cur_arr[5][0] + _cur_arr[9][0]
                                      + _cur_arr[13][0] + _cur_arr[17][0])
                _cy0_cur = 0.25 * (_cur_arr[5][1] + _cur_arr[9][1]
                                      + _cur_arr[13][1] + _cur_arr[17][1])
                _reset_this_hand = False
                if _lab in _ema_landmarks:
                    _last_arr = _ema_landmarks[_lab]
                    _cx0_prev = 0.25 * (_last_arr[5][0] + _last_arr[9][0]
                                           + _last_arr[13][0] + _last_arr[17][0])
                    _cy0_prev = 0.25 * (_last_arr[5][1] + _last_arr[9][1]
                                           + _last_arr[13][1] + _last_arr[17][1])
                    if abs(_cx0_cur - _cx0_prev) > _EMA_RESET_TOL or abs(_cy0_cur - _cy0_prev) > _EMA_RESET_TOL:
                        # 双帧确认：本帧跳且上帧也跳 → 认定真快速移动，硬重置；
                        # 仅单帧跳 → 当飞点，不重置（EMA 自然削幅），清除 pending 等下一帧
                        if _ema_reset_pending.get(_lab, False):
                            _reset_this_hand = True
                            _ema_reset_pending[_lab] = False
                        else:
                            _ema_reset_pending[_lab] = True
                    else:
                        _ema_reset_pending[_lab] = False

                if _lab in _ema_landmarks and not _reset_this_hand:
                    _last_arr = _ema_landmarks[_lab]
                    _smoothed = [
                        [_ema_a * _cur_arr[i][0] + (1 - _ema_a) * _last_arr[i][0],
                         _ema_a * _cur_arr[i][1] + (1 - _ema_a) * _last_arr[i][1],
                         _ema_a * _cur_arr[i][2] + (1 - _ema_a) * _last_arr[i][2]]
                        for i in range(21)
                    ]
                else:
                    # 首次出现 / 丢失后重新出现 / 掌心大位移硬重置（双帧确认）→ 用当前原数值（不拖历史尾巴）
                    if _reset_this_hand and _lab in _ema_landmarks:
                        pass  # 马上会用 _cur_arr 覆盖掉
                    _smoothed = _cur_arr
                _ema_landmarks[_lab] = _smoothed
                _ema_miss_count[_lab] = 0
                _ema_reset_pending.setdefault(_lab, False)
                # 把平滑后的 wrapped 回填到 hand["landmarks"]（后续所有业务都用平滑值）
                _hand["landmarks"] = [_LmWrapper(_v[0], _v[1], _v[2]) for _v in _smoothed]
            # 丢失的手：miss_count 累计，超时清空
            for _lab in list(_ema_landmarks.keys()):
                if _lab not in _cur_labels_in_frame:
                    _ema_miss_count[_lab] = _ema_miss_count.get(_lab, 0) + 1
                    if _ema_miss_count[_lab] > 8:
                        _ema_landmarks.pop(_lab, None)
                        _ema_miss_count.pop(_lab, None)
                        _ema_reset_pending.pop(_lab, None)

            # ================================================================
            # 骨架绘制（EMA 平滑后唯一出口）：画面骨架 = 下游手势/鼠标/点击
            # 实际使用的同一份平滑关键点，正常帧与跳帧复用帧都走这里。
            # ================================================================
            for _h in hands_data:
                _pts = [(int(float(_l.x) * w), int(float(_l.y) * h))
                        for _l in _h["landmarks"]]
                for (_px, _py) in _pts:
                    cv2.circle(frame, (_px, _py), 3, (255, 0, 255), 2)
                for _a, _b in HAND_CONNECTIONS:
                    if _a < len(_pts) and _b < len(_pts):
                        cv2.line(frame, _pts[_a], _pts[_b], (0, 255, 0), 2)

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
                # FIST 判定走用户的 3D world 五指卷心比
                # r_i=dist(TIP_i,P)/dist(PIP_i,P)（P=0/5/17 掌心三角点，五指全 <1.2
                # 重建为全弯 → 表判 FIST），再经时钟区间稳定器去抖；
                # world 缺失时识别器内部自动回退旧 2D 五指全弯门。
                gesture_raw, ext, ratios = recognize_gesture(
                    lms, hand.get("world_landmarks"))
                # 左手是和弦修饰手，只做握拳/张开，不会捏合；PINCH/OK 一律按 ext 重判
                # （握拳全 0→FIST，张开全 1→OPEN，否则 OTHER），避免拇食指靠拢误判。
                # OK 是右手单手手势，左手做 OK 无特殊语义，且 OK 的 ext=(1,1,1,1,1) 与 OPEN 相同，
                # 若重判为 OPEN 会误触发窗口移动 → 左手 OK 统一重判为 OTHER。
                # ⭐ 2026-10-29 OK 手势识别已在 gesture_recognizer 注释 → gesture_raw 不再可能是
                #   "OK"；此处仅保留 PINCH 重判（ext 全1→OPEN 标签已无任何动作逻辑，inert）。
                if label == "Right" and gesture_raw == "PINCH":
                    # if gesture_raw == "OK":
                    #     gesture_raw = "OTHER"
                    if ext == (0, 0, 0, 0, 0):
                        gesture_raw = "FIST"
                    elif ext == (1, 1, 1, 1, 1):
                        gesture_raw = "OPEN"
                    else:
                        gesture_raw = "OTHER"
                # --- 时钟区间稳定（替代旧帧投票；墙钟确认见 GestureTimeStabilizer）---
                stab = _gest_stab.get(label)
                if stab is None:
                    stab = GestureTimeStabilizer()
                    _gest_stab[label] = stab
                    _mv_final.pop(label, None)   # 新入镜：清旧值，首帧直接采纳
                gesture_final = stab.update(gesture_raw, _wall_now)
                if label in _mv_final and _mv_final[label] != gesture_final:
                    _gestures_changed_any = True
                _mv_final[label] = gesture_final
                # 用投票后的手势（不改变 ext/ratios，保持原始弯曲度反馈）
                hand_results[label] = (gesture_final, ext, ratios, lms)

                # ---- [FIST-CHK] 左手握拳判定日志（物理左手=标签 Right）----
                #   ⭐ 2026-10-29 与捏合手势观察无关，按用户要求注释（和弦滚轮逻辑本身不动）。
                # if label == "Right":
                #     _now_chk = time.monotonic()
                #     _stable_fist = (gesture_final == "FIST")
                #     if "fold_i" in ratios:
                #         _fold_s = (f"fold 拇/食/中/无/小={ratios['fold_t']}/"
                #                    f"{ratios['fold_i']}/{ratios['fold_m']}/"
                #                    f"{ratios['fold_r']}/{ratios['fold_p']}"
                #                    f"(<1.2=弯) ext(拇/食/中/无/小)={ext}")
                #     else:
                #         _fold_s = (f"fold n/a（world缺失→2D布尔）"
                #                    f" ext(拇/食/中/无/小)={ext}")
                #     _enter = not _fist_chk_present
                #     _flip = (_fist_chk_stable_fist is not None
                #              and _fist_chk_stable_fist != _stable_fist)
                #     if (_enter or _flip
                #             or _now_chk - _fist_chk_t0 >= 0.5):
                #         _fist_chk_t0 = _now_chk
                #         _tag = "入镜·" if _enter else ("【握拳翻转】" if _flip else "")
                #         print(f"[FIST-CHK] 左手 {_tag}判定握拳="
                #               f"{'是' if _stable_fist else '否'} "
                #               f"(raw={gesture_raw}/stable={gesture_final}) {_fold_s}",
                #               flush=True)
                #     _fist_chk_present = True
                #     _fist_chk_stable_fist = _stable_fist
                # 记录掌位（归一化）——⭐ 2026-10-01 与光标锚点同源（args.anchor，默认 mcp），
                # 跳帧稳定性判定的位移基准必须等于光标实际跟随的点，否则两者门限语义脱节
                _palm_x, _palm_y = HandDetector.cursor_anchor_from_landmarks(lms, args.anchor)
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

            # 手离镜：复位该手的时钟稳定器与翻转记录（防单向锁；重新入镜首帧直接采纳）
            for _lab in ("Left", "Right"):
                if _lab not in hand_results:
                    if _lab in _gest_stab:
                        _gest_stab[_lab].reset()
                        del _gest_stab[_lab]
                    _mv_final.pop(_lab, None)
                    # 右手离镜：
                    #   - GameViewer 左键长按：立即释放（无鼠标拖动连续性要保持）；
                    #   - OK 缩放（SetWindowPos）：丢手即停止，直接复位（无鼠标键）。
                    # ⭐ 2026-10-29 GV拖动/OK缩放分支均已注释（_gv_press/_ok_resize_started
                    #   不再会被置位），本清理块成为死代码，一并注释。
                    # if _lab == "Left":
                    #     if getattr(run_loop, "_gv_press", False):
                    #         try:
                    #             pyautogui.mouseUp(button='left')
                    #         except Exception:
                    #             pass
                    #         run_loop._gv_press = False
                    #     if getattr(run_loop, "_ok_resize_started", False):
                    #         run_loop._ok_resize_started = False
                    #         run_loop._ok_miss_t0 = None

            # [FIST-CHK] 左手本帧不在镜（且上一帧在）→ 离镜事件打印一次
            #   ⭐ 2026-10-29 与捏合手势观察无关，按用户要求注释。
            # if _fist_chk_present and "Right" not in hand_results:
            #     print("[FIST-CHK] 左手 离镜（本帧无左手）", flush=True)
            #     _fist_chk_present = False
            #     _fist_chk_stable_fist = None

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
            # (B1) 全局手势布尔：
            #   both_hands_global —— 双手同时在镜：屏蔽点击（旧拉伸/移窗防误触语义；
            #                        2026-10-14 拉伸模块整体删除后保留此最小双手门，
            #                        也为后续 FIST+POINT 滚轮和弦预留互斥位）
            #   any_fist_global（任一手握拳即冻结光标/取消点击）已删除——
            #   单手握拳无任何语义；FIST 仅保留标签供未来双手修饰角色。
            both_hands_global     = ("Left" in hand_results) and ("Right" in hand_results)

            # ==================================================================
            # 双手滚轮和弦（FIST 修饰角色的唯一消费者）：
            #   左手（物理左手=标签 "Right"）稳定手势 FIST 即成立和弦；
            #   右手（物理右手=标签 "Left"）【不做任何姿势门】——不要求食指伸直、
            #   不要求中/无名指弯曲，只锁定右手食指 TIP(8) 轨迹的转动趋势
            #   （航向角判定：约 1/4 圈趋势即出格；直线划动航向恒定、天然不触发）
            #   → 右手食指 TIP 轨迹交 CircleScrollDetector 判转动趋势，
            #     顺时针=下滑、逆时针=上滑；和弦期间光标冻结（不写 move_only/handle），
            #     点击检测器沿既有 both_hands_global 分支每帧 reset（本块零改动）。
            # 🔴 入口隔离：该布尔是【双手专用支路】唯一开关。单手场景
            #    both_hands_global=False → 永远 False，单手点击/长按/移窗链路不受任何影响；
            #    双手但左手非握拳同样 False（维持旧双手行为：光标可跟、点击屏蔽）。
            # ==================================================================
            # 三次改版：和弦【迟滞保持】。真机画圈时控鼠手漏检率
            #    高达 60%（DET-STAT "无控鼠手"帧），旧规则"双手同帧才成立"会让
            #    active 随检出抖动、连续漏检 0.25s 即整体 reset，表现为越画越卡、
            #    最终锁死。新规则：
            #      起手：双手同帧 且 左手 FIST；
            #      保持：一旦成立，只要左手仍在镜且 FIST，右手丢帧【不解除】
            #            （以 active=True + 坐标 None 喂检测器，内部按漏检衰减）；
            #      解除：左手松拳/离镜（修饰语义消失，立即退出）。
            _mod_fist = False
            try:
                _mod_fist = ("Right" in hand_results
                             and hand_results["Right"][0] == "FIST")
            except Exception:
                _mod_fist = False
            _prev_chord = bool(getattr(run_loop, "_chord_scroll_hold", False))
            if _prev_chord:
                _chord_scroll = _mod_fist
            else:
                _chord_scroll = bool(both_hands_global and _mod_fist)
            run_loop._chord_scroll_hold = bool(_chord_scroll)
            # 🔴 和弦保持期【禁止自适应跳帧】：画圈时掌心近似原地、只有食指尖画
            #    弧，跳帧判定（掌位移<4.3px 连续12帧）会把"指尖运动中"误判成稳定
            #    场景 → 隔帧复用冻结的关键点喂给滚轮检测器：墙钟在走、坐标不动，
            #    ω 被假观测一路拖到 0 → LOCK↔RELEASE 空转、滚动积分≈0（真机日志：
            #    LOCK ω=8~11 后 0.2s 即 RELEASE ω=0 循环，左手 FIST 全程稳定）。
            if _chord_scroll:
                _skip_skip_next = False
                _skip_stable_cnt = 0

            # 每帧喂检测器：active=和弦迟滞保持（见上方）；active=True 但坐标
            #   None=右手漏检帧（检测器内衰减不 reset）；active=False=左手松拳/离镜，
            #   0.25s 宽限后 reset（杜绝跨场景单向锁）。
            #   🔴 跳帧复用帧（本帧根本没推理）=传感器未更新，和弦期【不喂】检测器：
            #   冻结坐标会被当成"原地不动"的真观测污染测量窗（假停转）。
            #   坐标按画面短边归一（无宽高比畸变、与分辨率无关），用原始 TIP8 关键点。
            #   返回值是【亚格连续量】（notch，可小数），交给 mouse.scroll 平滑投递。
            _scroll_q = 0.0
            _circle_px = None        # 本帧 TIP 像素（UI 轨迹绘制）
            _circle_m = {"state": "IDLE", "dir": 0, "m_dir": 0,
                         "speed": 0.0, "speed_pct": 0.0, "trail": (),
                         "ev": "", "ev_dir": 0, "ev_t": 0.0}
            _circle_short = float(min(w, h))
            if "Left" in hand_results:
                _clms = hand_results["Left"][3]
                _circle_px = (float(_clms[8].x) * w, float(_clms[8].y) * h)
                _tip_u = (float(_clms[8].x) * w / _circle_short,
                          float(_clms[8].y) * h / _circle_short)
            else:
                _tip_u = (None, None)
            _cs_stale = bool(_chord_scroll and _skip_was_skipped)
            if not _cs_stale:
                _scroll_q = float(circle_scroll.update(
                    _chord_scroll, _tip_u[0], _tip_u[1], _wall_now))
            if _chord_scroll:
                _circle_m = circle_scroll.metrics()
                # STALE 帧没有 update，_ev 是上一帧残留：丢弃，禁止重复打印/计数
                if _cs_stale:
                    _circle_m["ev"] = ""
                # ⭐ 2026-10-29 [CIRCLE-SCROLL]/[CS-STAT] 和弦滚轮观测日志——
                #   与捏合手势观察无关，按用户要求注释（metrics() 调用与 STALE ev 清理保留，
                #   UI 绘制仍读 _circle_m；滚轮行为零改动）。
                # # 仅在锁定/翻向/释放状态跳变时打日志（连续滚动帧不刷屏）
                # _cv_ev = _circle_m.get("ev", "")
                # if _cv_ev:
                #     _cv_dir = int(_circle_m.get("ev_dir", 0))
                #     _cv_txt = {"LOCK": "锁定方向", "FLIP": "翻向", "RELEASE": "停转释放"}[_cv_ev]
                #     print(f"[CIRCLE-SCROLL] {_cv_txt}："
                #           f"{'逆时针⬆上滑' if _cv_dir > 0 else '顺时针⬇下滑' if _cv_dir < 0 else '-'}"
                #           f" ω={_circle_m.get('speed', 0.0):.2f}rad/s（左手握拳和弦）",
                #           flush=True)
                # # ---- [CS-DBG] 和弦期逐帧追踪（定位"后半程卡死"，纯观测）----
                # _cs_rs["held"] += 1
                # if _cs_stale:
                #     _cs_rs["stale"] += 1
                # elif _tip_u[0] is None:
                #     _cs_rs["miss"] += 1
                # else:
                #     _cs_rs["fresh"] += 1
                # if _cv_ev:
                #     _cs_rs[_cv_ev] += 1
                # _cs_rs["qp"] += max(_scroll_q, 0.0)
                # _cs_rs["qn"] += min(_scroll_q, 0.0)
                # if time.monotonic() - _cs_stat_t0 >= _DET_STAT_SEC:
                #     print(
                #         f"[CS-STAT] 和弦帧{_cs_rs['held']}:"
                #         f" 新鲜{_cs_rs['fresh']} 漏检{_cs_rs['miss']}"
                #         f" 跳帧复用{_cs_rs['stale']} | "
                #         f"LOCK{_cs_rs['LOCK']} FLIP{_cs_rs['FLIP']}"
                #         f" RELEASE{_cs_rs['RELEASE']} | "
                #         f"积分 q+={_cs_rs['qp']:.1f} q-={_cs_rs['qn']:.1f}",
                #         flush=True)
                #     _cs_stat_t0 = time.monotonic()
                #     for _k in ("held", "stale", "fresh", "miss",
                #                "LOCK", "FLIP", "RELEASE"):
                #         _cs_rs[_k] = 0
                #     _cs_rs["qp"] = _cs_rs["qn"] = 0.0
            if abs(_scroll_q) > 1e-6:
                mouse.scroll(_scroll_q)

            # 单手张开移窗 ACTIVE 期间 window_mover 每帧 SetCursorPos 锁窗口中心（防误触），
            # 掌心跟随 move_only 若同时写光标会在窗口中心↔掌心映射点间闪动 → 锁定期跳过一切掌心光标写入。
            # 读上一帧 mover_info（update 在本帧稍后才跑），进出 ACTIVE 各 1 帧交接、幂等不可见。
            _cursor_locked_by_window_move = bool(
                mover_info is not None and mover_info.move_state == _WMS.MOVE_ACTIVE
            )

            # ======================================================================
            # ⭐ B-1：鼠标移动急活【最前置】
            #   全局门控算完 → Pass C / Tap 等慢活之前，
            #   先让掌心光标出帧，消除慢活阻塞造成的“鼠标钉住 + 之后大跳”。
            #   单手 FIST 不再冻结光标，按中性 OTHER 出帧（握拳无单手语义）。
            # ======================================================================
            # ======================================================================
            # 每帧判定光标所在窗口是否 gameviewer.exe（驱动交互分支切换）
            # ⭐ 2026-10-29 GV 交互分支（五指张开/OK=左键长按拖动）已按用户需求注释，
            #   该每帧 Win32 探测不再有消费者，一并注释省开销。
            # ======================================================================
            # try:
            #     run_loop._is_gv = bool(is_gameviewer_cursor_window())
            # except Exception:
            #     run_loop._is_gv = False

            # ======================================================================
            # Pass B-1：控鼠手（Left=用户右手）掌心光标【先出帧】
            # ======================================================================
            _pre_move_only_done_right = False
            if _cursor_locked_by_window_move:
                # 移窗锁定期：本帧掌心 move_only 禁止执行；置位阻止 Pass C 兜底再调一次 moveTo
                _pre_move_only_done_right = True
            elif _chord_scroll:
                # 画圈滚轮和弦期：光标冻结在和弦进入前的位置，指尖画圈只产生滚轮，不移动光标。
                # 置位同时阻止 Pass C 兜底 move_only / handle。
                _pre_move_only_done_right = True
            elif "Left" in hand_results:
                try:
                    _gesture, _ext, _ratios, _lms = hand_results["Left"]
                    _cursor_anchor_norm = HandDetector.cursor_anchor_from_landmarks(_lms, args.anchor)
                    # _is_gv = bool(getattr(run_loop, "_is_gv", False))

                    # ⭐⭐ 2026-10-29 用户需求：注释五指张开(OPEN) / OK 手势的全部动作逻辑
                    #   ① GameViewer 下 OPEN/OK = 左键长按拖动（mouseDown + 掌心跟随）；
                    #   ② 普通桌面 OK = SetWindowPos 拖窗口右下角缩放。
                    #   两分支整体注释，仅剩正常控鼠路径（手势标签只影响平滑档位）。
                    # ---- GameViewer：五指张开 / OK 都视为鼠标左键长按拖动 ----
                    # if _is_gv and _gesture in ("OPEN", "OK"):
                    #     #   光标跟随掌心（与正常控鼠一致），左键按住=系统原生拖拽；
                    #     #   手势离开/手离镜时释放左键。不再走移窗/缩放分支。
                    #     if not getattr(run_loop, "_gv_press", False):
                    #         pyautogui.mouseDown(button='left')
                    #         run_loop._gv_press = True
                    #         # 清掉可能残留的 OK 缩放态
                    #         if getattr(run_loop, "_ok_resize_started", False):
                    #             run_loop._ok_resize_started = False
                    #     mouse.move_only(_cursor_anchor_norm, gesture_hint=_gesture)
                    #     _pre_move_only_done_right = True
                    # ---- 非 GameViewer：OK 直接 SetWindowPos 缩放，不模拟鼠标 ----
                    # elif _gesture == "OK":
                    #     #   进入帧：选中光标下窗口、快照 rect + 掌心参考点；
                    #     #   保持帧：掌心屏幕位移 → 改窗口右下角（左/上固定），EMA 平滑；
                    #     #   离开/离镜：无鼠标键，直接复位状态。
                    #     if not getattr(run_loop, "_ok_resize_started", False):
                    #         _u32 = ctypes.windll.user32
                    #         class _OK_POINT(ctypes.Structure):
                    #             _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
                    #         _ok_pt = _OK_POINT()
                    #         _hwnd = 0
                    #         try:
                    #             if bool(_u32.GetCursorPos(ctypes.byref(_ok_pt))):
                    #                 _child = WindowFromPoint(_ok_pt)
                    #                 if _child:
                    #                     _hwnd = int(GetAncestor(
                    #                         ctypes.wintypes.HWND(int(_child)), GA_ROOT) or 0)
                    #         except Exception:
                    #             _hwnd = 0
                    #         _rc = RECT()
                    #         if not (_hwnd and bool(GetWindowRect(_hwnd, ctypes.byref(_rc)))):
                    #             _hwnd = int(GetForegroundWindow() or 0)
                    #             if not (_hwnd and bool(GetWindowRect(_hwnd, ctypes.byref(_rc)))):
                    #                 _rc = RECT()
                    #                 _rc.left = _rc.top = 0
                    #                 _rc.right = mouse.sw
                    #                 _rc.bottom = mouse.sh
                    #         run_loop._ok_hwnd = _hwnd
                    #         run_loop._ok_left0 = int(_rc.left)
                    #         run_loop._ok_top0 = int(_rc.top)
                    #         run_loop._ok_right0 = int(_rc.right)
                    #         run_loop._ok_bottom0 = int(_rc.bottom)
                    #         _sx0, _sy0 = mouse._norm_to_screen(float(_cursor_anchor_norm[0]),
                    #                                            float(_cursor_anchor_norm[1]))
                    #         run_loop._ok_ref_sx, run_loop._ok_ref_sy = _sx0, _sy0
                    #         run_loop._ok_sm_x, run_loop._ok_sm_y = _sx0, _sy0
                    #         run_loop._ok_acc_x = run_loop._ok_acc_y = 0.0
                    #         run_loop._ok_resize_started = True
                    #         try:
                    #             _tb = ctypes.create_unicode_buffer(256)
                    #             _u32.GetWindowTextW(ctypes.wintypes.HWND(_hwnd), _tb, 256)
                    #             _title = _tb.value
                    #         except Exception:
                    #             _title = "?"
                    #         print(f"[OK-RESIZE] 窗口 hwnd={_hwnd} '{_title}' "
                    #               f"rect=({_rc.left},{_rc.top},{_rc.right},{_rc.bottom}) "
                    #               f"掌心参考@({_sx0:.0f},{_sy0:.0f}) 开始缩放",
                    #               flush=True)
                    #     else:
                    #         _mx, _my = mouse._norm_to_screen(float(_cursor_anchor_norm[0]),
                    #                                          float(_cursor_anchor_norm[1]))
                    #         _OK_EMA_A = 0.30
                    #         _nsx = run_loop._ok_sm_x + _OK_EMA_A * (_mx - run_loop._ok_sm_x)
                    #         _nsy = run_loop._ok_sm_y + _OK_EMA_A * (_my - run_loop._ok_sm_y)
                    #         run_loop._ok_sm_x, run_loop._ok_sm_y = _nsx, _nsy
                    #         _dx = _nsx - run_loop._ok_ref_sx
                    #         _dy = _nsy - run_loop._ok_ref_sy
                    #         _new_right = int(round(run_loop._ok_right0 + _dx))
                    #         _new_bottom = int(round(run_loop._ok_bottom0 + _dy))
                    #         _min_w, _min_h = 160, 100
                    #         if _new_right - run_loop._ok_left0 < _min_w:
                    #             _new_right = run_loop._ok_left0 + _min_w
                    #         if _new_bottom - run_loop._ok_top0 < _min_h:
                    #             _new_bottom = run_loop._ok_top0 + _min_h
                    #         _w = _new_right - run_loop._ok_left0
                    #         _h = _new_bottom - run_loop._ok_top0
                    #         try:
                    #             SetWindowPos(
                    #                 ctypes.wintypes.HWND(run_loop._ok_hwnd),
                    #                 ctypes.wintypes.HWND(HWND_TOP),
                    #                 ctypes.wintypes.INT(run_loop._ok_left0),
                    #                 ctypes.wintypes.INT(run_loop._ok_top0),
                    #                 ctypes.wintypes.INT(_w),
                    #                 ctypes.wintypes.INT(_h),
                    #                 ctypes.wintypes.UINT(SWP_NOZORDER | SWP_NOACTIVATE),
                    #             )
                    #         except Exception:
                    #             pass
                    #     _pre_move_only_done_right = True
                    # 非 OK / 非 GV-拖动（GV左键/OK缩放态已随分支注释，无残留需清理）：正常控鼠
                    _move_hint = "OTHER" if _gesture == "FIST" else _gesture
                    mouse.move_only(_cursor_anchor_norm, gesture_hint=_move_hint)
                    _pre_move_only_done_right = True
                except Exception:
                    _pre_move_only_done_right = False

            # ======================================================================
            # Pass C：对每只手做 鼠标映射 + 画「光标锚点 绿/橙圆」
            # ======================================================================
            for label, (gesture, ext, _ratios, lms) in hand_results.items():
                # --- 鼠标事件：先移动、后判断点击 ---
                #   移动：label=="Left"（控鼠手=用户右手）每帧先 move_only()，与绿圈 1:1 对齐；
                #         单手 FIST 按中性 OTHER 跟随掌心（不再冻结光标）。
                #   点击：双手同时在镜(both_hands_global)屏蔽旧手势点击；tap 拖拽(PRESSED)期间
                #         鼠标键所有权归食指 tap 状态机，旧路径整体跳过（防每帧 up/down 被 OS
                #         判成多次双击；见 2026-09-21 长按开三窗修复）。
                if label == "Left":
                    # ↓↓ 先取锚点（画绿圈 + 移动鼠标用的**同一份**坐标，绝对保证对齐）↓↓
                    cursor_anchor_norm = HandDetector.cursor_anchor_from_landmarks(lms, args.anchor)
                    # 单手 FIST 中性化：不出点击/拖拽语义，也不冻结移动
                    move_gesture_hint = "OTHER" if gesture == "FIST" else gesture
                    # ⭐ B-1：最前置已跑过就不再重复 moveTo（仅前置异常时兜底）
                    if not _pre_move_only_done_right:
                        mouse.move_only(cursor_anchor_norm, gesture_hint=move_gesture_hint)
                    # 移窗锁定帧额外跳过 handle：OPEN 分支内部也会 moveTo，否则与窗口中心锁打架
                    # OK 手势：左键已按下抓窗口右下角，跳过 handle 避免干扰拖拽
                    if (gesture != "OK"
                            and (not _cursor_locked_by_window_move)
                            and (not both_hands_global)
                            and (tap_detector.state != _TS.PRESSED)):
                        fingertip_norm = (lms[8].x, lms[8].y)
                        mouse.handle(move_gesture_hint, fingertip_norm,
                                     cursor_anchor_norm=cursor_anchor_norm)
                    # ↓↓ 画锚点（绿圈=光标锚点 args.anchor，默认 mcp 指根；橙圈=食指TIP）
                    # ——和 move_only 用的 cursor_anchor_norm 完全同一份 ↓↓
                    # 和弦滚轮期不画绿锚点（光标冻结，绿圈跟着画圈会误导），轨迹由专用卡片绘制
                    # OK 手势：画锚点，让用户看到掌心拖动位置
                    if not _chord_scroll:
                        mx, my = HandDetector.landmark_to_pixel(lms[8], w, h)
                        cv2.circle(frame, (mx, my), 7, (0, 140, 255), 2)
                        ax = int(cursor_anchor_norm[0] * w)
                        ay = int(cursor_anchor_norm[1] * h)
                        cv2.circle(frame, (ax, ay), 12, (255, 255, 255), 2)
                        cv2.circle(frame, (ax, ay), 9, (0, 230, 80), -1)

            # ---------- 左键 Tap-Click（仅右手） ----------
            # ⭐ 2026-10-29 信号源切换：拇指-食指 TIP 捏合(pinch_bend) 替换原食指弯度(index_bend)。
            #   index_bend 保留声明(恒 None)仅为兼容下方纯观测埋点的引用，不参与任何判定；
            #   观测上有意义的引用已同步改读 pinch_bend。
            index_bend: float | None = None
            pinch_bend: float | None = None   # 拇食TIP距/掌宽（3D优先2D兜底）：高=张开 低=捏合
            thumb_middle_tip_distance: float | None = None  # 拇指TIP4-中指TIP12距/掌宽：高=张开 低=接近
            _bend_none_reason: str | None = None  # 诊断：本帧 bend=None 归因（palm=掌宽过小 / curl=食指卷曲遮蔽）；None=正常有值或无控鼠手
            _long_ev: dict | None = None          # 3D观测数据，仅用于日志，不参与手势判定
            two_finger_bend: float | None = None  # 双指右键：双指单元合成伸直度（四关节中点，0.20弯~0.88直）
            two_finger_together: bool | None = None  # 双指右键：食/中四排关节是否都并拢（None=本帧不可判定）
            two_finger_gap: float | None = None      # 双指右键：四排关节距 max（诊断/判定共用）
            two_finger_gap_parts: tuple | None = None  # (MCP,PIP,DIP,TIP) 四排指距比值，仅诊断日志用
            # ⭐ 2026-10-03 [2F-SEG] 纯观测：食/中指尖各自到 MCP5 的归一距（历史单指 bend 口径），
            #   右键段全程跟踪两者摆幅——中指摆幅≈0 而食指大摆幅即"单指点击误判右键"实锤
            rf_idx_b: float | None = None
            rf_mid_b: float | None = None
            # ⭐ 2026-10-04 [BEND3D] 纯观测：3D world 食指探针(dict) / 双指 3D 合成弯度
            _b3d_probe: dict | None = None
            rf_bend3d: float | None = None
            right_screen_pt = None
            # 🔴 只要双手同帧，整块 bend/右键量/落点评估一律跳过——用户语义：
            #   "出现双手时右手点击失效，只能用于转动翻页"（无论左手是否握拳）。
            #   tap/right_tap 检测器在下方 both_hands_global 分支每帧 reset，
            #   这里跳过评估还能避免 _eval_anchor_once 把转动中的掌心写进光标平滑器。
            #   None 变量在上方已初始化，下游均有 None 防护。
            if "Left" in hand_results and not both_hands_global:
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
                #     3) 不稳定判定（满足任一条 → palm_stable=False；⭐2026-09-21 起【不再】把 index_bend
                #        置 None：bend 95% 来自关节夹角、5% 距离项分母也是 EMA 掌宽，单帧掌宽抖动不影响 bend。
                #        palm_stable 仅用于抑制双指右键的 3D 原始掌宽判据 + 日志显示）：
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

                    # ⭐⭐ 2026-09-21 废除掌心不稳黑帧：只要关键点齐全(palm_n_raw>1e-6)就照常算 bend。
                    #   依据：bend = 95% 关节夹角(与掌宽无关) + 5% 距离项(分母 denom 用 EMA 掌宽)，
                    #   当前帧掌宽瞬时抖动(遮挡/顶部clip/透视)根本不进入 bend 数值。
                    #   旧逻辑把这些帧置 None → tap 检测器在长按中断连、恢复帧伪造下弯 → 凭空双击。
                    #   palm_stable 仍计算，但只用于：①双指右键(3D world 原始掌宽归一)抑制 ②诊断日志。
                    bend_allowed = True
                    if bend_allowed:
                        # 分母 palm_n 用 EMA 平滑后的，而不是 raw 单帧，进一步降低单帧抖动
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
                            # dist_bend_old = tip_m5 / denom
                            # index_bend = _index_bend_from_joint_angles(
                            #     lms, palm_denom=denom, dist_bend_fallback=dist_bend_old)
                            # ⭐⭐⭐ 2026-10-29 用户需求：新启手势 = 拇指TIP(4)-食指TIP(8) 捏合 → 鼠标左键
                            #   信号 pinch_bend = dist(TIP4,TIP8) / 掌宽dist(MCP5,MCP17)：
                            #     · 3D world 优先（米制距离随手旋转不变，杜绝 2D 投影前后缩短的假捏合；
                            #       与旧双指并拢判据同一坐标系口径，掌宽比有真机标定值可参照：
                            #       两指尖相触≈0.10~0.30，自然张开≈0.8~1.5+）
                            #     · world 缺失/退化 → 2D 投影兜底（分子分母同帧同缩放，比值可用）。
                            #   方向与旧 index_bend 完全同向：值高=拇食张开(待机基线 ratio≈1.0)、
                            #   值低=TIP捏合(≈0.2) → 【差分下降=按下起势、回升=回弹】语义不变，
                            #   可原样喂入同一个 tap 状态机（单击/双击间隔窗口+下弯趋势识别+
                            #   长按锁定全部复用原食指勾按方案；阈值都在 ratio=值/基线 尺度，
                            #   与信号绝对量纲无关，无需改 finger_tap_click_detector 任何参数）。
                            _wl_pinch = next((h.get("world_landmarks") for h in hands_data
                                              if h.get("label") == "Left"), None)
                            if _wl_pinch is not None and len(_wl_pinch) >= 18:
                                _p4w = np.array([_wl_pinch[4].x, _wl_pinch[4].y, _wl_pinch[4].z],
                                                dtype=np.float64)
                                _p8w = np.array([_wl_pinch[8].x, _wl_pinch[8].y, _wl_pinch[8].z],
                                                dtype=np.float64)
                                _p5w = np.array([_wl_pinch[5].x, _wl_pinch[5].y, _wl_pinch[5].z],
                                                dtype=np.float64)
                                _p17w = np.array([_wl_pinch[17].x, _wl_pinch[17].y, _wl_pinch[17].z],
                                                 dtype=np.float64)
                                _pw3 = float(np.linalg.norm(_p5w - _p17w))
                                if _pw3 > 1e-6:
                                    pinch_bend = float(np.linalg.norm(_p4w - _p8w)) / _pw3
                            if pinch_bend is None:
                                # 2D 兜底：拇食TIP距 / EMA掌宽（归一化坐标）
                                _d48 = float(np.linalg.norm(
                                    np.array([lms[4].x, lms[4].y]) - np.array([lms[8].x, lms[8].y])))
                                pinch_bend = _d48 / denom
                            # 拇指TIP(4)-中指TIP(12)距离：优先 world 3D，缺失时用 2D，统一除以掌宽。
                            if _wl_pinch is not None and len(_wl_pinch) >= 18 and _pw3 > 1e-6:
                                _p12w = np.array([_wl_pinch[12].x, _wl_pinch[12].y, _wl_pinch[12].z], dtype=np.float64)
                                thumb_middle_tip_distance = float(np.linalg.norm(_p4w - _p12w)) / _pw3
                            else:
                                _d4_12 = float(np.linalg.norm(
                                    np.array([lms[4].x, lms[4].y]) - np.array([lms[12].x, lms[12].y])))
                                thumb_middle_tip_distance = _d4_12 / denom
                            # ⭐⭐ 2026-10-29 捏合 vs 食指卷曲 鉴别门 v2（双特征 AND，真机两轮迭代）：
                            #   v1 单用 PIP 角误伤真捏合：真机日志 [TAP-SEG] 长按段内
                            #   None5帧[食指卷曲遮蔽4]——真捏合段大半帧被遮蔽 → 冻结
                            #   706ms 把"快速捏放"拖成 2109ms 转长按 → 单击/双击全灭。
                            #   （真捏合食指为够到拇指尖 PIP 也会弯到 <135°，特征区间重叠。）
                            #   v2：PIP 折叠 ∠(5,6,7) < _PINCH_CURL_PIP_DEG 且
                            #       d85 = dist(TIP8,MCP5)/掌宽 < _PINCH_CURL_D85_MAX
                            #       （指尖卷回食指根附近=卷指；对捏食指整体伸展 d85 大）
                            #       两条同时成立才遮蔽，任一不成立放行。
                            #   遮蔽 = pinch_bend 置 None（状态机冻结不产生假事件）+
                            #   [CURL] 埋点打印实测特征值（限频2s），供真机数据校准。
                            #   world 缺失的 2D 兜底帧不设防（降级容忍）。
                            if (pinch_bend is not None and _wl_pinch is not None
                                    and len(_wl_pinch) >= 18):
                                _pip_deg = _three_point_angle(
                                    np.array([_wl_pinch[5].x, _wl_pinch[5].y, _wl_pinch[5].z],
                                             dtype=np.float64),
                                    np.array([_wl_pinch[6].x, _wl_pinch[6].y, _wl_pinch[6].z],
                                             dtype=np.float64),
                                    np.array([_wl_pinch[7].x, _wl_pinch[7].y, _wl_pinch[7].z],
                                             dtype=np.float64))
                                _p5g = np.array([_wl_pinch[5].x, _wl_pinch[5].y, _wl_pinch[5].z],
                                                dtype=np.float64)
                                _p8g = np.array([_wl_pinch[8].x, _wl_pinch[8].y, _wl_pinch[8].z],
                                                dtype=np.float64)
                                _p17g = np.array([_wl_pinch[17].x, _wl_pinch[17].y, _wl_pinch[17].z],
                                                 dtype=np.float64)
                                _pw_g = float(np.linalg.norm(_p5g - _p17g))
                                _d85 = (float(np.linalg.norm(_p8g - _p5g)) / _pw_g
                                        if _pw_g > 1e-6 else 1.0)
                                if (_pip_deg < _PINCH_CURL_PIP_DEG
                                        and _d85 < _PINCH_CURL_D85_MAX):
                                    pinch_bend = None
                                    _bend_none_reason = "curl"
                                    _curl_now = time.monotonic()
                                    if _curl_now - getattr(run_loop, "_curl_dbg_last", 0.0) > 2.0:
                                        run_loop._curl_dbg_last = _curl_now
                                        print(f"[CURL] 遮蔽 PIP={_pip_deg:.0f}° "
                                              f"d85={_d85:.2f} "
                                              f"(门: PIP<{_PINCH_CURL_PIP_DEG:.0f}° 且 "
                                              f"d85<{_PINCH_CURL_D85_MAX})",
                                              flush=True)
                            # 原"握拳卷曲全局 guard"（r 门/近节角状态机/world 缺失宁拦/
                            #   bend 置 None/cancel/force_long_up）已整体拆除——单手握拳无任何语义。
                            #   握拳唯一防线在 tap 检测器长按转换点的 3D 正向门（_long_ev，本帧稍后
                            #   取 world 计算）；index_bend 此后永远是正常算出来的食指弯度，不再被置空。
                            # ----------------------------------------------------------
                            # ⭐⭐ 2026-10-29 用户需求：双指点击(食中并拢右键)逻辑整体注释。
                            #   下方 two_finger_* / rf_idx_b / rf_mid_b / _b3d_probe / rf_bend3d
                            #   全部保持 None（本帧开头已初始化），下游观测埋点按 None 跳过，
                            #   right_tap_detector 不再被 update（见下方检测器仲裁块）。
                            # ⭐ 2026-09-17 食中双指并拢右键 v2：不再分别算食指/中指 bend
                            #   用户要求：前置条件已是并拢 → 食中两指对应关节(PIP/PIP, DIP/DIP, TIP/TIP)
                            #   应保持在容忍距离内；弯曲度只算"双指单元"一个合成值（四关节中点代入同公式）。
                            #   同步性由四排关节距离直接保证：一指先直 → TIP 排距离超阈 → together 失效 → 作废。
                            # ----------------------------------------------------------
                            # try:
                            #     _p5  = np.array([lms[5].x,  lms[5].y],  dtype=np.float64)
                            #     _p9  = np.array([lms[9].x,  lms[9].y],  dtype=np.float64)
                            #     _p6  = np.array([lms[6].x,  lms[6].y],  dtype=np.float64)
                            #     _p10 = np.array([lms[10].x, lms[10].y], dtype=np.float64)
                            #     _p7  = np.array([lms[7].x,  lms[7].y],  dtype=np.float64)
                            #     _p11 = np.array([lms[11].x, lms[11].y], dtype=np.float64)
                            #     _p8  = np.array([lms[8].x,  lms[8].y],  dtype=np.float64)
                            #     _p12 = np.array([lms[12].x, lms[12].y], dtype=np.float64)
                            #
                            #     # 四排食-中指距（归一化到掌心宽）：MCP/PIP/DIP/TIP
                            #     _g_mcp = float(np.linalg.norm(_p5  - _p9))
                            #     _g_pip = float(np.linalg.norm(_p6  - _p10))
                            #     _g_dip = float(np.linalg.norm(_p7  - _p11))
                            #     _g_tip = float(np.linalg.norm(_p8  - _p12))
                            #
                            #     # ⭐ 并拢判据（2026-09-17 v7，分阶段阈值，全部 3D 米制 world 坐标）：
                            #     _wlms = hand.get("world_landmarks")
                            #     _gap_space = "3D"
                            #     _w = None   # 2026-10-04 [BEND3D]：3D 分支构建 (21,3) 数组，供探针复用
                            #     if _wlms is not None and len(_wlms) >= 18:
                            #         _w = np.array([[float(p.x), float(p.y), float(p.z)]
                            #                        for p in _wlms], dtype=np.float64)
                            #         def _wd3(a: int, b: int) -> float:
                            #             return float(np.linalg.norm(_w[a] - _w[b]))
                            #         _wden = _wd3(5, 17) or 1e-6          # 米制掌宽（与 2D denom 同语义点对）
                            #         _r_mcp  = _wd3(5, 9)  / _wden
                            #         _r_pip  = _wd3(6, 10) / _wden
                            #         _r_dip  = _wd3(7, 11) / _wden
                            #         _r_tip  = _wd3(8, 12) / _wden
                            #     else:
                            #         # 兜底：老检测器无 world 点时退回 2D（2D 下 MCP/PIP 最稳，用指根）
                            #         _gap_space = "2D!"
                            #         _r_mcp, _r_pip = _g_mcp / denom, _g_pip / denom
                            #         _r_dip, _r_tip = _r_pip, _r_pip
                            #     _sync_spread = _r_tip - _r_pip
                            #     two_finger_gap = float(_r_tip)
                            #     two_finger_gap_parts = (round(_r_mcp, 3), round(_r_pip, 3),
                            #                             round(_r_dip, 3), round(_r_tip, 3),
                            #                             round(_sync_spread, 3))
                            #     # 起势口径只看 TIP；手势中 DIP 门控在检测器内部（TWO_FINGER_DIP_HOLD_R）
                            #     two_finger_together = bool(_r_tip <= TWO_FINGER_TIP_ENTRY_R)
                            #     run_loop._cached_two_finger_gap = round(float(_r_tip), 3)
                            #     run_loop._cached_two_finger_gap_space = _gap_space
                            #
                            #     # 双指单元合成 bend：四关节取中点，代入与单指完全相同的 _bend_from_points
                            #     _mcp_mid = (_p5  + _p9)  * 0.5
                            #     _pip_mid = (_p6  + _p10) * 0.5
                            #     _dip_mid = (_p7  + _p11) * 0.5
                            #     _tip_mid = (_p8  + _p12) * 0.5
                            #     _p0 = np.array([lms[0].x, lms[0].y], dtype=np.float64)
                            #     _dist_fb = float(np.linalg.norm(_tip_mid - _mcp_mid)) / denom
                            #     two_finger_bend = _bend_from_points(
                            #         _p0, _mcp_mid, _pip_mid, _dip_mid, _tip_mid,
                            #         palm_denom=denom, dist_bend_fallback=_dist_fb)
                            #     # ⭐ 2026-10-03 [2F] 食/中指【独立关节角弯度】（与左键同公式，
                            #     #   0.20弯~0.88直）：右键双门用——资格帧两指须近直(≥0.80)、
                            #     rf_idx_b = _index_bend_from_joint_angles(
                            #         lms, palm_denom=denom, joint_ids=(5, 6, 7, 8))
                            #     rf_mid_b = _index_bend_from_joint_angles(
                            #         lms, palm_denom=denom, joint_ids=(9, 10, 11, 12))
                            #
                            #     # ---- ⭐ 2026-10-04 [BEND3D] 纯观测探针（不参与任何判定）----
                            #     if _w is not None and len(_w) >= 21:
                            #         _b3d_probe = _bend3d_probe(lms, _wlms, palm_n_raw)
                            #     if _b3d_probe is not None:
                            #         _prev_tip3 = getattr(run_loop, "_b3d_prev_tip3", None)
                            #         if _prev_tip3 is not None:
                            #             _b3d_probe["jump_mm"] = float(
                            #                 np.linalg.norm(_b3d_probe["tip3"] - _prev_tip3)
                            #                 * 1000.0)
                            #         else:
                            #             _b3d_probe["jump_mm"] = None
                            #         run_loop._b3d_prev_tip3 = _b3d_probe["tip3"]
                            #         # 双指单元 3D 合成弯度（食/中对应关节 3D 中点，同映射）
                            #         rf_bend3d = _bend_from_points(
                            #             _w[0],
                            #             (_w[5] + _w[9]) * 0.5,
                            #             (_w[6] + _w[10]) * 0.5,
                            #             (_w[7] + _w[11]) * 0.5,
                            #             (_w[8] + _w[12]) * 0.5,
                            #             palm_denom=None, dist_bend_fallback=None)
                            # except Exception:
                            #     two_finger_bend = None
                            #     two_finger_together = None
                            #     two_finger_gap = None
                            #     two_finger_gap_parts = None
                            #     rf_idx_b = None
                            #     rf_mid_b = None
                            #     _b3d_probe = None
                            #     rf_bend3d = None
                            # if not palm_stable:
                            #     # ⭐ 食指 bend 已放开（关节角主导）；但双指并拢判据的四排指距用 3D world
                            #     #   【原始掌宽】归一，掌宽不稳帧不可信 → 沿用旧策略不给右键检测器喂数据，
                            #     two_finger_bend = None
                            #     two_finger_together = None
                            #     two_finger_gap = None
                            #     two_finger_gap_parts = None
                            #     rf_idx_b = None
                            #     rf_mid_b = None
                else:
                    # palm_n_raw 太小（手离镜头特别远/截断）→ 直接丢弃
                    index_bend = None
                    pinch_bend = None
                    _bend_none_reason = "palm"
                    two_finger_bend = None
                    two_finger_together = None
                    two_finger_gap = None
                    two_finger_gap_parts = None
                    rf_idx_b = None
                    rf_mid_b = None

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

                # ---- 3D 姿态观测（仅用于日志，不参与长按、点击或释放判定） ----
                #   world 缺失时不影响长按；检测器只按捏合持续时间转换长按。
                _wl = next((h.get("world_landmarks") for h in hands_data
                            if h.get("label") == "Left"), None)
                if _wl is not None and len(_wl) >= 21:
                    _wpts = [np.array([float(p.x), float(p.y), float(p.z)], dtype=np.float64)
                             for p in _wl]
                    _mcp_ang = _four_finger_mcp_angles(_wpts)
                    if _mcp_ang is not None:
                        _long_ev = {"mcp_idx_deg": float(_mcp_ang[0])}
                if _b3d_probe is not None:
                    if _long_ev is None:
                        _long_ev = {}
                    _long_ev["bend3d"] = float(_b3d_probe["bend"])
                    _long_ev["dip3d"] = float(_b3d_probe["dip"])

            # 快路径双击(DOUBLE_TAP)挂起的首击落点（2026-09-19 位置门控，见下方 WAIT_DBL 捕获）
            _pending_first_xy = None

            # ---------- 丢手诊断：逐帧归因 bend=None（纯计数，不影响任何判定） ----------
            # ⭐ 2026-10-29 起归因对象 = pinch_bend（捏合信号 None 的原因与旧 index_bend 同构：
            #   无控鼠手 / 掌宽过小）
            _det_rs["frames"] += 1
            if _skip_was_skipped:
                _det_rs["skip"] += 1
            if pinch_bend is None:
                if "Left" not in hand_results:
                    _det_rs["no_left"] += 1
                elif _bend_none_reason == "palm":
                    _det_rs["palm"] += 1
                elif _bend_none_reason == "curl":
                    _det_rs["curl"] += 1
                else:
                    _det_rs["other"] += 1
            # ⭐ 2026-10-29 [DET-STAT]/[MOUSE-STAT] 检测层/鼠标平滑统计日志——
            #   与捏合手势观察无关，按用户要求注释（整块含计数重置一并注释；
            #   上方每帧的 _det_rs 累计保留，纯内存计数无输出）。
            # if time.monotonic() - _det_stat_t0 >= _DET_STAT_SEC:
            #     _dd = detector.take_det_diag()
            #     _n = max(1, _det_rs["frames"])
            #     _ni = max(1, _dd["infer"])
            #     _none_tot = (_det_rs["no_left"] + _det_rs["palm"]
            #                  + _det_rs["curl"] + _det_rs["other"])
            #     _b_avg = _det_rs["b_sum"] / _n
            #     _b_min = _det_rs["b_min"]
            #     print(
            #         f"[DET-STAT] 循环帧{_det_rs['frames']} 跳帧{_det_rs['skip']}"
            #         f"({100.0*_det_rs['skip']/_n:.0f}%) | "
            #         f"推理{_dd['infer']} 超时{_dd['timeout']}"
            #         f"({100.0*_dd['timeout']/_ni:.0f}%) 空手{_dd['empty']}"
            #         f"({100.0*_dd['empty']/_ni:.0f}%) 有手{_dd['hand1']} | "
            #         f"逆光增强{_dd['enh']}帧 暗占max{100.0*_dd['dark_max']:.0f}% | "
            #         f"等待avg{_dd['lat_avg_ms']}/p90{_dd['lat_p90_ms']}ms | "
            #         f"bend=None共{_none_tot}({100.0*_none_tot/_n:.0f}%): "
            #         f"无控鼠手{_det_rs['no_left']} "
            #         f"掌宽过小{_det_rs['palm']} "
            #         f"卷指遮蔽{_det_rs['curl']} "
            #         f"其他{_det_rs['other']} | "
            #         f"黑帧{_det_rs['black']}(最长连{_det_rs['black_run_max']}) 亮度min"
            #         f"{_b_min if _b_min is not None else 0:.0f}/avg{_b_avg:.0f} | "
            #         f"读帧max{_det_rs['read_max_ms']:.0f}ms/慢帧{_det_rs['read_slow']} "
            #         f"waitKeyMax{_det_rs['wk_max_ms']:.0f}ms",
            #         flush=True)
            #     _det_stat_t0 = time.monotonic()
            #     _det_rs = {"frames": 0, "skip": 0,
            #                "no_left": 0, "palm": 0, "curl": 0, "other": 0,
            #                "black": 0, "b_sum": 0.0, "b_min": None,
            #                "black_run": 0, "black_run_max": 0,
            #                "read_max_ms": 0.0, "read_slow": 0, "wk_max_ms": 0.0}
            #     # ---- 鼠标微震基线：原始锚点位移 vs 平滑后输出位移 + 静止锁进出统计 ----
            #     _md = mouse._smoother.take_mouse_diag()
            #     print(
            #         f"[MOUSE-STAT] 有手帧{_md['frames']} 锁定{_md['locked_pct']:.0f}% "
            #         f"入锁{_md['enters']}次 解锁{_md['unlocks']}次 | "
            #         f"原始位移px p50/{_md['raw_p50']} p90/{_md['raw_p90']} max/{_md['raw_max']} | "
            #         f"输出位移px p50/{_md['out_p50']} p90/{_md['out_p90']} max/{_md['out_max']} | "
            #         f"解锁时 drift_max/{_md['unl_dist_max']} p90/{_md['unl_dist_p90']} "
            #         f"3帧漂移max/{_md['unl_trend_max']}(阈8.96) σavg/{_md['unl_std_avg']}(进锁阈5.6)",
            #         flush=True)

            _tap_updated = False
            # OK 手势：鼠标锁定右下角，食指弯曲不触发点击（与双手同帧一样 reset 检测器）
            # ⭐ 2026-10-29 OK 手势识别已在 gesture_recognizer 注释 → 本分支不再存在；
            #   仅保留【双手同帧】reset 语义（用户既定规则：双手在镜时右手只能翻页，不出点击）。
            # _right_gesture = hand_results.get("Left", (None,))[0]
            # _ok_gesture_active = (_right_gesture == "OK")
            if both_hands_global:
                tap_detector.reset()
                right_tap_detector.reset()
                tap_event = 0
                right_tap_event = 0
            else:
                # 新右键：拇指TIP4与中指TIP12的距离变化，沿用左键差分/回弹状态机。
                right_tap_event = right_tap_detector.update(thumb_middle_tip_distance)
                # 右键链路诊断：确认输入距离、差分门和状态机是否真的收到数据。
                if "Left" in hand_results:
                    _rt_now = time.monotonic()
                    if _rt_now - getattr(run_loop, "_right_dbg_t", 0.0) >= 0.5:
                        run_loop._right_dbg_t = _rt_now
                        _rm = right_tap_detector.metrics()
                        print(
                            f"[RIGHT-DBG] input={None if thumb_middle_tip_distance is None else round(float(thumb_middle_tip_distance), 3)} "
                            f"raw={_rm['raw']} base={_rm['baseline']} ratio={_rm['ratio']} "
                            f"Σ下={_rm['di_sum_down']:.3f} state={_rm['state']} event={right_tap_event}",
                            flush=True)
                # 左键仍独立吃拇食捏合信号；右键进行中不冻结左键。
                #   状态机/阈值/双击窗/趋势/长按锁全部复用原食指勾按方案，一条不改：
                #     · 单击 = 捏合快速回弹（松开=绝对范围回升≥PINCH_RELEASE_ABS；
                #              段内raw谷底≤PINCH_ABS_MAX=指尖真捏到一起，超过范围整段取消）
                #     · 双击 = 0.38s 窗口内完成第二击；趋势探测只确认二次下弯意图，
                #              最终由程序直接发送原子 doubleClick，不交给 Windows 拼接
                #     · 长按 = 捏合保持 ≥0.733s 后锁定拖拽，不读取 3D 角度门，
                #              拇食重新张开（连续越过松开线）才释放
                #   第二参 _long_ev 仅在检测器内部门 LONG_DOWN，点击链路不读。
                tap_event = tap_detector.update(pinch_bend, _long_ev)
                _tap_updated = True
                #   首击回弹进入 WAIT_DBL 时记录落点，供原子 DOUBLE_TAP 使用首击坐标；
                #   事件本身已由检测器完成0.50s窗口配对，不再依赖 Windows 双击时间/位置判断
                _pending_first_xy = getattr(run_loop, "_first_tap_xy", None)
                if tap_detector.state == _TS.WAIT_DBL:
                    if _pending_first_xy is None:
                        run_loop._first_tap_xy = right_screen_pt
                else:
                    run_loop._first_tap_xy = None
                # ⭐ 2026-10-29 [PINCH-DBG] 捏合链路诊断（纯观测，0.5s 限频，零行为）：
                #   raw=原始pinch_bend(TIP4-TIP8距/掌宽) base=张开基线
                #   ratio=EMA后比值(起弯门≤0.75) Σ下=近窗累计下弯差分(起弯门≥DI_DOWN_THR)
                #   谷raw=段内raw谷底(点击有效须≤PINCH_ABS_MAX) 松开线=PINCH_RELEASE_ABS
                #   st=状态机(0IDLE/1ARM/2HOLD/3长按/4冷却/6等双击/7补救)
                #   定位卡点：raw=None→卷指门/palm丢弃；ratio压不到0.75→起弯门未过；
                #   谷raw>0.40→指尖没捏到一起(段将被"未捏合取消")；st=2且raw<0.60→仍在点击范围内等松开。
                if "Left" in hand_results:
                    _pd_now = time.monotonic()
                    if _pd_now - getattr(run_loop, "_pinch_dbg_t", 0.0) >= 0.5:
                        run_loop._pinch_dbg_t = _pd_now
                        _pm = tap_detector.metrics()
                        _pd_raw = (pinch_bend if pinch_bend is None
                                   else round(float(pinch_bend), 3))
                        _pd_none = (f" None原因:{_bend_none_reason}"
                                    if pinch_bend is None else "")
                        print(f"[PINCH-DBG] raw={_pd_raw} base={_pm['baseline']} "
                              f"ratio={_pm['bend_ratio']} "
                              f"Σ下={_pm['di_sum_down']:.3f}/{_pm['DI_DOWN_THR']:.3f} "
                              f"谷raw={_pm['seg_min_raw']}(门{PINCH_ABS_MAX}) "
                              f"st={_pm['state']}{_pd_none}",
                              flush=True)

            # -------- 2026-10-04 [BEND3D-JIT] 悬停双口径噪声（纯观测，约每 2.0s 一行） --------
            #   回答"3D 的 z 抖动会不会冲过 2.5% 差分噪声门"：IDLE 悬停帧 2D/3D 弯度 mean±std。
            #   手丢/非 IDLE/握拳帧不入队且清空计时，避免跨动作污染。
            # ⭐ 2026-10-29 [BEND3D-JIT] 悬停双口径噪声观测日志——
            #   与捏合手势观察无关，按用户要求注释（_b3d_probe 现恒 None，本块本就不再输出）。
            # if _b3d_probe is None:
            #     run_loop._b3d_prev_tip3 = None
            # if ((not both_hands_global)
            #         and tap_detector.state == _TS.IDLE
            #         and index_bend is not None and _b3d_probe is not None):
            #     _jit3d_q.append((float(index_bend), float(_b3d_probe["bend"]),
            #                      _b3d_probe["comp"]))
            #     _jit3d_sec += _frame_dt
            #     if _jit3d_sec >= 2.0 and len(_jit3d_q) >= 8:
            #         _a2 = np.array([_q[0] for _q in _jit3d_q], dtype=np.float64)
            #         _a3 = np.array([_q[1] for _q in _jit3d_q], dtype=np.float64)
            #         _ac = np.array([_q[2] for _q in _jit3d_q if _q[2] is not None],
            #                        dtype=np.float64)
            #         _cs = (f" comp中位{float(np.median(_ac)):.2f}/min{float(_ac.min()):.2f}"
            #                if len(_ac) else "")
            #         print(f"[BEND3D-JIT] {_jit3d_sec:.1f}s n={len(_jit3d_q)} "
            #               f"2D弯 {_a2.mean():.3f}±{_a2.std():.4f} | "
            #               f"3D弯 {_a3.mean():.3f}±{_a3.std():.4f} | "
            #               f"均值差2D-3D{(_a2.mean() - _a3.mean()):+.3f}{_cs}",
            #               flush=True)
            #         _jit3d_q.clear()
            #         _jit3d_sec = 0.0
            # else:
            #     if _jit3d_q:
            #         _jit3d_q.clear()
            #     _jit3d_sec = 0.0

            # -------- 2026-10-03 [TAP-SEG] 弯曲段生命周期跟踪（纯观测埋点） --------
            # 目的：标题栏慢点击/重复尝试时，看见失败段（尖峰/浅弯/抢占）的完整过程，
            #   并把 ARM 瞬间坐标 vs 段末坐标的漂移、段内 None 归因打出来。
            if _tap_updated and tap_detector.armed_this_frame:
                _seg_arm_xy = right_screen_pt
                _seg_none_rs = {}
                # ⭐ 2026-10-04 [2F-OBS] 左键段起势快照：两指尖距比值 + 中指独立弯度
                _lseg_tip_arm = two_finger_gap_parts[3] if two_finger_gap_parts else None
                _lseg_tip_min = _lseg_tip_arm
                _lseg_mid_arm = rf_mid_b
                _lseg_mid_min = _lseg_mid_max = rf_mid_b
                # ⭐ 2026-10-04 [BEND3D] 起势快照（2D 弯度 + 3D 探针全套）
                _lseg_b2d_arm = index_bend
                _lseg_mcp_max = (float(_long_ev["mcp_idx_deg"])
                                 if isinstance(_long_ev, dict)
                                 and _long_ev.get("mcp_idx_deg") is not None else None)
                if _b3d_probe is not None:
                    _lseg_b3d_arm = _b3d_probe["bend"]
                    _lseg_b3d_min = _b3d_probe["bend"]
                    _lseg_comp_min = _b3d_probe["comp"]
                    _lseg_dip3d_min = _b3d_probe["dip"]
                    _lseg_diff_max = None
                    _lseg_diff_comp = None
                    _lseg_jump_max = _b3d_probe.get("jump_mm")
                    _lseg_phal = _b3d_probe["phal_mm"]
                else:
                    _lseg_b3d_arm = None
                    _lseg_b3d_min = _lseg_comp_min = _lseg_dip3d_min = None
                    _lseg_diff_max = _lseg_diff_comp = None
                    _lseg_jump_max = _lseg_phal = None
            # 左键段进行中：累计 TIP 段内最小值与中弯摆幅（数据缺失帧跳过）
            if (_tap_updated
                    and tap_detector.state in (_TS.BEND_ARMED, _TS.BEND_HOLD, _TS.PRESSED)):
                if two_finger_gap_parts is not None:
                    _ltv = two_finger_gap_parts[3]
                    if _ltv is not None:
                        _lseg_tip_min = (float(_ltv) if _lseg_tip_min is None
                                         else min(_lseg_tip_min, float(_ltv)))
                if rf_mid_b is not None:
                    _lseg_mid_min = (float(rf_mid_b) if _lseg_mid_min is None
                                     else min(_lseg_mid_min, float(rf_mid_b)))
                    _lseg_mid_max = (float(rf_mid_b) if _lseg_mid_max is None
                                     else max(_lseg_mid_max, float(rf_mid_b)))
                # ⭐ 2026-10-15 [LONG3D] 食近节角段内 max（握拳角变大；max 越接近/超过
                #   LONG_GATE_MCP_IDX_DEG 说明段内出现过握拳构型）
                if isinstance(_long_ev, dict) and _long_ev.get("mcp_idx_deg") is not None:
                    _mv = float(_long_ev["mcp_idx_deg"])
                    _lseg_mcp_max = _mv if _lseg_mcp_max is None else max(_lseg_mcp_max, _mv)
                # ⭐ 2026-10-04 [BEND3D] 段内累计：3D 谷底/3D DIP 角谷底/压缩 min/
                #   同帧 2D-3D 弯度差 max（正值=3D 看到了 2D 没看到的弯）/TIP 跳变 max
                if _b3d_probe is not None:
                    _bv = float(_b3d_probe["bend"])
                    _lseg_b3d_min = _bv if _lseg_b3d_min is None else min(_lseg_b3d_min, _bv)
                    _lseg_dip3d_min = (_b3d_probe["dip"] if _lseg_dip3d_min is None
                                       else min(_lseg_dip3d_min, _b3d_probe["dip"]))
                    if _b3d_probe["comp"] is not None:
                        _cv = float(_b3d_probe["comp"])
                        _lseg_comp_min = _cv if _lseg_comp_min is None else min(_lseg_comp_min, _cv)
                    if index_bend is not None:
                        _df = float(index_bend) - _bv
                        if _lseg_diff_max is None or _df > _lseg_diff_max:
                            _lseg_diff_max = _df
                            _lseg_diff_comp = _b3d_probe["comp"]
                    _jm = _b3d_probe.get("jump_mm")
                    if _jm is not None:
                        _lseg_jump_max = _jm if _lseg_jump_max is None else max(_lseg_jump_max, _jm)
                    _lseg_phal = _b3d_probe["phal_mm"]
            # ⭐ 2026-10-29 归因对象随信号源切换为 pinch_bend（原 index_bend 恒 None，沿用会段段误计）
            if (_tap_updated and pinch_bend is None
                    and tap_detector.state in (_TS.BEND_ARMED, _TS.BEND_HOLD, _TS.PRESSED)):
                if "Left" not in hand_results:
                    _nr = "无控鼠手"
                else:
                    _nr = {"palm": "掌宽过小",
                           "curl": "食指卷曲遮蔽"}.get(_bend_none_reason or "", "其他")
                _seg_none_rs[_nr] = _seg_none_rs.get(_nr, 0) + 1
            for _sr in tap_detector.take_seg_logs():
                try:
                    _sh = max(1, int(getattr(mouse, "sh", 1080)))
                    if right_screen_pt is not None:
                        _ex, _ey = right_screen_pt
                        _ez = ("顶部标题栏" if _ey / _sh < 0.06
                               else ("底部任务栏" if _ey / _sh > 0.94 else "中部内容区"))
                        _ep = f"({int(_ex)},{int(_ey)})"
                        if _seg_arm_xy is not None:
                            _drift = _mt.hypot(_ex - _seg_arm_xy[0], _ey - _seg_arm_xy[1])
                            _ap = f"ARM({int(_seg_arm_xy[0])},{int(_seg_arm_xy[1])})→"
                        else:
                            _drift = -1.0
                            _ap = ""
                    else:
                        _ez, _ep, _drift, _ap = "-", "无点", -1.0, ""
                    _nrs = "".join(f"{k}{v} " for k, v in sorted(_seg_none_rs.items())).strip()
                    _gates = ""
                    if _sr.get("ratio") is not None:
                        # ⭐ 2026-10-29 捏合口径：raw谷底门(≤PINCH_ABS_MAX=真捏到一起)+绝对松开线
                        _gates = (f" 谷底{_sr['seg_min']} raw谷底{_sr.get('seg_min_raw')}"
                                  f"(捏合门{PINCH_ABS_MAX}) 末ratio{_sr['ratio']}"
                                  f"(松开线{PINCH_RELEASE_ABS})")
                    _fz = f" 冻结{_sr['freeze_ms']}ms" if _sr.get("none_n") else ""
                    _dr = f" 漂{_drift:.0f}px" if _drift >= 0 else ""
                    # ⭐ 2026-10-04 [2F-OBS] 附本段起势两指尖距/段内 min/中弯摆幅（单指 vs 双指分组依据）
                    def _lf(_v):
                        return f"{_v:.2f}" if isinstance(_v, (int, float)) else "n/a"
                    _swm = ((_lseg_mid_max - _lseg_mid_min)
                            if _lseg_mid_min is not None and _lseg_mid_max is not None
                            else None)
                    _lobs = (f" | TIP起{_lf(_lseg_tip_arm)}/段内min{_lf(_lseg_tip_min)}"
                             f" 中弯起{_lf(_lseg_mid_arm)}/摆{_lf(_swm)}")
                    # ⭐ 2026-10-04 [BEND3D] 双口径对比：3D 起/谷 vs 2D 谷、3D DIP 角、
                    #   压缩 min、2D-3D 差 max(及该帧压缩比)、末节骨长、TIP 跳变 max
                    # ⭐ 2026-10-15 [LONG3D] 食近节角段内 max（长按门 53° 标定观测）
                    _lobs3d = ""
                    if _lseg_mcp_max is not None:
                        _lobs3d = f" | 食近节角段max{_lseg_mcp_max:.0f}°(长按门53)"
                    if _lseg_b3d_arm is not None:
                        _lobs3d += (f" 3D起{_lf(_lseg_b3d_arm)}/谷{_lf(_lseg_b3d_min)}"
                                   f"(2D起{_lf(_lseg_b2d_arm)}/谷{_lf(_sr.get('seg_min'))})"
                                   f" DIP3D谷{_lf(_lseg_dip3d_min)}°"
                                   f" 压缩min{_lf(_lseg_comp_min)}"
                                   f" 2D-3D差max{_lf(_lseg_diff_max)}@压{_lf(_lseg_diff_comp)}"
                                   f" 末节{_lf(_lseg_phal)}mm 跳max{_lf(_lseg_jump_max)}mm")
                    print(
                        f"[TAP-SEG] {_sr['result']}({_sr['ctx']}) 区={_ez} "
                        f"段{_sr['wall_ms']}ms{_fz} None{_sr['none_n']}帧"
                        f"{('[' + _nrs + ']') if _nrs else ''} "
                        f"{_ap}末点{_ep}{_dr}{_gates}{_lobs}{_lobs3d}",
                        flush=True)
                except Exception:
                    pass

            # -------- 2026-10-04 [2F-ENTRY]/[2F-SEG] 右键检测器埋点（纯观测，零判定变更） --------
            #   起势资格：TIP≤0.50 掌宽比 + 合成 bend≥0.55 + 食/中独立弯度都≥0.80 连续 0.11s；
            #   段终点：合成 r 全程 + 食/中独立弯度摆幅 + TIP 段内谷底。
            #   注：10-03 同弯门已被真机数据证伪并删除（真右键中弯摆幅仅 0.02~0.07），
            #   10-04 起 ARM 后持续校验指尖距（TWO_FINGER_TIP_HOLD_R=0.60，结局 *_TIP*作废）。
            #   另有检测器本体直打 [2F-ARM!]/[2F-END!]，不受本块 try/except 影响。
            # ⭐ 2026-10-30 [2F-ENTRY]/[2F-SEG] 右键观测埋点日志——双指右键逻辑已整体注释
            #   （right_tap_detector 不再 update，恒 IDLE，本块永不触发），且与捏合手势
            #   观察无关，按用户要求注释。_rf_* 初始化变量保留（无引用、无输出）。
            # def _b2f(_v):
            #     return f"{_v:.2f}" if isinstance(_v, (int, float)) else "n/a"
            # try:
            #     # 每有效帧滚动记录 TIP 实测值（ARM 前 12 帧≈0.66s 轨迹，纯观测）
            #     if (not both_hands_global
            #             and two_finger_gap_parts is not None
            #             and len(two_finger_gap_parts) >= 4):
            #         _sp = getattr(run_loop, "_cached_two_finger_gap_space", "3D")
            #         _rf_pre_tip.append((float(two_finger_gap_parts[3]), _sp))
            #         if len(_rf_pre_tip) > 12:
            #             _rf_pre_tip.pop(0)
            #     if right_tap_detector.armed_this_frame:
            #         _rf_arm_xy = right_screen_pt
            #         _rf_arm_idx, _rf_arm_mid = rf_idx_b, rf_mid_b
            #         _rf_idx_min = _rf_idx_max = rf_idx_b
            #         _rf_mid_min = _rf_mid_max = rf_mid_b
            #         _rf_arm_pretip = list(_rf_pre_tip)
            #         # ⭐ 2026-10-04 [BEND3D] 右键段 3D 合成弯度快照
            #         _rf_b3d_arm = rf_bend3d
            #         _rf_b3d_min = rf_bend3d
            #         if two_finger_gap_parts is not None and len(two_finger_gap_parts) >= 4:
            #             _rf_tip_min = _rf_tip_max = float(two_finger_gap_parts[3])
            #             _rf_dip_min = _rf_dip_max = float(two_finger_gap_parts[2])
            #         else:
            #             _rf_tip_min = _rf_tip_max = None
            #             _rf_dip_min = _rf_dip_max = None
            #     elif right_tap_detector.is_active and not both_hands_global:
            #         if rf_idx_b is not None:
            #             _rf_idx_min = rf_idx_b if _rf_idx_min is None else min(_rf_idx_min, rf_idx_b)
            #             _rf_idx_max = rf_idx_b if _rf_idx_max is None else max(_rf_idx_max, rf_idx_b)
            #         if rf_mid_b is not None:
            #             _rf_mid_min = rf_mid_b if _rf_mid_min is None else min(_rf_mid_min, rf_mid_b)
            #             _rf_mid_max = rf_mid_b if _rf_mid_max is None else max(_rf_mid_max, rf_mid_b)
            #         if two_finger_gap_parts is not None and len(two_finger_gap_parts) >= 4:
            #             _tv, _dv = float(two_finger_gap_parts[3]), float(two_finger_gap_parts[2])
            #             _rf_tip_min = _tv if _rf_tip_min is None else min(_rf_tip_min, _tv)
            #             _rf_tip_max = _tv if _rf_tip_max is None else max(_rf_tip_max, _tv)
            #             _rf_dip_min = _dv if _rf_dip_min is None else min(_rf_dip_min, _dv)
            #             _rf_dip_max = _dv if _rf_dip_max is None else max(_rf_dip_max, _dv)
            #         # ⭐ 2026-10-04 [BEND3D] 右键段中累计 3D 合成弯度谷底
            #         if rf_bend3d is not None:
            #             _rf_b3d_min = (rf_bend3d if _rf_b3d_min is None
            #                            else min(_rf_b3d_min, rf_bend3d))
            # except Exception:
            #     pass
            # for _rl in right_tap_detector.take_rf_logs():
            #     try:
            #         _sh = max(1, int(getattr(mouse, "sh", 1080)))
            #         _pt0 = right_screen_pt
            #         _rz = "-"
            #         if _pt0 is not None:
            #             _rx, _ry = _pt0
            #             _rz = ("顶部标题栏" if _ry / _sh < 0.06
            #                    else ("底部任务栏" if _ry / _sh > 0.94 else "中部内容区"))
            #         _gs = getattr(run_loop, "_cached_two_finger_gap_space", "?")
            #         _parts = _rl.get("parts")
            #         if _parts and len(_parts) >= 4:
            #             _pstr = (f"MCP{_parts[0]:.2f} PIP{_parts[1]:.2f} "
            #                      f"DIP{_parts[2]:.2f}(门1.30) TIP{_parts[3]:.2f}")
            #         else:
            #             _pstr = "parts n/a"
            #         _xy = f"({int(_rx)},{int(_ry)})" if _pt0 is not None else "无点"
            #         _arm_xy_s = (f"({int(_rf_arm_xy[0])},{int(_rf_arm_xy[1])})"
            #                      if _rf_arm_xy is not None else "无")
            #         if _rl.get("result") == "起势资格":
            #             print(
            #                 f"[2F-ENTRY] 资格取得 区={_rz} 空间={_gs} @{_xy} "
            #                 f"TIP{_b2f(_rl.get('entry_tip'))}(门0.50) "
            #                 f"合成bend{_b2f(_rl.get('entry_bend'))}(门0.55) {_pstr} | "
            #                 f"食弯{_b2f(rf_idx_b)} 中弯{_b2f(rf_mid_b)}(资格直门0.80)",
            #                 flush=True)
            #             continue
            #         _sw_i = (_rf_idx_max - _rf_idx_min
            #                  if _rf_idx_min is not None and _rf_idx_max is not None else None)
            #         _sw_m = (_rf_mid_max - _rf_mid_min
            #                  if _rf_mid_min is not None and _rf_mid_max is not None else None)
            #         _extra = ""
            #         if _rl.get("total") is not None:
            #             _extra = (f" 总{_rl['total']:.2f}s 保持{_rl.get('hold', 0):.2f}s"
            #                       f" Σ↑{_rl.get('di_up', 0):.2f}")
            #         if _rl.get("dip") is not None:
            #             _extra += f" 解散DIP{_rl['dip']:.2f}"
            #         if _rl.get("idx") is not None or _rl.get("mid") is not None:
            #             _extra += (f" 解散帧食弯{_b2f(_rl.get('idx'))}"
            #                        f" 中弯{_b2f(_rl.get('mid'))}(弯门0.80)")
            #         # ARM 前滚动窗 TIP 轨迹（! = 该帧走 2D 兜底，拿 PIP 冒充 TIP）
            #         _pre_vals = ",".join(f"{_v:.2f}{'!' if _s == '2D!' else ''}"
            #                              for _v, _s in _rf_arm_pretip) or "无"
            #         _pre_2d_n = sum(1 for _, _s in _rf_arm_pretip if _s == "2D!")
            #         print(
            #             f"[2F-SEG] {_rl['result']} 区={_rz} 空间={_gs} "
            #             f"段{_rl.get('wall_ms', 0)}ms None{_rl.get('none_n', 0)}帧"
            #             f" 入口={_rl.get('kind') or '?'}{_extra} | "
            #             f"起势TIP{_b2f(_rl.get('entry_tip'))} 资格bend{_b2f(_rl.get('entry_bend'))} | "
            #             f"ARM r{_b2f(_rl.get('arm_r'))} 谷底{_b2f(_rl.get('min_r'))} "
            #             f"末r{_b2f(_rl.get('r'))} | {_pstr} | "
            #             f"TIP段内{_b2f(_rf_tip_min)}~{_b2f(_rf_tip_max)}(门0.50) "
            #             f"DIP段内{_b2f(_rf_dip_min)}~{_b2f(_rf_dip_max)}(门1.30) | "
            #             f"ARM前12帧TIP[{_pre_vals}]{(' 含2D兜底x' + str(_pre_2d_n)) if _pre_2d_n else ''} | "
            #             f"食弯 ARM{_b2f(_rf_arm_idx)} 段内{_b2f(_rf_idx_min)}~{_b2f(_rf_idx_max)}"
            #             f"摆{_b2f(_sw_i)} | "
            #             f"中弯 ARM{_b2f(_rf_arm_mid)} 段内{_b2f(_rf_mid_min)}~{_b2f(_rf_mid_max)}"
            #             f"摆{_b2f(_sw_m)} | "
            #             f"3D合成r ARM{_b2f(_rf_b3d_arm)} 谷{_b2f(_rf_b3d_min)}(2D谷{_b2f(_rl.get('min_r'))}) | "
            #             f"ARM{_arm_xy_s}→末点{_xy}",
            #             flush=True)
            #     except Exception:
            #         pass

            if not both_hands_global:
                # 1=单击(窗口超时补发) 2=长按按下 5=双击(快速原子) 6=慢节奏补救双击 7=双指右键：都要求鼠标钉在当前位置完成动作
                if tap_event in (1, 2, 5, 6) or right_tap_event == _TEV.RIGHT_TAP:
                    # freeze_for_click 冻结位置 = 同帧 move_only 输出位置（smoother 已被冻结，后续几帧都返回相同像素点）
                    mouse._smoother.freeze_for_click()
                if tap_event != 0:
                    # handle_finger_tap_event 内部左键 click / drag 用对齐后的 right_screen_pt
                    _hb_stage[0] = "tap发送"
                    mouse.handle_finger_tap_event(tap_event, right_screen_pt,
                                                  first_xy=_pending_first_xy)
                    _hb_stage[0] = "business"
                    # ⭐ 2026-10-02 逐击诊断（纯观测）：落点屏幕区域 × ARM→送达耗时
                    if tap_event in (1, 5, 6):
                        try:
                            _hit_x = int(right_screen_pt[0])
                            _hit_y = int(right_screen_pt[1])
                            _sh = max(1, int(getattr(mouse, "sh", 1080)))
                            _yr = _hit_y / _sh
                            _zone = ("顶部标题栏" if _yr < 0.06
                                     else ("底部任务栏" if _yr > 0.94 else "中部内容区"))
                            _cms = tap_detector._click_ms
                            _deliver_ms = round(float(_cms[-1]), 0) if len(_cms) else -1
                            _m = tap_detector.metrics()
                            print(
                                f"[TAP-HIT] ev={tap_event} 点({_hit_x},{_hit_y}) 区={_zone} "
                                f"ARM→送达{_deliver_ms:.0f}ms hold={_m['hold_sec']}s "
                                f"ratio={_m['bend_ratio']} 谷raw={_m['seg_min_raw']}(捏合门{PINCH_ABS_MAX}) "
                                f"bend={_m['bend']}(松开线{PINCH_RELEASE_ABS}) base={_m['baseline']} "
                                f"差分按={'是' if _m['armed_by_delta'] else '否'}",
                                flush=True)
                        except Exception:
                            pass
                if right_tap_event != 0:
                    # 拇指TIP4与中指TIP12接近后回弹 → 右键单击
                    _hb_stage[0] = "右键发送"
                    mouse.handle_finger_tap_event(right_tap_event, right_screen_pt)
                    _hb_stage[0] = "business"
                    print(f"[右键] ev={right_tap_event} 点({int(right_screen_pt[0])},{int(right_screen_pt[1])})",
                          flush=True)

            # ---------- 单手五指张开 → 移动掌心下窗口（双手场景不触发） ----------
            #   GameViewer 下五指张开改为左键长按拖动，不走移窗状态机（传 n_hands=0 让其回 IDLE）。
            # ⭐⭐ 2026-10-29 用户需求：注释五指张开(OPEN)手势逻辑——
            #   ① gesture_recognizer.GESTURE_TABLE 已删 OPEN 表项（全张开 → OTHER 正常控鼠）；
            #   ② window_mover.update 不再被调用（mover_info 保持初始 None，
            #      下方绘制/光标锁定等所有 mover_info 消费点均有 None 防护，自动失效）；
            #   ③ GV 分支（五指张开=左键拖动）已在 Pass B-1 注释。
            # _mv_n = len(hand_results)
            # if getattr(run_loop, "_is_gv", False):
            #     _mv_n = 0
            # _mv_lms = None
            # _mv_px = None
            # _mv_screen = None
            # if _mv_n == 1:
            #     _, _, _, _mv_lms = next(iter(hand_results.values()))
            #     try:
            #         _nx, _ny = HandDetector.cursor_anchor_from_landmarks(_mv_lms, args.anchor)
            #         _mv_px = (float(_nx) * w, float(_ny) * h)
            #         _sx, _sy = mouse._norm_to_screen(float(_nx), float(_ny))
            #         _mv_screen = (float(_sx), float(_sy))
            #     except Exception:
            #         _mv_px, _mv_screen = None, None
            # mover_info = window_mover.update(_mv_n, _mv_lms, _mv_px, _mv_screen, dt=_frame_dt)

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

            # ---------- 2026-10-15 双手转动滚轮和弦 UI（轨迹/锁定方向/速度，仅和弦期绘制）----------
            #   每帧必画（中文不跳帧，避免闪烁）；和弦不成立时整块零开销跳过。
            if _chord_scroll:
                try:
                    _trail_pts = [(int(round(float(tx) * _circle_short)),
                                   int(round(float(ty) * _circle_short)))
                                  for tx, ty in _circle_m.get("trail", ())]
                    _locked = (_circle_m.get("state") == "LOCKED")
                    _dir = int(_circle_m.get("dir", 0))
                    _mdir = int(_circle_m.get("m_dir", 0))
                    # 已锁定按锁定方向上色；未锁定按短窗趋势预览上色
                    _c_sign = _dir if _locked else _mdir
                    _trail_color = ((0, 200, 255) if _c_sign > 0      # 逆时针=橙黄
                                    else (255, 80, 160) if _c_sign < 0  # 顺时针=品红
                                    else (180, 180, 180))
                    # 指尖轨迹（最近 40 点，平滑后坐标）
                    for i in range(1, len(_trail_pts)):
                        cv2.line(frame, _trail_pts[i - 1], _trail_pts[i],
                                 _trail_color, 2, cv2.LINE_AA)
                    # 当前指尖点（锁定=实心大，未锁=空心小）
                    if _circle_px is not None:
                        cv2.circle(frame,
                                   (int(round(_circle_px[0])), int(round(_circle_px[1]))),
                                   7 if _locked else 5, _trail_color,
                                   -1 if _locked else 1)
                    # 状态卡（顶部居中偏下，黑底每帧重画）
                    _card_w, _card_h = 780, 78
                    _cx_card = (w - _card_w) // 2
                    _cy_card = 64
                    cv2.rectangle(frame, (_cx_card, _cy_card),
                                  (_cx_card + _card_w, _cy_card + _card_h),
                                  (0, 0, 0), -1)
                    cv2.rectangle(frame, (_cx_card, _cy_card),
                                  (_cx_card + _card_w, _cy_card + _card_h),
                                  _trail_color if _locked else (0, 230, 230), 2)
                    put_text_cn(frame,
                                "🔒 左手握拳 + 右手转动 = 滚轮：顺时针⬇下滑 / 逆时针⬆上滑（方向锁定，越快越滚）",
                                (_cx_card + 14, _cy_card + 12),
                                font_scale=0.6,
                                color=_trail_color if _locked else (0, 230, 230),
                                thickness=2)
                    if _locked and _dir > 0:
                        _state_txt = "🔒 已锁定：逆时针 ⬆ 持续上滑中（停转或改顺时针才解除）"
                        _state_col = (0, 200, 255)
                    elif _locked and _dir < 0:
                        _state_txt = "🔒 已锁定：顺时针 ⬇ 持续下滑中（停转或改逆时针才解除）"
                        _state_col = (255, 80, 160)
                    elif _mdir > 0:
                        _state_txt = "识别到逆时针趋势…（约 1/12 圈即锁定）"
                        _state_col = (0, 200, 255)
                    elif _mdir < 0:
                        _state_txt = "识别到顺时针趋势…（约 1/12 圈即锁定）"
                        _state_col = (255, 80, 160)
                    else:
                        _state_txt = "转动右手食指即可滚动（直线划动无效）"
                        _state_col = (220, 220, 220)
                    put_text_cn(frame, _state_txt, (_cx_card + 14, _cy_card + 40),
                                font_scale=0.55, color=_state_col, thickness=2)
                    # 速度条（当前角速度 / 满幅 1.5圈/秒）
                    _pb_x, _pb_y, _pb_w, _pb_h = _cx_card + 470, _cy_card + 32, 200, 12
                    cv2.rectangle(frame, (_pb_x, _pb_y), (_pb_x + _pb_w, _pb_y + _pb_h),
                                  (60, 60, 60), -1)
                    _pct = float(_circle_m.get("speed_pct", 0.0))
                    cv2.rectangle(frame, (_pb_x, _pb_y),
                                  (_pb_x + int(_pb_w * min(1.0, _pct)), _pb_y + _pb_h),
                                  _trail_color, -1)
                    put_text_cn(frame, f"速度{int(_pct * 100):3d}%",
                                (_pb_x + _pb_w + 10, _pb_y + 11),
                                font_scale=0.5, color=_trail_color, thickness=2)
                    # 锁定/翻向瞬间方向大字（0.5s 闪烁）
                    _ev = _circle_m.get("ev", "")
                    if _ev in ("LOCK", "FLIP") and _wall_now - _circle_m.get("ev_t", 0) < 0.5:
                        _ev_txt = ("⬆ 上滑" if int(_circle_m.get("ev_dir", 0)) > 0
                                   else "⬇ 下滑")
                        _ev_txt = ("锁定 " if _ev == "LOCK" else "翻向 ") + _ev_txt
                        put_text_cn(frame, _ev_txt, (_cx_card + 14, _cy_card + 66),
                                    font_scale=0.62, color=_state_col, thickness=2)
                except Exception:
                    pass

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

            # ---------- 绘制：单手五指张开移窗 ----------
            try:
                if (mover_info is not None and _mv_lms is not None
                        and (mover_info.need_redraw or mover_info.ui_active)
                        and mover_info.anchor_px is not None):
                    mv_color = (90, 255, 120)
                    x1, y1 = int(mover_info.anchor_px[0]), int(mover_info.anchor_px[1])
                    cv2.circle(frame, (x1, y1), 26, mv_color, 3)
                    for _tip_i in (4, 8, 12, 16, 20):
                        cv2.circle(frame,
                                   (int(float(_mv_lms[_tip_i].x) * w),
                                    int(float(_mv_lms[_tip_i].y) * h)),
                                   8, mv_color, -1)
                    put_text_cn(frame, "张开", (max(4, x1 - 26), max(4, y1 + 38)),
                                font_scale=0.55, color=mv_color, thickness=2)

                    if (mover_info.move_state == _WMS.MOVE_CONFIRM
                            or (mover_info.ui_active and mover_info.move_state == _WMS.IDLE)):
                        hold_n = MOVE_CONFIRM_SEC
                        pct = mover_info.confirm_sec / max(1e-6, hold_n)
                        pct = min(1.0, pct)
                        prog_w = 420
                        bx = x1 + 40 if x1 < w // 2 else (x1 - 40 - prog_w)
                        by = y1 - 66
                        cv2.rectangle(frame, (bx - 6, by - 30), (bx + prog_w + 6, by + 60), (0, 0, 0), -1)
                        cv2.rectangle(frame, (bx, by), (bx + prog_w, by + 14), (50, 50, 50), -1)
                        if mover_info.ui_active and mover_info.move_state == _WMS.IDLE:
                            cv2.rectangle(frame, (bx, by), (bx + int(prog_w * 0.02), by + 14), mv_color, -1)
                            put_text_cn(frame,
                                        f"✋预判定【移动窗口(单手)】掌心指向要移动的窗口 ｜ 保持五指张开{hold_n:.2f}s即可选中",
                                        (bx, by + 24), font_scale=0.55, color=(255, 255, 255), thickness=2)
                        else:
                            cv2.rectangle(frame, (bx, by), (bx + int(prog_w * pct), by + 14), mv_color, -1)
                            put_text_cn(frame,
                                        f"✋选中窗口确认中(单手)… {mover_info.confirm_sec:.2f}/{hold_n:.2f}s({int(pct*100)}%)",
                                        (bx, by + 24), font_scale=0.58, color=(255, 255, 255), thickness=2)
                    elif mover_info.move_state == _WMS.MOVE_ACTIVE:
                        dx_int = int(round(float(mover_info.delta_xy[0])))
                        dy_int = int(round(float(mover_info.delta_xy[1])))
                        rect = mover_info.window_rect
                        if rect is not None and len(rect) == 4:
                            ww = int(rect[2] - rect[0]); wh = int(rect[3] - rect[1])
                        else:
                            ww = wh = 0
                        lines = [
                            (f"✋五指张开(单手) → 移动窗口中 Δx={dx_int:+d}  Δy={dy_int:+d}", (255, 255, 255), 0.72),
                            (f"方向：{'右' if dx_int > 4 else ('左' if dx_int < -4 else '—')}  "
                             f"{'下' if dy_int > 4 else ('上' if dy_int < -4 else '—')}  ｜ 光标已锁定窗口中心",
                             mv_color, 0.55),
                            (f"窗口: HWND 0x{int(mover_info.move_hwnd):08X}  {ww}×{wh}",
                             (230, 230, 230), 0.52),
                        ]
                        card_w, card_h = 620, 110
                        bx = x1 + 40 if x1 < w // 2 else (x1 - 40 - card_w)
                        by = y1 - card_h // 2
                        cv2.rectangle(frame, (bx, by), (bx + card_w, by + card_h), (0, 0, 0), -1)
                        cv2.rectangle(frame, (bx, by), (bx + card_w, by + card_h), mv_color, 2)
                        put_text_cn(frame, lines[0][0], (bx + 22, by + 14), font_scale=lines[0][2], color=lines[0][1], thickness=2)
                        put_text_cn(frame, lines[1][0], (bx + 22, by + 46), font_scale=lines[1][2], color=lines[1][1], thickness=2)
                        put_text_cn(frame, lines[2][0], (bx + 22, by + 86), font_scale=lines[2][2], color=lines[2][1], thickness=2)

                    # 左上角状态条
                    s_text, s_color = "", None
                    if mover_info.move_state == _WMS.MOVE_CONFIRM or (mover_info.ui_active and mover_info.move_state == _WMS.IDLE):
                        s_color = (90, 255, 120)
                        s_text = f"✋ 单手五指张开-移动窗口 确认中 {mover_info.confirm_sec:.2f}/{MOVE_CONFIRM_SEC:.2f}s"
                    elif mover_info.move_state == _WMS.MOVE_ACTIVE:
                        s_color = (90, 255, 120)
                        s_text = (f"✋ 单手移动窗口中 Δx={int(round(float(mover_info.delta_xy[0]))):+d} "
                                  f"Δy={int(round(float(mover_info.delta_xy[1]))):+d}  光标锁窗口中心")
                    if s_text:
                        cv2.rectangle(frame, (8, 90), (8 + 470, 118), (0, 0, 0), -1)
                        put_text_cn(frame, s_text, (14, 94), font_scale=0.6, color=s_color, thickness=2)
            except Exception:
                pass

            # ---------- 顶部模式标签 ----------
            anchor_text = {"mcp": "指根", "palm": "掌心", "index_tip": "食指尖"}.get(
                args.anchor, args.anchor)
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
            # ⭐ 2026-10-29 新手势：点击源 = 拇指TIP(4)+食指TIP(8) 捏合（单击/双击/长按拖拽全套）；
            #   五指张开移窗已按用户要求注释移除
            help_line = "右手=控鼠 | 拇食指尖捏合=单击 | 快速二次捏合=双击 | 捏住不放=拖拽 | ✊左手握拳+右手转动=滚轮(顺↓逆↑,无需画圆) | q退"
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
            _opt_skip_pct = float(100.0 * _skip_skipped_frames / max(_skip_total_frames, 1))
            status_store.update(dict(
                fps=float(_fps_display),
                uptime_ms=int(time.time() * 1000) - start_ms,
                last_left_gesture=lg,
                last_right_gesture=rg,
                move_mode=(mover_info.move_state if mover_info is not None else "IDLE"),
                chord_scroll=bool(_chord_scroll),
                preview_active=bool(preview),
                # 优化诊断字段
                opt_infer_size=480,
                opt_ema_alpha=float(_ema_a),  # 2026-10-02 墙钟化：最近一帧实际 α（τ=60ms，随帧率变化；跳帧=0）
                opt_mv_window=0,  # 2026-10-15 帧投票已停用（0），改走时钟区间稳定器
                opt_skip_total=int(_skip_total_frames),
                opt_skip_skipped=int(_skip_skipped_frames),
                opt_skip_pct=float(_opt_skip_pct),
            ))

            # ---------- imshow / waitKey（仅 preview=True） ----------
            if preview:
                _hb_stage[0] = "imshow"
                cv2.imshow("DualHand -> Mouse (双手体感鼠标)", frame)
                # waitKey(1) 必须在创建窗口的同一线程调用，所以这里直接调
                # （拖动 HighGUI 窗口标题栏/弹出系统菜单时，消息泵会进入模态循环，
                #   waitKey 可在此阻塞数秒~数十秒——watchdog 会以阶段=imshow 抓到）
                _wk_t0 = time.perf_counter()
                _wk_key = cv2.waitKey(1) & 0xFF
                try:
                    _wk_ms = (time.perf_counter() - _wk_t0) * 1000.0
                    if _wk_ms > _det_rs["wk_max_ms"]:
                        _det_rs["wk_max_ms"] = _wk_ms
                except Exception:
                    pass
                if _wk_key == ord("q"):
                    break
                _hb_stage[0] = "帧间"
            else:
                # ⚠️ 2026-08-26 核心修复（无预览=不能控制鼠标根因）：
                #   无任何 imshow / namedWindow 的情况下，Windows 版 OpenCV 的 cv2.waitKey(1)
                #   因为 HighGUI 消息泵没有窗口可挂，会进入「死等键盘事件」阻塞（不是1ms立即返回）
                #   → 主循环每帧卡死在这里，move_only() / handle_finger_tap_event() 虽然已经执行完，
                #     但下一轮循环永远到不了 → 用户体感「鼠标完全不动」。
                #   修复：纯后台模式直接用 time.sleep(0)（仅让出 CPU 时间片给同优先级线程，
                #   不进 HighGUI、不等待定时器）。
                # ⭐ 2026-09-20：曾用 time.sleep(0.001)，但 Windows 默认定时器粒度 15.6ms，
                #   实测该调用每帧阻塞 15.2ms（主循环从~20fps 被压到~8fps 的三大元凶之一）。
                #   sleep(0) 不阻塞、零定时器税，仍把执行权让给 Flask 请求线程；
                #   且 detect() 等待回调的 Event.wait 期间本就释放 GIL，Flask 有充足调度窗口。
                time.sleep(0)
                # ⭐ 2026-10-29 [RUN-OK bg-mode] 无预览心跳日志——
                #   与捏合手势观察无关，按用户要求注释（捏合诊断看 [PINCH-DBG]/[TAP-SEG]）。
                # if (_bg_heartbeat_cnt % 120 == 0) and (_bg_heartbeat_cnt > 0):
                #     print(f"[RUN-OK bg-mode] frame={_bg_heartbeat_cnt:>5} FPS={_fps_display:5.1f} preview=OFF gesture_left={hand_results.get('Left',(None,))[0]} move_ok={right_screen_pt is not None} tap_event={tap_event}",
                #           flush=True)

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
