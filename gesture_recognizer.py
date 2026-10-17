import numpy as np

"""
手势识别模块 v2
=================
修复（基于用户反馈 V 手势识别不到）：
  1. 四指伸直阈值 ratio 从 0.65 降至 0.58（更宽松，避免 V 手势时中指被判为弯）；
  2. PEACE 增加「无名指/小指必须弯曲」的二次校验 + 食指中指必须充分伸展（>0.7）
     —— 防止张开手掌时误判为 V；
  3. 新增 PINCH（距离捏合）手势：拇指 TIP(4) 与食指 TIP(8) 距离 < 掌宽*0.28
     —— 作为替代点击方案，触发更直观；
  4. 匹配优先级调整：PINCH → PEACE → ROCK → POINT → THUMB → FIST → OPEN

设计原则：
  - 所有距离阈值均以「掌宽 index_mcp(5) → pinky_mcp(17)」归一化；
  - 两层判定：伸/屈布尔值 → 手势映射；
  - 所有 debug 比例值都会在画面上显示，方便调参。
"""

# ⭐⭐ 2026-10-15 用户方案：3D world 五指卷心比重建布尔位（替换不准的 2D 布尔）
#   对每根手指 i ∈ {拇,食,中,无名,小} 计算
#       r_i = dist(TIP_i, P) / dist(PIP_i, P)   （拇指 PIP 角色 = IP 关节3）
#   P = 掌心三角点 = wrist(0)、食根 MCP(5)、小根 MCP(17) 三点质心。
#   几何含义：手指伸直时 TIP 在 PIP 之外、离掌心更远（r_i>1）；手指折过 PIP 卷回
#   掌心时 TIP 比 PIP 更靠近 P（r_i<1）。
#   布尔重建（用户拍板 1.2 分界）：
#       r_i < FOLD_EXTEND_R(1.2) → 该指位=0（弯）；r_i ≥ 1.2 → 该指位=1（伸）。
#   五指全部 <1.2 → ext=(0,0,0,0,0) → 表判 FIST。2026-10-15 真机松握临界抖动达
#   1.15（连续2~3帧），1.1 硬阈值被双向穿透，1.2 留出噪声余量；真勾按 1.34+、
#   真张开 1.3+ 仍在 1.2 外侧。残余临界噪声由 GestureTimeStabilizer 时钟区间吞掉。
#   重建后的五指位【整体替换】2D TIP-MCP 布尔（拇指同形式：r_t=‖TIP4−P‖/‖IP3−P‖，
#   IP 关节3 扮演其他指 PIP 的角色——真机证 2D 拇指位在握拳帧同样被误读成伸直，
#   若保留 2D 拇指位，FIST 会被 GESTURE_TABLE 抢先匹配成 THUMB）。
#   随后只走同一张 GESTURE_TABLE 匹配——FIST 即五指位全 0 的普通表项，无任何特判门。
#   真机分布：深握拳 0.44~0.90、松握 1.00~1.10；勾按/点击食指 1.34~1.74；张开 >1.2。
#   拇指 r_t 暂无独立标定，阈值先同 1.2（真机握拳漂 ≤1.18、真竖拇指 ≥1.22），
#   fold_t 随日志输出待更多样本修正。
#   曾试过 P=四指根弓质心(MCP5/9/13/17)，日志标注"应失效"（握拳时比值恒 >1，无区分
#   度），不得再换回。
#   world 缺失/退化/异常 → 返回 None，五指位保留原 2D 布尔（缺失路径行为不变）。
FOLD_EXTEND_R = 1.2
# (TIP, PIP/IP) 拇4/3、食8/6、中12/10、无名16/14、小20/18
_FOLD_TIP_PIP_PAIRS = ((4, 3), (8, 6), (12, 10), (16, 14), (20, 18))


def _five_finger_fold_ratios(wlms):
    """五指卷心比 r = ‖TIP−P‖ / ‖PIP(IP)−P‖（3D world），P=0/5/17 掌心三角质心。

    返回 (拇,食,中,无名,小)；关键点不足/退化向量/异常 → None（调用方回退旧 2D 判定）。
    """
    try:
        if wlms is None or len(wlms) < 21:
            return None
        pts = [np.array([float(p.x), float(p.y), float(p.z)], dtype=np.float64)
               for p in wlms]
        p_center = (pts[0] + pts[5] + pts[17]) / 3.0
        out = []
        for tip, pip in _FOLD_TIP_PIP_PAIRS:
            d_tip = float(np.linalg.norm(pts[tip] - p_center))
            d_pip = float(np.linalg.norm(pts[pip] - p_center))
            if d_pip < 1e-9:
                return None
            out.append(d_tip / d_pip)
        return tuple(out)
    except Exception:
        return None


# 七类手势（拇,食,中,无,小） 1=伸直 0=弯曲
# 注意：PINCH 不由布尔组合判定，而由距离判定，单独放在最优先级匹配
GESTURE_TABLE = [
    ("PEACE", (0, 1, 1, 0, 0)),   # V字 食+中
    ("ROCK",  (1, 1, 0, 0, 1)),   # 拇+食+小
    ("POINT", (0, 1, 0, 0, 0)),   # 仅食指
    ("THUMB", (1, 0, 0, 0, 0)),   # 仅拇指
    ("FIST",  (0, 0, 0, 0, 0)),   # 握拳
    ("OPEN",  (1, 1, 1, 1, 1)),   # 全张开
]


def _p2(lm):
    return np.array([lm.x, lm.y], dtype=np.float64)


def _finger_extended(tip_lm, mcp_lm, palm_width: float, ratio: float = 0.58):
    """单指伸直判定（ratio 从 0.65 降到 0.58，更宽松以适配 V 手势）。

    2026-10-14 实测记录：曾试 0.58→0.62 以提高 FIST 召回，结果 OPEN 五指张开
    失效（小指在手掌倾斜时 2D r 率先跌破门限，全 1 组合匹配失败）。全局门限
    不可动；FIST 强化请走 3D GUARD / 旁路融合判据，不要改此默认值。
    """
    dist = np.linalg.norm(_p2(tip_lm) - _p2(mcp_lm))
    r = dist / palm_width
    return r > ratio, r


def _pinch_distance(lms, palm_width: float):
    """拇指(4)-食指(8)捏合距离 / 掌宽；值越小越近（2D 投影）。"""
    d = np.linalg.norm(_p2(lms[4]) - _p2(lms[8]))
    return d / palm_width


def _pinch_index_pip_mcp(lms):
    """拇指(4)-食指(8) TIP 距离 / 食指 PIP(6)-MCP(5) 距离（2D 归一化）。
    食指 PIP→MCP 是近节指骨，长度最稳定，不受手指弯曲/手掌侧对镜头影响；
    2D 比值的分子分母同受远近缩放，比值稳定，不依赖 world 坐标。
    """
    try:
        tip_d = np.linalg.norm(_p2(lms[4]) - _p2(lms[8]))
        ref_d = np.linalg.norm(_p2(lms[6]) - _p2(lms[5]))
        if ref_d < 1e-6:
            return None
        return float(tip_d / ref_d)
    except Exception:
        return None


# 中、无名、小指的 (MCP, PIP, DIP) 关键点索引（测 PIP 关节角用）
_OK_THREE_BEND_TRIPLES = ((9, 10, 11), (13, 14, 15), (17, 18, 19))


def _pip_bend_2d(lms, mcp: int, pip: int, dip: int):
    """单指 PIP 关节 2D 弯曲度：∠MCP-PIP-DIP 归一化。
    完全伸直=180° → 0.0；弯成 90° → 0.5；卷回 → 接近 1.0。纯 2D，不看距离/不看 world。
    """
    try:
        v1 = _p2(lms[mcp]) - _p2(lms[pip])
        v2 = _p2(lms[dip]) - _p2(lms[pip])
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1e-9 or n2 < 1e-9:
            return None
        c = float(np.dot(v1, v2)) / (n1 * n2)
        c = -1.0 if c < -1.0 else (1.0 if c > 1.0 else c)
        ang = float(np.degrees(np.arccos(c)))
        return (180.0 - ang) / 180.0
    except Exception:
        return None


def _three_finger_bend_2d(lms):
    """中/无名/小指 PIP 关节 2D 弯曲度，返回 (中, 无名, 小)；越小越直（伸直≈0~0.15）。退化 → None。"""
    try:
        out = []
        for mcp, pip, dip in _OK_THREE_BEND_TRIPLES:
            b = _pip_bend_2d(lms, mcp, pip, dip)
            if b is None:
                return None
            out.append(b)
        return tuple(out)
    except Exception:
        return None


def _pinch_distance_3d(wlms):
    """拇指(4)-食指(8) TIP 的 3D world 距离（米）；值越小越近。
    wlms 缺失/退化 → None。
    """
    try:
        if wlms is None or len(wlms) < 9:
            return None
        p4 = np.array([wlms[4].x, wlms[4].y, wlms[4].z], dtype=np.float64)
        p8 = np.array([wlms[8].x, wlms[8].y, wlms[8].z], dtype=np.float64)
        return float(np.linalg.norm(p4 - p8))
    except Exception:
        return None


def _palm_width_3d(wlms):
    """3D 掌宽：MCP5(5) → MCP17(17) 的 world 距离（米）；用于归一化，消除手远近影响。"""
    try:
        if wlms is None or len(wlms) < 18:
            return None
        p5 = np.array([wlms[5].x, wlms[5].y, wlms[5].z], dtype=np.float64)
        p17 = np.array([wlms[17].x, wlms[17].y, wlms[17].z], dtype=np.float64)
        return float(np.linalg.norm(p5 - p17))
    except Exception:
        return None


def _index_length_3d(wlms):
    """3D 食指长度：MCP5(5) → TIP8(8) 的 world 距离（米）。
    OK 手势时手掌侧对镜头，掌宽(MCP5-MCP17)不稳定；食指伸直，长度稳定，用作归一化参考。
    """
    try:
        if wlms is None or len(wlms) < 9:
            return None
        p5 = np.array([wlms[5].x, wlms[5].y, wlms[5].z], dtype=np.float64)
        p8 = np.array([wlms[8].x, wlms[8].y, wlms[8].z], dtype=np.float64)
        return float(np.linalg.norm(p5 - p8))
    except Exception:
        return None


def recognize_gesture(lms, wlms=None):
    """识别单只手手势。

    Args:
        lms:  2D 归一化关键点（21 个，含 .x/.y）
        wlms: 3D world 关键点（21 个，含 .x/.y/.z）；提供时五指伸/屈布尔位由卷心比
              r=dist(TIP,P)/dist(PIP或IP,P)（P=0/5/17 掌心三角质心，r<1.1=弯）
              重建后再走 GESTURE_TABLE；缺失/异常时沿用原 2D 布尔（PINCH 距离门、
              PEACE 二次校验不受影响）。

    Returns:
        (gesture_name, ext_tuple(5个0/1), debug_info_dict)
    """
    # 1. 基准尺度：掌宽
    palm_width = float(np.linalg.norm(_p2(lms[5]) - _p2(lms[17]))) + 1e-6
    # 1b. 3D 五指卷心比 (拇,食,中,无,小)：用于重建五指伸/屈布尔位
    _fold = _five_finger_fold_ratios(wlms)

    # 2. 五指伸/屈 + 比值
    index_ext,  index_r  = _finger_extended(lms[8],  lms[5],  palm_width)
    middle_ext, middle_r = _finger_extended(lms[12], lms[9],  palm_width)
    ring_ext,   ring_r   = _finger_extended(lms[16], lms[13], palm_width)
    pinky_ext,  pinky_r  = _finger_extended(lms[20], lms[17], palm_width)
    thumb_raw = np.linalg.norm(_p2(lms[4]) - _p2(lms[5])) / palm_width
    thumb_ext = thumb_raw > 0.50  # 拇指阈值也略降

    ext_tuple = (int(thumb_ext), int(index_ext), int(middle_ext),
                 int(ring_ext), int(pinky_ext))

    # 2b. ⭐ 3D 卷心比重建五指布尔位（用户 2026-10-15 方案）：
    #   world 有效时，五指伸/屈位不再信 2D TIP-MCP（真机握拳帧 2D 把五指全误读成
    #   伸直），改用 r：r<1.1 → 0（弯），r≥1.1 → 1（伸）。
    #   重建后 PINCH 距离门之外的所有手势（含 FIST/POINT/OPEN/PEACE/ROCK/THUMB）
    #   都只按这同一份 ext_tuple 走原有判定，不再有任何 FIST 特判门。
    if _fold is not None:
        ext_tuple = tuple(0 if r < FOLD_EXTEND_R else 1 for r in _fold)

    # 3. 距离手势：PINCH 捏合（最高优先级，距离判定）
    pinch_r = _pinch_distance(lms, palm_width)
    # OK 手势捏合判定：2D 拇食指 TIP 距 / 2D 食指 PIP(6)-MCP(5) 距
    #   食指 PIP→MCP 是近节指骨，长度最稳定，不受手掌侧对镜头影响；
    #   2D 比值的分子分母同受远近缩放，比值稳定，不依赖 world 坐标。
    pinch_pip = _pinch_index_pip_mcp(lms)
    # 捏合阈值（真机标定 2026-10-16 晚，捏定 40 帧）：OK 簇 0.26~0.37（分母近节骨稳定 0.128~0.132），
    #   过渡/张开帧 ≥0.53（上一批日志）→ 门取 0.50。旧掌宽口径侧对时 1.24~2.65 不可用，已弃用。
    _ok_pinch_ok = pinch_pip is not None and pinch_pip < 0.50
    # 中/无名/小指 2D 弯曲度（PIP 关节角，0=直 1=弯）：真机捏定 中≤0.15/无≤0.08/小≤0.05；弯帧无≥0.36。
    #   门 0.25 = PIP 角 >135° 即算伸直。
    _ok_three = _three_finger_bend_2d(lms)
    _ok_mid_ok = _ok_three is not None and all(v < 0.25 for v in _ok_three)

    # 3b. OK 手势：拇指食指成圈 + 中/无名/小三指伸直
    #   判据1：2D dist(拇TIP4,食TIP8) / dist(食PIP6,食MCP5) < 0.50（真机捏定0.26~0.37）。
    #   判据2：三指 PIP 关节 2D 弯曲度全部 < 0.25（真机捏定 中≤0.15）。就这两条，无其它门控。
    # [OK-DBG] 调试打印已禁用（用户要求减少刷屏，2026-10-17）
    # if pinch_pip is not None and pinch_pip < 2.5:
    #     _num = float(np.linalg.norm(_p2(lms[4]) - _p2(lms[8])))
    #     _den = float(np.linalg.norm(_p2(lms[6]) - _p2(lms[5])))
    #     if _ok_pinch_ok and _ok_mid_ok:
    #         _verdict = "OK"
    #     elif not _ok_mid_ok:
    #         _verdict = "非OK(三指未全伸)"
    #     else:
    #         _verdict = "张开"
    #     print(f"[OK-DBG] 捏合比={pinch_pip:.2f}(门0.50) [TIP4-8={_num:.3f} / PIP6-MCP5={_den:.3f} "
    #           f"掌宽={palm_width:.3f} 旧掌宽比={pinch_r:.2f}] 2D弯曲 中/无/小="
    #           f"({_ok_three[0]:.2f},{_ok_three[1]:.2f},{_ok_three[2]:.2f})(门0.25,小=直) "
    #           f"=> {_verdict}",
    #           flush=True)
    if _ok_pinch_ok and _ok_mid_ok:
        ratios = {
            "palm_w": round(palm_width, 4),
            "thumb_r": round(thumb_raw, 3),
            "index_r": round(index_r, 3),
            "middle_r": round(middle_r, 3),
            "ring_r": round(ring_r, 3),
            "pinky_r": round(pinky_r, 3),
            "pinch_r": round(pinch_r, 3),
            "pinch_pip": round(pinch_pip, 3) if pinch_pip is not None else None,
            "okb_m": round(_ok_three[0], 3), "okb_r": round(_ok_three[1], 3),
            "okb_p": round(_ok_three[2], 3),
        }
        if _fold is not None:
            ratios.update({"fold_t": round(_fold[0], 3), "fold_i": round(_fold[1], 3),
                           "fold_m": round(_fold[2], 3), "fold_r": round(_fold[3], 3),
                           "fold_p": round(_fold[4], 3)})
        # OK 三指实际伸直，返回给下游的布尔位也置成伸直形态
        ext_tuple = (ext_tuple[0], ext_tuple[1], 1, 1, 1)
        return "OK", ext_tuple, ratios

    # ✨ 用户反馈"捏合太严格无法触发"：阈值同步放松 原 0.18/0.28 → 现 0.25/0.40
    # 条件：捏合中等 (<0.40) 且 食指中指都有一定弯曲避免张开时误触发
    # 或者 捏合比较紧 (<0.25) 时无附加条件（轻捏就能被识别成 PINCH）
    if pinch_r < 0.25 or (pinch_r < 0.40 and not (index_ext and middle_ext)):
        ratios = {
            "palm_w": round(palm_width, 4),
            "thumb_r": round(thumb_raw, 3),
            "index_r": round(index_r, 3),
            "middle_r": round(middle_r, 3),
            "ring_r": round(ring_r, 3),
            "pinky_r": round(pinky_r, 3),
            "pinch_r": round(pinch_r, 3),
        }
        if _fold is not None:
            ratios.update({"fold_t": round(_fold[0], 3), "fold_i": round(_fold[1], 3),
                           "fold_m": round(_fold[2], 3), "fold_r": round(_fold[3], 3),
                           "fold_p": round(_fold[4], 3)})
        return "PINCH", ext_tuple, ratios

    # 4. PEACE (V字) 二次校验：食中指必须充分伸展 (>0.7)，且无名指小指必须弯曲
    if ext_tuple == (0, 1, 1, 0, 0):
        if index_r > 0.70 and middle_r > 0.70 and ring_r < 0.55 and pinky_r < 0.55:
            ratios = {
                "palm_w": round(palm_width, 4),
                "thumb_r": round(thumb_raw, 3),
                "index_r": round(index_r, 3),
                "middle_r": round(middle_r, 3),
                "ring_r": round(ring_r, 3),
                "pinky_r": round(pinky_r, 3),
                "pinch_r": round(pinch_r, 3),
            }
            if _fold is not None:
                ratios.update({"fold_t": round(_fold[0], 3), "fold_i": round(_fold[1], 3),
                               "fold_m": round(_fold[2], 3), "fold_r": round(_fold[3], 3),
                               "fold_p": round(_fold[4], 3)})
            return "PEACE", ext_tuple, ratios

    ratios = {
        "palm_w": round(palm_width, 4),
        "thumb_r": round(thumb_raw, 3),
        "index_r": round(index_r, 3),
        "middle_r": round(middle_r, 3),
        "ring_r": round(ring_r, 3),
        "pinky_r": round(pinky_r, 3),
        "pinch_r": round(pinch_r, 3),
    }
    if _fold is not None:
        ratios.update({"fold_t": round(_fold[0], 3), "fold_i": round(_fold[1], 3),
                       "fold_m": round(_fold[2], 3), "fold_r": round(_fold[3], 3),
                       "fold_p": round(_fold[4], 3)})

    # 5. 统一走 GESTURE_TABLE：ext_tuple 五指位已由 3D r 重建（world 有效时），
    #   FIST=(0,0,0,0,0) 只是表中普通一项，无特判、无前置门；world 缺失时五指位是
    #   原 2D 布尔，即旧行为（PEACE 已在第 4 步单独处理并返回）。
    for name, pattern in GESTURE_TABLE:
        if name == "PEACE":
            continue
        if ext_tuple == pattern:
            return name, ext_tuple, ratios

    return "OTHER", ext_tuple, ratios


# ⭐⭐ 2026-10-15 手势输出时钟区间稳定器（用户拍板：不按帧、按墙钟区间判断）：
#   候选新手势必须【连续持续满确认秒数】才能翻转稳定输出；确认期间冒出任意一帧
#   旧手势（或第三种手势），确认计时立即清零重来。
#   为什么不用帧计数：主循环有自适应跳帧，同一"3帧"在不同负载下是 0.1~0.3s 不等；
#   墙钟区间给出确定性的去抖时长。
#   进出握拳时长沿用 2026-10-01 真机标定（FIST_CURL_ENTER_SEC=0.10/EXIT_SEC=0.17）：
#     进入 FIST 快（0.10s，保证握拳响应与未来和弦武装跟手）；
#     离开 FIST 慢（0.17s，松握 r 在 1.2 临界的短时漂移不漏成 OTHER/THUMB）；
#   其余标签互相切换 0.10s。手离镜必须 reset()，杜绝单向锁（经验教训 630142）。
FIST_ENTER_SEC = 0.10
FIST_LEAVE_SEC = 0.17
GESTURE_SWITCH_SEC = 0.10


class GestureTimeStabilizer:
    """按墙钟区间确认的单只手手势稳定器（每只手一个实例）。"""

    __slots__ = ("_stable", "_cand", "_cand_t0")

    def __init__(self):
        self.reset()

    def reset(self):
        """手离镜/初始化时调用：下一个 raw 直接采纳为稳定输出。"""
        self._stable: str | None = None   # 当前稳定输出的手势
        self._cand: str | None = None     # 待确认的候选手势
        self._cand_t0: float = 0.0        # 候选首次出现的墙钟

    def update(self, raw: str, now: float) -> str:
        """喂入当帧 raw 手势与 time.monotonic() 墙钟，返回稳定后的手势。"""
        # 首帧：直接采纳，不做确认延迟
        if self._stable is None:
            self._stable = raw
            self._cand = None
            return raw
        # raw 与稳定输出一致：任何在途确认作废
        if raw == self._stable:
            self._cand = None
            return raw
        # 出现新候选（或候选中途换成别的手势）：重新开始计时
        if raw != self._cand:
            self._cand = raw
            self._cand_t0 = now
            return self._stable
        # 候选持续中：按切换方向取确认时长
        if self._stable == "FIST":
            need = FIST_LEAVE_SEC
        elif raw == "FIST":
            need = FIST_ENTER_SEC
        else:
            need = GESTURE_SWITCH_SEC
        if now - self._cand_t0 >= need:
            self._stable = raw
            self._cand = None
        return self._stable
