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
    """单指伸直判定（ratio 从 0.65 降到 0.58，更宽松以适配 V 手势）。"""
    dist = np.linalg.norm(_p2(tip_lm) - _p2(mcp_lm))
    r = dist / palm_width
    return r > ratio, r


def _pinch_distance(lms, palm_width: float):
    """拇指(4)-食指(8)捏合距离 / 掌宽；值越小越近。"""
    d = np.linalg.norm(_p2(lms[4]) - _p2(lms[8]))
    return d / palm_width


def recognize_gesture(lms):
    """识别单只手手势。

    Returns:
        (gesture_name, ext_tuple(5个0/1), debug_info_dict)
    """
    # 1. 基准尺度：掌宽
    palm_width = float(np.linalg.norm(_p2(lms[5]) - _p2(lms[17]))) + 1e-6

    # 2. 五指伸/屈 + 比值
    index_ext,  index_r  = _finger_extended(lms[8],  lms[5],  palm_width)
    middle_ext, middle_r = _finger_extended(lms[12], lms[9],  palm_width)
    ring_ext,   ring_r   = _finger_extended(lms[16], lms[13], palm_width)
    pinky_ext,  pinky_r  = _finger_extended(lms[20], lms[17], palm_width)
    thumb_raw = np.linalg.norm(_p2(lms[4]) - _p2(lms[5])) / palm_width
    thumb_ext = thumb_raw > 0.50  # 拇指阈值也略降

    ext_tuple = (int(thumb_ext), int(index_ext), int(middle_ext),
                 int(ring_ext), int(pinky_ext))

    # 3. 距离手势：PINCH 捏合（最高优先级，距离判定）
    pinch_r = _pinch_distance(lms, palm_width)
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
            return "PEACE", ext_tuple, ratios

    # 5. 其它手势按「更具体→更通用」顺序精确匹配布尔组合
    ratios = {
        "palm_w": round(palm_width, 4),
        "thumb_r": round(thumb_raw, 3),
        "index_r": round(index_r, 3),
        "middle_r": round(middle_r, 3),
        "ring_r": round(ring_r, 3),
        "pinky_r": round(pinky_r, 3),
        "pinch_r": round(pinch_r, 3),
    }
    for name, pattern in GESTURE_TABLE:
        if name == "PEACE":
            continue  # PEACE 已单独处理
        if ext_tuple == pattern:
            return name, ext_tuple, ratios

    return "OTHER", ext_tuple, ratios
