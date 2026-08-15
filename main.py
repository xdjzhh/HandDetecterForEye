import argparse
import os
import time
import cv2
import numpy as np

from hand_detector import HandDetector, TasksHandDetector
from gesture_recognizer import recognize_gesture
from mouse_controller import MouseController
from draw_cn_text import put_text_cn
from finger_tap_click_detector import FingerTapClickDetector, _S as _TS
from diagonal_stretch import DiagonalStretchManager, _S as _DS, _MODE as _DM, _MS as _DMS

"""
双手手势 → 鼠标 映射 主入口
============================

核心策略（可按需修改）：
  - 右手 = 主控手：光标移动 + 点击/双击/拖拽（详见 mouse_controller.py）
  - 左手 = 功能手：当前仅做显示，用户可扩展为修饰键（Ctrl/Alt/Shift 等）
  - 双手组合手势：两只手同时做出特定手势 → 触发快捷键（示例见下方 COMBOS）

约定（经验驱动）：
  - 单点镜像：只在本文件 frame 读取后 flip 一次，任何模块不再做 1-x 换算
  - 手的身份 ID：始终使用 hand["label"]（MediaPipe 的 Left/Right）而非列表下标
"""

# 左右手显示颜色（BGR）：左手蓝，右手橙
HAND_COLORS = {"Left": (255, 100, 0), "Right": (0, 140, 255)}
# 手指标签（拇/食/中/无名/小）
FINGER_NAMES_CN = ["拇", "食", "中", "无", "小"]

# 双手组合手势：(左手手势, 右手手势) -> 提示文字
# 可配合 pyautogui.hotkey(...) 触发真实快捷键
COMBOS = [
    (("OPEN",  "OPEN"),  "双手张开 -> 放大窗口 (示例)"),
    (("FIST",  "FIST"),  "双手握拳 -> 最小化窗口 (示例)"),
    (("POINT", "POINT"), "双指指向 -> 屏幕截图 (示例)"),
    (("PEACE", "PEACE"), "双手V字 -> 播放/暂停媒体 (示例)"),
]


def _draw_finger_state(frame, ext_tuple, ratios, origin, color):
    """在画面角落绘制 5 根手指的伸/弯状态 + 关键比值 + 捏合距离（调参用）。"""
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
    # 掌宽基准值 + 捏合距离（<0.28 触发点击）
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
    """粗略估算中文文本宽高（用于居中对齐），避免 PIL 依赖调用失败。"""
    px = max(10, int(font_scale * 33))
    # 中文=px 宽，ASCII=px/2 宽，行高≈ px*1.3
    w_sum = 0
    for c in text:
        w_sum += px if ord(c) > 127 else px // 2
    return w_sum, int(px * 1.25)


# --------------------------------------------------------------------------
# ✨ 食指 Tap-Click：伸直度条（画面右上角，替代原深度条）
#     bend = 食指伸直度 = |TIP_8 - MCP_5| / |MCP_5 - MCP_17|
#     值越大 → 伸得越直；值越小 → 食指越弯曲
# --------------------------------------------------------------------------
def _draw_tap_bar(frame, tm: dict, det: FingerTapClickDetector | None = None):
    H, W = frame.shape[:2]
    bar_x0, bar_y0 = W - 320, 36
    bar_w, bar_h = 300, 28

    # bend 可见范围：0.2（完全弯） ~ 1.0（完全伸直）
    vis_min, vis_max = 0.20, 1.00
    def _pct(x): return (x - vis_min) / (vis_max - vis_min)
    bend = max(vis_min, min(vis_max, tm["bend"])) if tm else 0.80
    bend_px = int(bar_x0 + bar_w * _pct(bend))

    # 背景（黑色圆角感矩形）
    cv2.rectangle(frame, (bar_x0 - 2, bar_y0 - 2),
                  (bar_x0 + bar_w + 2, bar_y0 + bar_h * 3 + 14),
                  (0, 0, 0), -1)

    # 条本体：灰色底
    cv2.rectangle(frame, (bar_x0, bar_y0), (bar_x0 + bar_w, bar_y0 + bar_h),
                  (60, 60, 60), -1)
    # 白：伸直基线（用户实际伸直值）
    bx = int(bar_x0 + bar_w * _pct(tm["baseline"]))
    cv2.line(frame, (bx, bar_y0), (bx, bar_y0 + bar_h), (255, 255, 255), 1)
    # 黄：EXIT_BEND_RATIO（回到此线视为"回弹伸直" → 释放/点击）
    ex = int(bar_x0 + bar_w * _pct(tm["exit_t"]))
    cv2.line(frame, (ex, bar_y0), (ex, bar_y0 + bar_h), (60, 220, 255), 2)
    # 红：ENTER_BEND_RATIO（弯到此线以下视为"开始弯"）
    en = int(bar_x0 + bar_w * _pct(tm["enter_t"]))
    cv2.line(frame, (en, bar_y0), (en, bar_y0 + bar_h), (0, 0, 255), 2)

    # 填充（从左=0.2 直到当前bend），颜色随状态
    st = tm["state"]
    if st in (_TS.BEND_ARMED, _TS.BEND_HOLD, _TS.TAPPED):
        fill_c = (0, 0, 255)
    elif st == _TS.PRESSED:
        fill_c = (0, 200, 255)
    elif st == _TS.COOLDOWN:
        fill_c = (140, 140, 140)
    else:
        fill_c = (0, 220, 140)
    cv2.rectangle(frame, (bar_x0, bar_y0), (bend_px, bar_y0 + bar_h), fill_c, -1)
    # 当前值三角（朝上）
    tri_points = np.array([
        [bend_px, bar_y0 - 4],
        [bend_px - 6, bar_y0 - 11],
        [bend_px + 6, bar_y0 - 11],
    ], np.int32)
    cv2.drawContours(frame, [tri_points], 0, fill_c, -1)

    # 第2行：中文状态 + ✨新增 TAP/LONG 双进度条可视化（一眼看是否即将误触长按）
    state_cn_map = {
        _TS.IDLE:       ("伸直待机(基线学习中)", (160, 255, 160)),
        _TS.BEND_ARMED: ("食指弯曲确认中…",      (80, 180, 255)),
        _TS.BEND_HOLD:  (f"弯曲保持 {tm['hold_frames']} 帧",  (0, 180, 255)),
        _TS.TAPPED:     ("✅ TAP 单击触发!",     (60, 60, 255)),
        _TS.PRESSED:    ("🔒 按住拖拽中",          (0, 200, 255)),
        _TS.COOLDOWN:   (f"冷却 {tm['cooldown_left']} 帧",   (180, 180, 180)),
    }
    st_text, st_color = state_cn_map.get(st, (str(st), (255, 255, 255)))
    line2_y = bar_y0 + bar_h + 6
    put_text_cn(frame, st_text, (bar_x0, line2_y),
                font_scale=0.55, color=st_color, thickness=2)

    # -------- 第3行（新增）：TAP 短按时间窗进度条 / 长按稳定计数进度条 并排 --------
    # 解决用户"不知道到底识别成 TAP 还是 LONG"的可观测化痛点（经验405048）
    line3_y = line2_y + 22
    # 条参数：左侧 TAP 短按进度（total_frames / TAP_MAX → 越快结束越好，1.0=刚好18帧，<1.0=仍在短按安全区）
    #         右侧 LONG 长按进度（hold_frames / LONG_MAX → <1.0 还没触发拖拽）
    tap_max   = float(tm.get("TAP_MAX", 18))
    long_max  = float(tm.get("LONG_MAX", 22))
    tot_f     = float(tm.get("total_frames", 0))
    hold_f    = float(tm.get("hold_frames", 0))
    stab_f    = float(tm.get("stable_frames", 0))

    pb_w = int((bar_w - 10) / 2)
    pb_h = 10
    # --- 左：TAP 安全进度条（100%=即将到TAP阈值，>100%=进入长按风险区；用户应该在此进度条满之前完成回弹）---
    pb1_x = bar_x0
    p1 = 0.0 if tot_f <= 0 else min(1.2, tot_f / tap_max)
    cv2.rectangle(frame, (pb1_x, line3_y), (pb1_x + pb_w, line3_y + pb_h), (60, 60, 60), -1)
    tap_c = (30, 255, 90) if p1 < 0.7 else ((60, 220, 255) if p1 < 1.0 else (0, 0, 255))
    cv2.rectangle(frame, (pb1_x, line3_y), (pb1_x + int(pb_w * min(1.0, p1)), line3_y + pb_h), tap_c, -1)
    put_text_cn(frame, f"短按窗口 {int(tot_f):>3d}/{int(tap_max):>2d}", (pb1_x, line3_y - 14),
                font_scale=0.32, color=tap_c, thickness=1)
    # --- 右：LONG 长按进度条（=稳定帧数 / 长按阈值；满=触发拖拽，用户想单击就必须让它保持<100%）---
    pb2_x = bar_x0 + pb_w + 10
    p2 = 0.0 if hold_f <= 0 else min(1.0, hold_f / long_max)
    cv2.rectangle(frame, (pb2_x, line3_y), (pb2_x + pb_w, line3_y + pb_h), (60, 60, 60), -1)
    long_c = (40, 220, 255) if p2 < 0.5 else ((0, 150, 255) if p2 < 1.0 else (0, 0, 255))
    cv2.rectangle(frame, (pb2_x, line3_y), (pb2_x + int(pb_w * p2), line3_y + pb_h), long_c, -1)
    put_text_cn(frame, f"长按计数 {int(hold_f):>3d}/{int(long_max):>2d}  稳{int(stab_f):>2d}f",
                (pb2_x, line3_y - 14), font_scale=0.32, color=long_c, thickness=1)

    # 第4行：bend值 + 基线 + enter/exit
    line4_y = line3_y + pb_h + 10
    s4 = f"bend={bend:.2f} 基线={tm['baseline']:.2f}"
    put_text_cn(frame, s4, (bar_x0, line4_y),
                font_scale=0.42, color=(200, 220, 220), thickness=1)
    # 第5行：阈值 + 累计触发次数
    line5_y = line4_y + 20
    s5 = (f"弯入线={tm['enter_t']:.2f}(红) 伸直线={tm['exit_t']:.2f}(黄)"
          f" 单击×{tm['tap_count']} 长按×{tm['long_count']}")
    put_text_cn(frame, s5, (bar_x0, line5_y),
                font_scale=0.36, color=(160, 180, 200), thickness=1)
    # 第6行（新增）：硬度/速度等级（可观测化反馈，用户一眼知道当前参数档位）
    if det is not None:
        line6_y = line5_y + 20
        hp = tm.get("hardness_pct", 60)
        sp = tm.get("speed_pct", 60)
        enter_r = (det.params.get("ENTER_BEND_RATIO", 0.78) * 100) if det else 78.0
        drop    = (det.params.get("MIN_BEND_DROP",      0.12) * 100) if det else 12.0
        s6 = f"硬度 H={hp:>3d}% 速度 S={sp:>3d}%  弯幅≥{100-enter_r:.0f}%+降{drop:.0f}%"
        put_text_cn(frame, s6, (bar_x0, line6_y),
                    font_scale=0.38, color=(120, 220, 255), thickness=1)


# --------------------------------------------------------------------------
def main():
    # ---- 命令行参数：双模式切换 ----
    parser = argparse.ArgumentParser(description="双手体感鼠标 (DualHand -> Mouse)")
    parser.add_argument(
        "--use-tasks", action="store_true",
        help="使用 Tasks API + 本地 hand_landmarker.task 模型模式（纯离线，不依赖网络）",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="（仅 --use-tasks 时有效）指定 hand_landmarker.task 的绝对/相对路径",
    )
    parser.add_argument(
        "--max-hands", type=int, default=2, help="最多识别手的数量 (默认 2)",
    )
    parser.add_argument(
        "--anchor", type=str, default="palm", choices=["palm", "index_tip"],
        help="鼠标光标锚点：palm=掌心(0+9)/2(推荐，食指点击不漂) / index_tip=食指TIP(原兼容)",
    )
    parser.add_argument(
        "--cursor-mode", type=str, default="smooth", choices=["smooth", "absolute"],
        help="光标移动策略：smooth=AR 5层悬停平滑(稳但滞后) / absolute=掌心→屏幕绝对坐标直接跳(跟手不漂)",
    )
    parser.add_argument(
        "--roi-scale", type=float, default=0.70,
        help="画面→屏幕映射中心区域大小(0.30~1.00)：越小=越大范围给手指伸展(避免掌心锁死时手指不可见); 1.0=全屏映射; 默认0.70=中心70%映射,每边预留15%手活动空间",
    )
    # ✨ 点击调参：两个用户友好 0~1 参数（不用手改 8 个内部阈值）
    parser.add_argument(
        "--tap-hardness", type=float, default=0.70,
        help="点击弯曲幅度门槛 0.00~1.00：越大=需要弯得越狠才触发（小幅度完全不判定为点击，默认 0.70=较严）",
    )
    parser.add_argument(
        "--tap-speed", type=float, default=0.80,
        help="点击识别响应速度 0.00~1.00：越大=越快触发（默认 0.80=响应敏感且仍有防抖）",
    )
    args = parser.parse_args()
    # roi_scale 硬约束
    if args.roi_scale < 0.30: args.roi_scale = 0.30
    if args.roi_scale > 1.00: args.roi_scale = 1.00
    if args.tap_hardness < 0.0: args.tap_hardness = 0.0
    if args.tap_hardness > 1.0: args.tap_hardness = 1.0
    if args.tap_speed    < 0.0: args.tap_speed    = 0.0
    if args.tap_speed    > 1.0: args.tap_speed    = 1.0

    # ---- 摄像头 ----
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头，请检查设备/权限")
    # 降低分辨率可提升 FPS（按需调整）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # ---- 组件：根据参数选择 SOLUTIONS / TASKS_LOCAL_MODEL ----
    detector_kwargs = dict(max_hands=args.max_hands,
                           detection_confidence=0.7,
                           tracking_confidence=0.6)
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
        mode_color = (0, 200, 255)  # 橙黄色
    else:
        detector = HandDetector(**detector_kwargs)
        mode_label = "[MODE: 默认 Solutions]  mediapipe 内置模型"
        mode_color = (180, 255, 120)  # 青绿色
    mouse = MouseController(alpha=0.25, cursor_mode=args.cursor_mode, roi_scale=args.roi_scale)
    tap_detector = FingerTapClickDetector(hardness=args.tap_hardness, speed=args.tap_speed)
    stretch = DiagonalStretchManager()  # 双手对角拉伸 → 选中前台窗口 放大缩小

    combo_hint = ""
    combo_hint_counter = 0
    # ---- FPS 统计（EMA，~30帧平均）----
    _fps_smooth = None
    _fps_last_t = time.perf_counter()
    _fps_frame_cnt = 0
    _fps_display = 0.0
    _fps_cam_read_ms = 0.0       # cap.read 耗时(ms)：直接反映图像采集频率
    _fps_infer_ms = 0.0          # 检测+识别+映射耗时(ms)：MediaPipe计算耗时
    _fps_draw_ms = 0.0           # 绘制+imshow 耗时(ms)
    _t1 = _t2 = _t3 = 0.0

    try:
        while True:
            _t0 = time.perf_counter()
            ok, frame = cap.read()
            _t1 = time.perf_counter()
            if not ok:
                print("摄像头读取失败，退出")
                break

            # ========= 单点镜像（唯一位置）=========
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # ========= 双手检测 =========
            hands_data, frame = detector.detect(frame)

            # 每只手的识别结果缓存，供组合手势判定使用
            # key = "Left"/"Right"  value = (gesture_name, ext_tuple, ratios, lms)
            hand_results = {}

            for hand in hands_data:
                label = hand["label"]           # MediaPipe 原始 Left / Right
                lms = hand["landmarks"]
                gesture, ext, ratios = recognize_gesture(lms)
                hand_results[label] = (gesture, ext, ratios, lms)

                color = HAND_COLORS.get(label, (128, 128, 128))

                # 食指指尖 = 光标锚点，绘制高亮圆
                tip_px, tip_py = HandDetector.landmark_to_pixel(lms[8], w, h)
                cv2.circle(frame, (tip_px, tip_py), 11, color, -1)
                cv2.circle(frame, (tip_px, tip_py), 11, (255, 255, 255), 2)

                # 画面上显示：手的语义标签（把 MediaPipe 的 Left/Right 反转为用户视角）
                user_label = "右手" if label == "Left" else "左手"  # 因镜像，Left=用户右手
                top_y = 38
                hand_title = f"[{user_label}] {gesture}"
                if user_label == "左手":
                    put_text_cn(frame, hand_title, (12, top_y),
                                font_scale=0.85, color=color, thickness=2)
                    _draw_finger_state(frame, ext, ratios, (12, 65), color)
                else:
                    put_text_cn(frame, hand_title, (w - 260, top_y),
                                font_scale=0.85, color=color, thickness=2)
                    _draw_finger_state(frame, ext, ratios, (w - 260, 65), color)

                # ========= 鼠标映射：仅使用右手 =========
                # 🔴 用户新方案：只要「双手出现 + 双手都不是张开手掌(OPEN)」= 纯双手拉伸指令模式，禁用所有单手（非双手）指令
                #   → 按用户要求："双手五指张开与其他两种状态；其他状态暂时视为都是捏合手势" → 非OPEN即视为捏合
                both_hands_now = ("Left" in hand_results) and ("Right" in hand_results)
                left_gesture_now  = hand_results.get("Left",  (None,))[0]
                right_gesture_now = hand_results.get("Right", (None,))[0]
                # ✨ 新捏合判定：非OPEN即视为捏合；None=没识别到手，也算非OPEN（但 both_hands_now=True 时才有效）
                left_pinched_now  = bool(left_gesture_now  is None or left_gesture_now  != "OPEN")
                right_pinched_now = bool(right_gesture_now is None or right_gesture_now != "OPEN")
                is_pure_two_hand_mode = bool(both_hands_now and left_pinched_now and right_pinched_now)
                # 🔴 用户明确：「非OPEN即视为捏合」这个规则 **仅在双手场景生效**；单手场景绝不应用（避免单手做POINT手势被误禁点击）
                #   因此仅保留两类命中条件：(A) 双手同时出现 (B) 拉伸状态机已进入非IDLE (C) pure_two_hand_mode（纯双手拉伸）
                disable_click_for_stretch = (
                    both_hands_now                                              # (A) 双手出现（双手场景下规则生效）
                    or stretch.state != _DS.IDLE                                # (B) 已进入 ARMED/SCALING 拉伸状态机（即便中间有一只手暂丢，继续保持禁点击直到退出SCALING）
                    or is_pure_two_hand_mode                                    # (C) 纯双手拉伸模式（优先级最高，所有事件都禁）
                )
                stretch_click_gestures = ("PEACE", "PINCH", "ROCK", "THUMB", "FIST")   # 这些会触发单击/右键/双击/拖拽
                safe_gesture = gesture
                if disable_click_for_stretch and gesture in stretch_click_gestures:
                    # ✨ 拉伸场景下所有会触发点击的手势 → 降级为 POINT（只移动光标不触发任何点击/拖拽）
                    safe_gesture = "POINT"
                # ✨缩放过程中临时禁用右手鼠标移动，避免跟缩放手势冲突；🔥 纯双手拉伸模式直接不发任何单手事件
                # ✨ 新增：block_mouse_move_for_stretch（含「单手五指张开移动窗口」场景命中）→ 也完全禁 mouse.handle（避免鼠标跟手漂 vs SetCursorPos锁中心 竞争）
                if (not block_mouse_move_for_stretch) and (not is_pure_two_hand_mode) and label == "Left" and stretch.state != _DS.SCALING:
                    # ① 手势识别依旧用食指 TIP 8（fingertip_norm），保持识别逻辑不变
                    fingertip_norm = (lms[8].x, lms[8].y)
                    # ② 实际鼠标光标锚点（掌心或食指）→ 做食指屈伸点击动作时，掌心几乎不动 → 光标不漂！
                    cursor_anchor_norm = HandDetector.cursor_anchor_from_landmarks(lms, args.anchor)
                    mouse.handle(safe_gesture, fingertip_norm, cursor_anchor_norm=cursor_anchor_norm)
                    # ③ 画两个锚点（视觉区分，方便用户理解）：
                    #   - 绿实心圆 + 白边 = 实际鼠标锚点（掌心或食指TIP）
                    #   - 橙空心小圈 = 食指 TIP（仅用于观测，不控制光标，便于确认"食指点击时鼠标确实不动"）
                    mx, my = HandDetector.landmark_to_pixel(lms[8], w, h)
                    cv2.circle(frame, (mx, my), 7, (0, 140, 255), 2)  # 食指 TIP：橙空心
                    # 掌心/实际鼠标锚点
                    ax = int(cursor_anchor_norm[0] * w)
                    ay = int(cursor_anchor_norm[1] * h)
                    cv2.circle(frame, (ax, ay), 12, (255, 255, 255), 2)  # 白外环
                    cv2.circle(frame, (ax, ay), 9, (0, 230, 80), -1)     # 绿实心 = 光标锚点
                elif label == "Left":
                    # SCALING 状态下只画锚点（视觉仍有反馈，但不发鼠标事件）
                    fingertip_norm = (lms[8].x, lms[8].y)
                    cursor_anchor_norm = HandDetector.cursor_anchor_from_landmarks(lms, args.anchor)
                    mx, my = HandDetector.landmark_to_pixel(lms[8], w, h)
                    cv2.circle(frame, (mx, my), 7, (0, 140, 255), 2)
                    ax = int(cursor_anchor_norm[0] * w)
                    ay = int(cursor_anchor_norm[1] * h)
                    cv2.circle(frame, (ax, ay), 12, (255, 255, 255), 2)
                    cv2.circle(frame, (ax, ay), 9, (0, 230, 80), -1)

            # ========= ✨ 需求+用户新方案：点击/单手移动完全禁用判定 =========
            #   🔴 纯双手拉伸模式（双手出现 + 双手都非OPEN = 非OPEN视为捏合）→ 禁用一切非双手指令
            #   🔴 纯双手移动模式（双手出现 + 双手都OPEN五指张开 = MOVE移动窗口）→ 同样禁用一切点击/非双手指令
            left_gesture_global   = hand_results.get("Left",  (None,))[0]
            right_gesture_global  = hand_results.get("Right", (None,))[0]
            # ✨ 新捏合判定：非OPEN即视为捏合（None=未识别到也归为非捏合，不干扰 any 判定）
            left_pinch_global  = bool(left_gesture_global  is not None and left_gesture_global  != "OPEN")
            right_pinch_global = bool(right_gesture_global is not None and right_gesture_global != "OPEN")
            left_open_global   = bool(left_gesture_global  == "OPEN")
            right_open_global  = bool(right_gesture_global == "OPEN")
            # ✨ 单手场景判断：恰好存在一只手 + 那只手 OPEN → 也触发「单手五指张开=移动窗口」
            single_hand_exists = bool(("Left" in hand_results) ^ ("Right" in hand_results))  # 异或=恰好一只
            single_open_global = bool(single_hand_exists and ((left_open_global and ("Left" in hand_results)) or (right_open_global and ("Right" in hand_results))))
            both_hands_global     = ("Left" in hand_results) and ("Right" in hand_results)   # 用 hand_results 键判（此时 left_lms_main 尚未定义，安全）
            pure_two_hand_stretch = bool(both_hands_global and left_pinch_global and right_pinch_global)
            pure_two_hand_move    = bool(both_hands_global and left_open_global  and right_open_global)
            two_hand_any_active   = bool(pure_two_hand_stretch or pure_two_hand_move)
            # 🔴 移动/缩放任一状态机处于非IDLE/任一模式门控命中 → 禁用一切点击（避免手指弯曲误触）
            # ✨ 新增：single_open_global → 单手五指张开=移动窗口，也禁一切点击+禁鼠标移动（光标锁窗口中心=不跟手漂）
            disable_click_for_stretch_global = (
                both_hands_global                             # (A) 双手出现（双手场景下，「非OPEN即捏合」规则才生效）
                or single_open_global                         # (A') 恰好单手出现，且那只手是五指张开（移动窗口指令 → 禁点击）
                or stretch.state != _DS.IDLE                 # (B1) SCALE 状态机已激活
                or stretch._move_state != _DMS.IDLE          # (B2) MOVE 状态机已激活（移动窗口中）
                or pure_two_hand_stretch                     # (C1) 纯双手拉伸模式（优先级最高，禁用一切）
                or pure_two_hand_move                        # (C2) 纯双手移动模式（五指张开=移动窗口，禁用点击，防误触）
            )
            # 🔴 move_enabled_false_when_stretch：命中时 mouse.handle 不做鼠标移动 → 防止 移动窗口 时，鼠标还在被手掌移动牵着跑（会和SetCursorPos锁中心竞争）
            block_mouse_move_for_stretch = disable_click_for_stretch_global

            # ========= ✨ 食指 Tap-Click（仅动手指，手掌不动）：仅追踪右手 =========
            # bend = 食指伸直度 = dist(TIP_8, MCP_5) / dist(MCP_5, MCP_17)  归一化鲁棒
            index_bend: float | None = None
            right_screen_pt = None
            if "Left" in hand_results:  # MediaPipe Left = 用户右手
                _, _, _, lms = hand_results["Left"]
                # 用归一化坐标做归一化 → 对手远近完全鲁棒，也完全不依赖手掌是否运动
                tip = np.array([lms[8].x, lms[8].y])
                m5  = np.array([lms[5].x, lms[5].y])
                m17 = np.array([lms[17].x, lms[17].y])
                palm_n = float(np.linalg.norm(m5 - m17))
                tip_m5 = float(np.linalg.norm(tip - m5))
                if palm_n > 1e-6:
                    index_bend = tip_m5 / palm_n
                # 屏幕坐标（供长按拖拽用）→ 使用与 mouse.handle 相同的掌心锚点！
                anchor_norm = HandDetector.cursor_anchor_from_landmarks(lms, args.anchor)
                gx, gy = mouse._norm_to_screen(*anchor_norm)
                sx, sy = mouse._smooth(gx, gy)
                right_screen_pt = (sx, sy)

            # 🔴 核心：只要 disable_click_for_stretch_global 命中 → 立即 reset detector + 事件清零（彻底不产生点击）
            if disable_click_for_stretch_global:
                tap_detector.reset()
                tap_event = 0
            else:
                tap_event = tap_detector.update(index_bend)
            # 用户要求恢复：点击前锁死光标位置3帧（TAP和LONG_DOWN都锁一次，避免点击瞬间漂移）
            # ✨ 拉伸相关场景一律不触发任何食指点击事件
            if not disable_click_for_stretch_global and stretch.state != _DS.SCALING:
                if tap_event in (1, 2):  # E.TAP_CLICK=1 / LONG_DOWN=2
                    mouse._smoother.freeze_for_click()
                if tap_event != 0:
                    mouse.handle_finger_tap_event(tap_event, right_screen_pt)

            # ========= ✨ 双手捏合拉伸 → 选中前台窗口 放大缩小 =========
            # ✨ 注意：必须放在全局点击禁用判定之后（保证 left_lms_main / right_lms_main 被定义后，再交给 stretch.update）
            # ✨ 新增：left_palm_screen / right_palm_screen → 掌心锚点的**屏幕绝对坐标**，供MOVE模式(五指张开移动窗口)找掌心下窗口用
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
                # 掌心屏幕坐标 = mouse 的 norm→screen 映射（与鼠标光标移动完全同坐标系）
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

            _t2 = time.perf_counter()   # 检测 + 识别 + 映射链路 结束

            # ========= 绘制：食指 Tap 条（右上角，替代原深度条）=========
            tm = tap_detector.metrics()
            _draw_tap_bar(frame, tm, tap_detector)

            # ========= ✨ 绘制：双手捏合拉伸提示 / 缩放比例 OR 双手张开移动窗口提示 =========
            #   🔴 两种模式互斥：SCALE=双手非OPEN(捏合) 绘制原连线/θ分档；MOVE=双手OPEN(五指张开) 绘制绿色移动框/ΔxΔy卡片
            mode_now = getattr(stretch_info, 'mode', 'SCALE')
            # -------- 模式一：SCALE（拉伸缩放）UI 保持原逻辑 --------
            if mode_now == _DM.SCALE and (stretch_info.need_frame_redraw or stretch_info.ui_pinch_active):
                p1, p2 = stretch_info.p1, stretch_info.p2
                if p1 is not None and p2 is not None:
                    x1, y1 = int(p1[0]), int(p1[1])
                    x2, y2 = int(p2[0]), int(p2[1])
                    # 两掌心连线中点（=画卡片/进度条的中心锚点，不管方向都居中）
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    # ✨✨ UI预判定：即便 state=IDLE（还没进确认帧），只要 ui_pinch_active=True 就先算θ+分档，让用户直观知道当前连线角度属于哪档拉伸
                    #  计算：θ_deg = atan2(|dy|,|dx|) → 0~90°；区间映射与 diagonal_stretch 完全一致
                    import math as _mt_tmp
                    dx_abs = abs(x2 - x1); dy_abs = abs(y2 - y1)
                    theta_deg = abs(_mt_tmp.degrees(_mt_tmp.atan2(float(dy_abs), float(max(dx_abs, 1)))))
                    if theta_deg <= 20.0:
                        pre_mode = "水平(拉宽/拉窄)"; pre_c = (40, 180, 255)
                    elif theta_deg >= 70.0:
                        pre_mode = "垂直(拉高/拉低)"; pre_c = (80, 255, 150)
                    else:
                        pre_mode = "对角(等比缩放)"; pre_c = (220, 120, 255)
                    # 辅助：在指尖画捏合小圆（给用户直观看到"捏合是否成功"）
                    def _draw_pinch_dot(lms, px, py, color):
                        if lms is None:
                            return
                        t4 = (int(float(lms[4].x) * w), int(float(lms[4].y) * h))
                        t8 = (int(float(lms[8].x) * w), int(float(lms[8].y) * h))
                        cv2.line(frame, t4, t8, color, 2)
                        cv2.circle(frame, t4, 7, color, -1)   # 拇指点（实心）
                        cv2.circle(frame, t8, 7, color, -1)   # 食指点（实心）
                        # 掌心附近标注"捏合比"
                        put_text_cn(frame,
                                    f"捏: {min(99, int(stretch_info.left_pinch_r * 100))}%" if px < w // 2 else f"捏: {min(99, int(stretch_info.right_pinch_r * 100))}%",
                                    (max(4, px - 32), max(4, py + 32)), font_scale=0.5, color=color, thickness=2)
                    # (A) state=IDLE + ui_pinch_active → 预判定阶段（还没激活状态机，但先显示连线和预模式θ）
                    # (B) state=ARMED_CONFIRM → 仍在 3 帧确认中
                    if (stretch_info.state == _DS.ARMED_CONFIRM) or (stretch_info.ui_pinch_active and stretch_info.state == _DS.IDLE):
                        color = (255, 210, 80)   # 青蓝色：选中确认中 / 预判定（同一色系，视觉稳定）
                        thick = 2
                        # 两个掌心上各自画大圈 + 连线 → 🔴 只要 ui_pinch_active=True 就立刻出现连线，哪怕还没进状态机确认帧
                        cv2.circle(frame, (x1, y1), 22, color, 3)
                        cv2.circle(frame, (x2, y2), 22, color, 3)
                        cv2.line(frame, (x1, y1), (x2, y2), color, 2)
                        _draw_pinch_dot(left_lms_main,  x1, y1, color)
                        _draw_pinch_dot(right_lms_main, x2, y2, color)
                        # 连线中点正上方：预判定模式 + θ / 确认进度
                        hold_n = stretch.min_hold_frames
                        pct = stretch_info.confirm_frames / max(1, hold_n)
                        pct = 1.0 if pct > 1.0 else pct
                        prog_w = 340
                        bx = cx - prog_w // 2
                        by = cy - 62
                        cv2.rectangle(frame, (bx - 6, by - 24), (bx + prog_w + 6, by + 48), (0, 0, 0), -1)
                        cv2.rectangle(frame, (bx, by), (bx + prog_w, by + 12), (60, 60, 60), -1)
                        if stretch_info.state == _DS.IDLE and stretch_info.ui_pinch_active:
                            # 🔴 预判定阶段：按θ分档画彩色进度条 + 明确文字提示当前档位
                            fill_ratio = min(1.0, theta_deg / 90.0)
                            cv2.rectangle(frame, (bx, by), (bx + int(prog_w * fill_ratio), by + 12), pre_c, -1)
                            put_text_cn(frame,
                                        f"预判定【{pre_mode}】  θ={theta_deg:.0f}° ｜ 双手≥120px 并保持3帧即可激活",
                                        (bx, by + 20), font_scale=0.58, color=(255, 255, 255), thickness=2)
                        else:
                            # ARMED_CONFIRM 正常确认中：原帧进度条 + 附加显示θ与预模式
                            cv2.rectangle(frame, (bx, by), (bx + int(prog_w * pct), by + 12), color, -1)
                            put_text_cn(frame,
                                        f"选中确认中… {stretch_info.confirm_frames}/{hold_n}({int(pct*100)}%)  θ={theta_deg:.0f}° → {pre_mode}",
                                        (bx, by + 20), font_scale=0.6, color=(255, 255, 255), thickness=2)
                    elif stretch_info.state == _DS.SCALING:
                        color = (80, 230, 255)   # 亮黄：正在缩放
                        # 两个掌心大圈 + 粗连线（删除原对角实心矩形，因为方向任意，矩形无意义）
                        cv2.circle(frame, (x1, y1), 22, color, 4)
                        cv2.circle(frame, (x2, y2), 22, color, 4)
                        cv2.line(frame, (x1, y1), (x2, y2), color, 3)
                        _draw_pinch_dot(left_lms_main,  x1, y1, color)
                        _draw_pinch_dot(right_lms_main, x2, y2, color)
                        # 连线中点正上方大卡片：拉伸模式 + 动作 + 缩放比 + HWND + size
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
                        # 估算卡片宽高
                        card_w, card_h = 520, 100
                        bx = cx - card_w // 2
                        by = cy - card_h // 2 - 16
                        cv2.rectangle(frame, (bx, by), (bx + card_w, by + card_h), (0, 0, 0), -1)
                        cv2.rectangle(frame, (bx, by), (bx + card_w, by + card_h), color, 2)
                        put_text_cn(frame, lines[0][0], (bx + 24, by + 14), font_scale=lines[0][2], color=lines[0][1], thickness=2)
                        put_text_cn(frame, lines[1][0], (bx + 24, by + 44), font_scale=lines[1][2], color=lines[1][1], thickness=3)
                        put_text_cn(frame, lines[2][0], (bx + 24, by + 82), font_scale=lines[2][2], color=lines[2][1], thickness=2)

            # -------- 模式二：MOVE（单手五指张开/双手五指张开=移动窗口）UI 新增 --------
            if mode_now == _DM.MOVE and (stretch_info.move_need_frame_redraw or stretch_info.ui_move_active):
                hands_mode_now = getattr(stretch_info, 'hands_mode', 'DUAL')   # SINGLE/DUAL
                # 画面掌心锚点：优先用实际 lms 算；兜底用 stretch_info.p1/p2
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

                # -------- 掌心画"五指张开"的提示手势（散开花瓣圆点）共用 --------
                def _draw_open_palm_marks(lms, px, py):
                    if lms is None: return
                    for tip_i in (4, 8, 12, 16, 20):
                        try:
                            tx = int(float(lms[tip_i].x) * w)
                            ty = int(float(lms[tip_i].y) * h)
                            cv2.circle(frame, (tx, ty), 8, mv_color, -1)
                        except Exception:
                            pass
                    put_text_cn(frame, "张开",
                                (max(4, px - 26), max(4, py + 38)), font_scale=0.55, color=mv_color, thickness=2)

                # ---------- DUAL（双手五指张开）：两只手都存在 → 画双手+连线 ----------
                if hands_mode_now != "SINGLE" and p1_m is not None and p2_m is not None:
                    x1, y1 = int(p1_m[0]), int(p1_m[1])
                    x2, y2 = int(p2_m[0]), int(p2_m[1])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    # 两个掌心大圈 + 连线
                    cv2.circle(frame, (x1, y1), 24, mv_color, 3)
                    cv2.circle(frame, (x2, y2), 24, mv_color, 3)
                    cv2.line(frame, (x1, y1), (x2, y2), mv_color, 2)
                    _draw_open_palm_marks(llms_ui,  x1, y1)
                    _draw_open_palm_marks(rlms_ui, x2, y2)

                    # (1) 确认中 / IDLE预判定 → 进度条
                    if stretch_info.move_state == _DMS.MOVE_CONFIRM or (stretch_info.ui_move_active and stretch_info.move_state == _DMS.IDLE):
                        hold_n = 3
                        pct = stretch_info.move_confirm_frames / hold_n
                        pct = 1.0 if pct > 1.0 else pct
                        prog_w = 400
                        bx = cx - prog_w // 2
                        by = cy - 66
                        cv2.rectangle(frame, (bx - 6, by - 30), (bx + prog_w + 6, by + 60), (0, 0, 0), -1)
                        cv2.rectangle(frame, (bx, by), (bx + prog_w, by + 14), (50, 50, 50), -1)
                        if stretch_info.ui_move_active and stretch_info.move_state == _DMS.IDLE:
                            cv2.rectangle(frame, (bx, by), (bx + int(prog_w * 0.02), by + 14), mv_color, -1)
                            put_text_cn(frame,
                                        f"✋预判定【移动窗口(双手)】左手掌心指向要移动的窗口 ｜ 保持五指张开3帧即可选中",
                                        (bx, by + 24), font_scale=0.55, color=(255, 255, 255), thickness=2)
                        else:
                            cv2.rectangle(frame, (bx, by), (bx + int(prog_w * pct), by + 14), mv_color, -1)
                            put_text_cn(frame,
                                        f"✋选中窗口确认中… {stretch_info.move_confirm_frames}/{hold_n}({int(pct*100)}%)  → 左手掌心下窗口=目标",
                                        (bx, by + 24), font_scale=0.58, color=(255, 255, 255), thickness=2)
                    # (2) MOVE_ACTIVE → 大卡片
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
                        cv2.rectangle(frame, (bx, by), (bx + card_w, by + card_h), (0, 0, 0), -1)
                        cv2.rectangle(frame, (bx, by), (bx + card_w, by + card_h), mv_color, 2)
                        put_text_cn(frame, lines[0][0], (bx + 22, by + 14), font_scale=lines[0][2], color=lines[0][1], thickness=2)
                        put_text_cn(frame, lines[1][0], (bx + 22, by + 46), font_scale=lines[1][2], color=lines[1][1], thickness=2)
                        put_text_cn(frame, lines[2][0], (bx + 22, by + 86), font_scale=lines[2][2], color=lines[2][1], thickness=2)

                # ---------- SINGLE（单手五指张开）：只有一只手存在 → 画单掌心大圈（不要连线，卡片锚点在手的右下/右上避开手） ----------
                if hands_mode_now == "SINGLE" and p1_m is not None:
                    x1, y1 = int(p1_m[0]), int(p1_m[1])
                    # 单手：用该手掌心当锚点；卡片锚点 cx,cy = 手的位置（在手右侧/下方做卡片，避免挡住手）
                    cx = x1
                    cy = y1
                    # 掌心大圈 + 花瓣
                    cv2.circle(frame, (x1, y1), 26, mv_color, 3)
                    _draw_open_palm_marks(llms_ui if ("Left" in hand_results) else rlms_ui, x1, y1)

                    # (1) 确认中 / IDLE预判定 → 进度条（卡片锚点在手的右侧或左侧，避免挡住手）
                    if stretch_info.move_state == _DMS.MOVE_CONFIRM or (stretch_info.ui_move_active and stretch_info.move_state == _DMS.IDLE):
                        hold_n = 3
                        pct = stretch_info.move_confirm_frames / hold_n
                        pct = 1.0 if pct > 1.0 else pct
                        prog_w = 420
                        # 手在画面左半：卡片放在手右侧；手在右半：放在左侧
                        bx = x1 + 40 if cx < w // 2 else (x1 - 40 - prog_w)
                        by = y1 - 66
                        cv2.rectangle(frame, (bx - 6, by - 30), (bx + prog_w + 6, by + 60), (0, 0, 0), -1)
                        cv2.rectangle(frame, (bx, by), (bx + prog_w, by + 14), (50, 50, 50), -1)
                        if stretch_info.ui_move_active and stretch_info.move_state == _DMS.IDLE:
                            cv2.rectangle(frame, (bx, by), (bx + int(prog_w * 0.02), by + 14), mv_color, -1)
                            put_text_cn(frame,
                                        f"✋预判定【移动窗口(单手)】掌心指向要移动的窗口 ｜ 保持五指张开3帧即可选中",
                                        (bx, by + 24), font_scale=0.55, color=(255, 255, 255), thickness=2)
                        else:
                            cv2.rectangle(frame, (bx, by), (bx + int(prog_w * pct), by + 14), mv_color, -1)
                            put_text_cn(frame,
                                        f"✋选中窗口确认中(单手)… {stretch_info.move_confirm_frames}/{hold_n}({int(pct*100)}%)",
                                        (bx, by + 24), font_scale=0.58, color=(255, 255, 255), thickness=2)
                    # (2) MOVE_ACTIVE → 大卡片（单手）
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
                        cv2.rectangle(frame, (bx, by), (bx + card_w, by + card_h), (0, 0, 0), -1)
                        cv2.rectangle(frame, (bx, by), (bx + card_w, by + card_h), mv_color, 2)
                        put_text_cn(frame, lines[0][0], (bx + 22, by + 14), font_scale=lines[0][2], color=lines[0][1], thickness=2)
                        put_text_cn(frame, lines[1][0], (bx + 22, by + 46), font_scale=lines[1][2], color=lines[1][1], thickness=2)
                        put_text_cn(frame, lines[2][0], (bx + 22, by + 86), font_scale=lines[2][2], color=lines[2][1], thickness=2)
            # 左上角：缩放/移动状态提示（单手/无手场景也能看）
            sst = stretch_info.state
            msst = stretch_info.move_state
            hands_mode_now = getattr(stretch_info, 'hands_mode', 'DUAL')
            # 优先显示 MOVE（移动窗口）状态
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
                cv2.rectangle(frame, (8, 90), (8 + 470, 90 + 28), (0, 0, 0), -1)
                put_text_cn(frame, s_text, (14, 94), font_scale=0.6, color=s_color, thickness=2)

            # ========= 双手组合手势检测 =========
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
                combo_hint_counter = 60  # 保持显示 60 帧
            elif combo_hint_counter > 0:
                combo_hint_counter -= 1

            if combo_hint_counter > 0:
                tw, th = _measure_cn_text_size(combo_hint, 0.8)
                tx = (w - tw) // 2
                ty = h - 30
                cv2.rectangle(frame, (tx - 12, ty - th - 14),
                              (tx + tw + 12, ty + 12),
                              (0, 0, 0), -1)
                put_text_cn(frame, combo_hint, (tx, ty),
                            font_scale=0.8, color=(0, 230, 230), thickness=2)

            # ========= 顶部模式标签 =========
            anchor_text = "掌心" if args.anchor == "palm" else "食指尖"
            mode_text   = "绝对指哪打哪" if args.cursor_mode == "absolute" else "AR 悬停平滑"
            roi_pct = int(round(args.roi_scale * 100))
            cv2.rectangle(frame, (0, 0), (w, 30), (0, 0, 0), -1)
            put_text_cn(frame, mode_label, (10, 6),
                        font_scale=0.6, color=mode_color, thickness=2)
            put_text_cn(frame, f"🖱️{mode_text} | 锚点: {anchor_text} | 映射ROI: {roi_pct}%  🟢=光标  🟠=食指TIP",
                        (w - 640, 6), font_scale=0.6, color=(0, 230, 80), thickness=2)
            # ========= 绘制：ROI 矩形边框（把"画面中心X%映射全屏"可视化，用户一眼能看到允许掌心活动的区域）=========
            margin_px = int(round((1.0 - args.roi_scale) * 0.5 * min(w, h)))
            x1 = margin_px
            y1 = margin_px
            x2 = w - margin_px
            y2 = h - margin_px
            # 虚线绿色边框：内部=可映射到屏幕的掌心活动区域；外部=手指预留伸展空间
            _dash_len = 12
            _gap = 8
            _step = _dash_len + _gap
            # 顶边
            for x in range(x1, x2, _step):
                xe = min(x + _dash_len, x2)
                cv2.line(frame, (x, y1), (xe, y1), (0, 220, 120), 2)
            # 底边
            for x in range(x1, x2, _step):
                xe = min(x + _dash_len, x2)
                cv2.line(frame, (x, y2), (xe, y2), (0, 220, 120), 2)
            # 左边
            for y in range(y1, y2, _step):
                ye = min(y + _dash_len, y2)
                cv2.line(frame, (x1, y), (x1, ye), (0, 220, 120), 2)
            # 右边
            for y in range(y1, y2, _step):
                ye = min(y + _dash_len, y2)
                cv2.line(frame, (x2, y), (x2, ye), (0, 220, 120), 2)

            # ========= 底部提示 =========
            help_line = "右手=控鼠 | 🟢掌心移动=鼠标(食指点击不漂) | 食指轻弯=单击 | 保弯=拖 | 🤏V=单 | Rock=右 | 拇=双 | 拳=拖 | 🤏双手非张开=捏合拉伸→窗口放大缩小 | ✋单手五指张开=掌心下窗口拖动 | ✋双手五指张开=双手配合拖动窗口(光标锁窗口中心防误触) | q退"
            put_text_cn(frame, help_line, (12, h - 28),
                        font_scale=0.55, color=(220, 220, 220), thickness=2)
            _t3 = time.perf_counter()   # 绘制结束

            # ========= FPS + 链路耗时（右上角锚点提示下方）=========
            _fps_frame_cnt += 1
            _now = time.perf_counter()
            _dt = _now - _fps_last_t
            if _dt >= 0.5:   # 每 0.5 秒更新一次显示值（避免刷屏跳）
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
            fps_txt = f"FPS {_fps_display:5.1f}  采集{_fps_cam_read_ms:5.1f}ms 推理{_fps_infer_ms:5.1f}ms 绘制{_fps_draw_ms:5.1f}ms"
            # 背景条
            cv2.rectangle(frame, (w - 448, 30), (w - 4, 60), (0, 0, 0), -1)
            fps_color = (0, 230, 80) if _fps_display >= 25 else (80, 180, 255) if _fps_display >= 18 else (80, 80, 255)
            put_text_cn(frame, fps_txt, (w - 440, 36),
                        font_scale=0.55, color=fps_color, thickness=2)

            cv2.imshow("DualHand -> Mouse (双手体感鼠标)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        mouse.reset()
        try:
            detector.close()
        except Exception:
            pass
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
