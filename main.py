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
        "--roi-scale", type=float, default=0.85,
        help="画面→屏幕映射中心区域大小(0.30~1.00)：越小=越大范围给手指伸展(避免掌心锁死时手指不可见); 1.0=全屏映射",
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
                # 注意：这里使用 MediaPipe 原始标签判断。镜像后：
                #   用户右手 = 画面右侧 = MediaPipe label "Left"
                if label == "Left":
                    # ① 手势识别依旧用食指 TIP 8（fingertip_norm），保持识别逻辑不变
                    fingertip_norm = (lms[8].x, lms[8].y)
                    # ② 实际鼠标光标锚点（掌心或食指）→ 做食指屈伸点击动作时，掌心几乎不动 → 光标不漂！
                    cursor_anchor_norm = HandDetector.cursor_anchor_from_landmarks(lms, args.anchor)
                    mouse.handle(gesture, fingertip_norm, cursor_anchor_norm=cursor_anchor_norm)
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

            tap_event = tap_detector.update(index_bend)
            # 用户要求恢复：点击前锁死光标位置3帧（TAP和LONG_DOWN都锁一次，避免点击瞬间漂移）
            if tap_event in (1, 2):  # E.TAP_CLICK=1 / LONG_DOWN=2
                mouse._smoother.freeze_for_click()
            if tap_event != 0:
                mouse.handle_finger_tap_event(tap_event, right_screen_pt)
            _t2 = time.perf_counter()   # 检测 + 识别 + 映射链路 结束

            # ========= 绘制：食指 Tap 条（右上角，替代原深度条）=========
            tm = tap_detector.metrics()
            _draw_tap_bar(frame, tm, tap_detector)

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
            help_line = "右手=控鼠 | 🟢掌心移动=鼠标(食指点击不漂) | 食指轻弯=单击 | 保弯=拖 | 🤏V=单 | Rock=右 | 拇=双 | 拳=拖 | q退"
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
