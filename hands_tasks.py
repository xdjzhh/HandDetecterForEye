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
import os
import time
import cv2
import numpy as np

from hand_detector import HandDetector, TasksHandDetector, HAND_CONNECTIONS
from gesture_recognizer import recognize_gesture
from mouse_controller import MouseController
from draw_cn_text import put_text_cn
from finger_tap_click_detector import FingerTapClickDetector, _S as _TS
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
    parser.add_argument("--cursor-mode", type=str, default="smooth", choices=["smooth", "absolute"],
                        help="光标移动策略：smooth / absolute")
    parser.add_argument("--roi-scale", type=float, default=0.70,
                        help="画面→屏幕映射中心区域大小(0.30~1.00)")
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

    mouse = MouseController(alpha=0.25, cursor_mode=args.cursor_mode, roi_scale=args.roi_scale)
    tap_detector = FingerTapClickDetector(hardness=args.tap_hardness, speed=args.tap_speed)
    stretch = DiagonalStretchManager()

    return dict(
        args=args,
        detector=detector,
        mouse=mouse,
        tap_detector=tap_detector,
        stretch=stretch,
        mode_label=mode_label,
        mode_color=mode_color,
    )


# ======================================================================
# ④ run_loop(components, camera_id, stop_event, preview, status_store)
#    完全对应原 main.py L239 (cap = VideoCapture) → L908 (finally destroyAllWindows)
# ======================================================================
def run_loop(components, camera_id=0, stop_event=None, preview=True, status_store=None):
    """
    阻塞式识别主循环。

    参数：
      components   : init_models() 返回的 dict（每次 /run 建议传新的组件，避免复用 closed detector）
      camera_id    : 摄像头 ID，默认 0
      stop_event   : threading.Event；每帧开头先检查；set() 时立即干净退出（可 None = 纯脚本模式，靠 q 键退出）
      preview      : True=开 cv2.imshow 预览窗口；False=纯后台识别（不画窗）
      status_store : dict，写入实时状态（server.py 的 /status 接口直接读它）；None 时内部自动建一个（没用）

    写入 status_store 的字段（都是最简单的标量，读写无锁）：
      running            : bool   —— 是否处于主循环中（正常跑）
      fps                : float  —— 最近 ~0.5s 的 FPS（EMA 平滑）
      current_camera_id  : int
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
    stretch         = components["stretch"]
    mode_label      = components["mode_label"]
    mode_color      = components["mode_color"]

    # ---------- 初始化 status_store ----------
    if status_store is None:
        status_store = {}
    start_ms = int(time.time() * 1000)
    status_store.clear()
    status_store.update(dict(
        running=False, fps=0.0, current_camera_id=int(camera_id),
        start_time_ms=start_ms, uptime_ms=0,
        last_left_gesture=None, last_right_gesture=None,
        stretch_mode="IDLE", error=None,
        preview_active=bool(preview),
    ))

    combo_hint = ""
    combo_hint_counter = 0
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
    #     α=0.4 → 兼顾跟手+稳定（越小越平滑，越大越跟手）
    _EMA_ALPHA = 0.45
    _ema_landmarks: dict[str, list] = {}   # key = Left/Right, value = 21 个 [x,y,z] 列表
    _ema_miss_count: dict[str, int] = {}   # 连续丢失计数，超过阈值清空 EMA

    # (2) 手势标签 majority-vote 去抖：最近 5 帧投票取众数
    _MV_WINDOW = 5
    _mv_history: dict[str, list] = {}      # key = Left/Right, value = list[str] (gesture names)
    _mv_final: dict[str, str] = {}

    # (3) 自适应跳帧：
    #   · 连续 ≥_SKIP_MIN_STABLE 帧都稳定检测到手 + 掌心位移 < _SKIP_PX_TOL*min(w,h)
    #     → 下一帧**跳过推理**，直接复用上次结果 + EMA 预测延续
    #   · 好处：推理频率接近减半，FPS 明显提升；代价：延迟 +1 帧（16~20ms，人眼无感）
    #   · 任何手丢失 / 手势切换 / 掌心大跳 → 立即退出跳帧模式
    _SKIP_MIN_STABLE = 6
    _SKIP_PX_TOL = 0.012                    # 归一化位移：0.012 ≈ 1280×720 下短边720 → ≈ 8.6 像素
    _skip_stable_cnt = 0                    # 连续稳定帧计数
    _skip_last_hands_data = None            # 上次的 hands_data（深拷贝后备用）
    _skip_last_palms: dict[str, tuple] = {} # 上次掌心（归一化）：判断是否稳定
    _skip_skip_next = False                 # 下帧是否真的跳过推理
    _skip_was_skipped = False               # 刚刚发生了跳帧（写 status_store 用）
    _skip_total_frames = 0                  # 累计主循环帧数
    _skip_skipped_frames = 0                # 累计跳过推理帧数

    cap = None
    try:
        # ---------- 打开摄像头（Windows 外置 USB 摄像头必须探测 CAP_DSHOW 后端） ----------
        _cid = int(camera_id)
        cap = None
        _probe_log = []
        _opened = False
        # 优先尝试 DSHOW（外置 USB 摄像头在 Windows 上几乎必须用这个，否则要么打不开，要么 0.5fps 巨卡）
        for _backend_name, _backend in [("DSHOW", cv2.CAP_DSHOW), ("DEFAULT", 0)]:
            try:
                t0 = time.perf_counter()
                cap = cv2.VideoCapture(_cid, _backend) if _backend != 0 else cv2.VideoCapture(_cid)
                ok_open = cap is not None and cap.isOpened()
                t_open = (time.perf_counter() - t0) * 1000.0
                ok_set_w = cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) if ok_open else False
                ok_set_h = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) if ok_open else False
                ok_read, _ = cap.read() if ok_open else (False, None)
                _probe_log.append(f"[{_backend_name}({_backend})#try1] opened={ok_open} bp={_backend}.0 set1280x720=({ok_set_w},{ok_set_h}) read1frame={ok_read} elapsed_ms={int(t_open)}")
                if ok_open and ok_read:
                    _opened = True
                    _opened_backend = _backend_name
                    print(f"[hands_tasks.run_loop] ✅ 摄像头 camera_id={_cid} 打开成功（后端={_backend_name}, 耗时={int(t_open)}ms）")
                    break
                else:
                    try: cap.release()
                    except Exception: pass
                    cap = None
            except Exception as _e:
                _probe_log.append(f"[{_backend_name}] exception: {_e}")
                try: cap.release()
                except Exception: pass
                cap = None
        if not _opened or cap is None:
            msg = f"无法打开摄像头 camera_id={_cid}，尝试后端: {'; '.join(_probe_log)}"
            status_store["error"] = msg
            if status_store is not None:
                try: status_store["_diag_backend_probe"] = dict(attempts_summary=_probe_log, camera_id=_cid, success_attempt=0 if not _opened else len(_probe_log), success_backend=_opened_backend if _opened else None, total_elapsed_ms=0)
                except Exception: pass
            raise RuntimeError(msg)
        # 把探测结果写到 status_store（/status 接口可查看排障）
        if status_store is not None:
            try:
                # 🔴 实际读取一下摄像头设置后的真实分辨率（有些摄像头会悄悄 fallback）
                actual_w = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps_req = float(cap.get(cv2.CAP_PROP_FPS))
                status_store["_diag_backend_probe"] = dict(
                    attempts_summary=_probe_log, camera_id=_cid,
                    success_attempt=len(_probe_log), success_backend=_opened_backend,
                    total_elapsed_ms=0,
                    actual_frame_w=int(actual_w),
                    actual_frame_h=int(actual_h),
                    actual_fps=float(actual_fps_req),
                )
                print(f"[hands_tasks.run_loop] 摄像头实际分辨率 = {int(actual_w)}×{int(actual_h)}  标称FPS = {actual_fps_req:.1f}")
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
                if _lab in _ema_landmarks:
                    _last_arr = _ema_landmarks[_lab]
                    _smoothed = [
                        [_EMA_ALPHA * _cur_arr[i][0] + (1 - _EMA_ALPHA) * _last_arr[i][0],
                         _EMA_ALPHA * _cur_arr[i][1] + (1 - _EMA_ALPHA) * _last_arr[i][1],
                         _EMA_ALPHA * _cur_arr[i][2] + (1 - _EMA_ALPHA) * _last_arr[i][2]]
                        for i in range(21)
                    ]
                else:
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
                or single_open_global
                or stretch.state != _DS.IDLE
                or stretch._move_state != _DMS.IDLE
                or pure_two_hand_stretch
                or pure_two_hand_move
                or any_fist_global
                or any_near_fist_global
            )
            block_mouse_move_for_stretch = disable_click_for_stretch_global

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

                # --- 鼠标事件（只发给右手 = MediaPipe Left） ---
                if (not block_mouse_move_for_stretch) and (not is_pure_two_hand_mode) and label == "Left" and stretch.state != _DS.SCALING:
                    fingertip_norm = (lms[8].x, lms[8].y)
                    cursor_anchor_norm = HandDetector.cursor_anchor_from_landmarks(lms, args.anchor)
                    mouse.handle(safe_gesture, fingertip_norm, cursor_anchor_norm=cursor_anchor_norm)
                    mx, my = HandDetector.landmark_to_pixel(lms[8], w, h)
                    cv2.circle(frame, (mx, my), 7, (0, 140, 255), 2)
                    ax = int(cursor_anchor_norm[0] * w)
                    ay = int(cursor_anchor_norm[1] * h)
                    cv2.circle(frame, (ax, ay), 12, (255, 255, 255), 2)
                    cv2.circle(frame, (ax, ay), 9, (0, 230, 80), -1)
                elif label == "Left":
                    # SCALING / 门控命中 → 不发鼠标事件，但仍画锚点（保持视觉反馈）
                    fingertip_norm = (lms[8].x, lms[8].y)
                    cursor_anchor_norm = HandDetector.cursor_anchor_from_landmarks(lms, args.anchor)
                    mx, my = HandDetector.landmark_to_pixel(lms[8], w, h)
                    cv2.circle(frame, (mx, my), 7, (0, 140, 255), 2)
                    ax = int(cursor_anchor_norm[0] * w)
                    ay = int(cursor_anchor_norm[1] * h)
                    cv2.circle(frame, (ax, ay), 12, (255, 255, 255), 2)
                    cv2.circle(frame, (ax, ay), 9, (0, 230, 80), -1)

            # ---------- 食指 Tap-Click（仅右手） ----------
            index_bend: float | None = None
            right_screen_pt = None
            if "Left" in hand_results:
                _, _, _, lms = hand_results["Left"]
                tip = np.array([lms[8].x, lms[8].y])
                m5  = np.array([lms[5].x, lms[5].y])
                m17 = np.array([lms[17].x, lms[17].y])
                palm_n = float(np.linalg.norm(m5 - m17))
                tip_m5 = float(np.linalg.norm(tip - m5))
                if palm_n > 1e-6:
                    index_bend = tip_m5 / palm_n
                anchor_norm = HandDetector.cursor_anchor_from_landmarks(lms, args.anchor)
                gx, gy = mouse._norm_to_screen(*anchor_norm)
                sx, sy = mouse._smooth(gx, gy)
                right_screen_pt = (sx, sy)

            if disable_click_for_stretch_global:
                tap_detector.reset()
                tap_event = 0
            else:
                tap_event = tap_detector.update(index_bend)
            if not disable_click_for_stretch_global and stretch.state != _DS.SCALING:
                if tap_event in (1, 2):
                    mouse._smoother.freeze_for_click()
                if tap_event != 0:
                    mouse.handle_finger_tap_event(tap_event, right_screen_pt)

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
            _draw_tap_bar(frame, tm, tap_detector)

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
                        cv2.rectangle(frame, (bx - 6, by - 24), (bx + prog_w + 6, by + 48), (0, 0, 0), -1)
                        cv2.rectangle(frame, (bx, by), (bx + prog_w, by + 12), (60, 60, 60), -1)
                        if stretch_info.state == _DS.IDLE and stretch_info.ui_pinch_active:
                            fill_ratio = min(1.0, theta_deg / 90.0)
                            cv2.rectangle(frame, (bx, by), (bx + int(prog_w * fill_ratio), by + 12), pre_c, -1)
                            put_text_cn(frame,
                                        f"预判定【{pre_mode}】  θ={theta_deg:.0f}° ｜ 双手≥120px 并保持3帧即可激活",
                                        (bx, by + 20), font_scale=0.58, color=(255, 255, 255), thickness=2)
                        else:
                            cv2.rectangle(frame, (bx, by), (bx + int(prog_w * pct), by + 12), color, -1)
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
                cv2.rectangle(frame, (tx - 12, ty - th - 14),
                              (tx + tw + 12, ty + 12),
                              (0, 0, 0), -1)
                put_text_cn(frame, combo_hint, (tx, ty),
                            font_scale=0.8, color=(0, 230, 230), thickness=2)

            # ---------- 顶部模式标签 ----------
            anchor_text = "掌心" if args.anchor == "palm" else "食指尖"
            mode_text   = "绝对指哪打哪" if args.cursor_mode == "absolute" else "AR 悬停平滑"
            roi_pct = int(round(args.roi_scale * 100))
            cv2.rectangle(frame, (0, 0), (w, 30), (0, 0, 0), -1)
            put_text_cn(frame, mode_label, (10, 6),
                        font_scale=0.6, color=mode_color, thickness=2)
            put_text_cn(frame, f"🖱️{mode_text} | 锚点: {anchor_text} | 映射ROI: {roi_pct}%  🟢=光标  🟠=食指TIP",
                        (w - 640, 6), font_scale=0.6, color=(0, 230, 80), thickness=2)
            # ---------- ROI 矩形 ----------
            margin_px = int(round((1.0 - args.roi_scale) * 0.5 * min(w, h)))
            x1 = margin_px
            y1 = margin_px
            x2 = w - margin_px
            y2 = h - margin_px
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
            # 🔥 附加跳帧率提示（让用户知道优化在生效）
            if _skip_total_frames >= 10:
                _skip_pct = 100.0 * _skip_skipped_frames / max(_skip_total_frames, 1)
                if _skip_pct >= 15.0:
                    fps_txt += f"  跳帧{_skip_pct:.0f}%⚡"
            cv2.rectangle(frame, (w - 448, 30), (w - 4, 60), (0, 0, 0), -1)
            fps_color = (0, 230, 80) if _fps_display >= 25 else (80, 180, 255) if _fps_display >= 18 else (80, 80, 255)
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
                # 无预览模式也需要让 OpenCV 内部事件泵偶尔动动（其实纯 cv2 可以不用，但保险起见加个极短 wait）
                try:
                    cv2.waitKey(1)
                except Exception:
                    pass

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
