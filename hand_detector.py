import os
import time
import threading
import cv2
import mediapipe as mp
import numpy as np

"""
MediaPipe 双手检测封装 —— 双模式兼容
=======================================
模式 A（默认，稳定）: HandDetector
    基于 mp.solutions.hands，模型已打包在 mediapipe wheel 中，
    无需外部模型文件，但依赖 mediapipe 内置缓存。

模式 B（纯本地模型）: TasksHandDetector
    基于 mp.tasks.vision.HandLandmarker，加载项目目录下的
    assets/hand_landmarker.task 单文件模型。
    完全不依赖网络与 mediapipe 内部缓存，可离线使用。
    输出数据结构与模式 A 完全一致，调用方（gesture_recognizer /
    mouse_controller / main.py）无需做任何适配。
"""

# ---------- 通用工具：MediaPipe 21 点手部连线拓扑 ----------
# mp.solutions.hands.HAND_CONNECTIONS 的等价常量（tasks 模式需自行绘制）
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),           # 食指
    (5, 9), (9, 10), (10, 11), (11, 12),      # 中指
    (9, 13), (13, 14), (14, 15), (15, 16),    # 无名指
    (13, 17), (17, 18), (18, 19), (19, 20),   # 小指
    (0, 17),                                   # 掌边
]

_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "assets", "hand_landmarker.task"
)


# ==========================================================================
# 模式 A：mp.solutions.hands（默认，稳定）
# ==========================================================================
class HandDetector:
    """MediaPipe Solutions API 封装（双手 + 左右手标签）。

    输出约定（两种模式共用）:
        hands_data[i] = {
            "label":    "Left" | "Right",   # MP 原始标签（假定输入为镜像自拍图）；
                                            # 同视角不镜像，由 hands_tasks.run_loop 检测出口
                                            # 统一交换，交换后 "Left"=用户右手=控鼠手
            "landmarks": list[_LmLike],     # 21 个点，每个支持 .x/.y/.z ∈ [0,1]（归一化图像坐标）
            "world_landmarks": list[_LmLike] | None,
            #                       21 个点，.x/.y/.z 单位【米】，原点在腕部（身体坐标系，与视角无关）
            "score":    float,              # 左右手分类置信度
        }
    """

    def __init__(
        self,
        max_hands: int = 2,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.6,
    ):
        self.mode = "SOLUTIONS"
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self._connect_style = self.mp_draw.DrawingSpec(
            color=(0, 255, 0), thickness=2, circle_radius=2
        )
        self._landmark_style = self.mp_draw.DrawingSpec(
            color=(255, 0, 255), thickness=2, circle_radius=3
        )

    def take_det_diag(self) -> dict:
        """solutions 同步模式无异步超时/排队概念，诊断桩返回零值（字段与 Tasks 版一致）。"""
        return {"infer": 0, "timeout": 0, "empty": 0, "hand1": 0,
                "lat_avg_ms": 0.0, "lat_p90_ms": 0.0}

    # ---------------- 主接口 ----------------
    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        hands_data = []
        if results.multi_hand_landmarks and results.multi_handedness:
            _world_all = getattr(results, "multi_hand_world_landmarks", None) or []
            for _i, (lms_wrapper, handedness) in enumerate(zip(
                results.multi_hand_landmarks, results.multi_handedness
            )):
                label = handedness.classification[0].label
                score = handedness.classification[0].score
                # 米制 3D 点（原点腕部，与视角无关）；proto Landmark 本身支持 .x/.y/.z，直接透传
                _wlms = _world_all[_i].landmark if _i < len(_world_all) and _world_all[_i] else None
                hands_data.append(
                    {
                        "label": label,
                        "landmarks": lms_wrapper.landmark,
                        "world_landmarks": _wlms,
                        "score": score,
                    }
                )
                self.mp_draw.draw_landmarks(
                    frame,
                    lms_wrapper,
                    self.mp_hands.HAND_CONNECTIONS,
                    self._landmark_style,
                    self._connect_style,
                )
        return hands_data, frame

    # ---------------- 静态工具 ----------------
    @staticmethod
    def landmark_to_pixel(landmark, frame_w: int, frame_h: int):
        return int(landmark.x * frame_w), int(landmark.y * frame_h)

    @staticmethod
    def cursor_anchor_from_landmarks(lms, mode: str = "palm"):
        """从 21 个 landmarks 计算光标锚点（归一化 0~1 坐标）。

        Args:
            lms: list[NormalizedLandmark|_LmWrapper]，21 个关键点
            mode: "mcp" = 四指根 MCP(5,9,13,17) 质心【同视角默认】——手腕出画面不漂、
                            勾按/握拳时掌骨关节位置不变（满足"点击只动手指、光标不漂"）；
                  "palm" = (WRIST_0 + MIDDLE_MCP_9)/2，掌心锚点（面对面视角旧默认，保留可回退）；
                  "index_tip" = INDEX_FINGER_TIP_8，原食指指尖锚点（兼容；勾按时会跟着动）
        Returns:
            (x_norm, y_norm) 在 0~1 内
        """
        if mode == "index_tip":
            return float(lms[8].x), float(lms[8].y)
        if mode == "mcp":
            # ⭐ 2026-10-01 同视角锚点：四个掌指关节质心。
            #   同视角手从画面下方伸入、手腕常被截断（#0 会被模型"猜"出来→漂移）；
            #   MCP 是掌骨/指骨铰接点，勾按与握拳时其位置基本不动，四点平均还抗单点遮挡。
            return (float(lms[5].x) + float(lms[9].x)
                    + float(lms[13].x) + float(lms[17].x)) * 0.25, \
                   (float(lms[5].y) + float(lms[9].y)
                    + float(lms[13].y) + float(lms[17].y)) * 0.25
        # palm：掌心（腕部 0 + 中指掌指关节 9 的中点，业界标准）
        return (float(lms[0].x) + float(lms[9].x)) * 0.5, \
               (float(lms[0].y) + float(lms[9].y)) * 0.5

    def close(self):
        if hasattr(self, "hands"):
            self.hands.close()


# ==========================================================================
# 模式 B：mp.tasks.vision.HandLandmarker（纯本地模型，离线可用）
# ==========================================================================

# 为兼容原代码的 ".x / .y / .z" 访问方式，对 tasks 返回的 np.ndarray 做轻量包装
class _LmWrapper:
    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class TasksHandDetector(HandDetector):
    """使用本地 hand_landmarker.task 模型的手部检测器（纯离线）。

    对外接口、返回结构与 HandDetector 完全一致，可无缝替换。
    """

    def __init__(
        self,
        model_asset_path: str = _DEFAULT_MODEL_PATH,
        max_hands: int = 2,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.6,
    ):
        # 不调用父类 __init__，避免创建 mp.solutions.Hands
        self.mode = "TASKS_LOCAL_MODEL"
        self._model_path = os.path.abspath(model_asset_path)
        if not os.path.isfile(self._model_path):
            raise FileNotFoundError(
                f"[TasksHandDetector] 本地模型文件不存在: {self._model_path}\n"
                f"请从官方下载 hand_landmarker.task 并放到 assets/ 目录，\n"
                f"或改用默认模式（去掉 --use-tasks 参数）。"
            )

        # 注意：Windows 下 model_asset_path=绝对路径 会被 mediapipe C++ 层错误地
        # 拼接到 site-packages 后，造成 "Unable to open file ... errno=22"。
        # 解决方案：使用 model_asset_buffer 直接传文件二进制，彻底绕开路径解析。
        with open(self._model_path, "rb") as _f:
            model_buffer = _f.read()

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        self._detector = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_buffer=model_buffer),
                running_mode=VisionRunningMode.LIVE_STREAM,
                num_hands=max_hands,
                min_hand_detection_confidence=detection_confidence,
                min_hand_presence_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence,
                result_callback=self._on_result,
            )
        )

        # LIVE_STREAM 回调结果暂存
        self._pending_result = None
        self._frame_timestamp_ms = 0
        # ⭐ 2026-10-02 检测层诊断（纯观测，零行为变更；take_det_diag 取窗口并清零）：
        #   infer   = 实际发起 detect_async 的次数（跳帧复用不计）
        #   timeout = _result_event.wait(0.080s) 超时（回调没回来 → 当帧空手）
        #   empty   = 回调正常回来但 hand_landmarks 为空（模型本帧判定无手）
        #   hand1+  = 回调回来且至少 1 只手
        #   lat_ms  = 每次成功等到回调的等待耗时（≈排队+推理），窗口内取 avg/p90
        self._det_diag = {"infer": 0, "timeout": 0, "empty": 0, "hand1": 0,
                          "lat_ms": [],
                          # 2026-10-02 逆光补偿诊断：enh=本窗增强帧数；dark_max=暗像素占比峰值
                          "enh": 0, "dark_max": 0.0}
        # ⭐ 2026-09-20：回调用 Event 唤醒，替代 time.sleep(0.001) 轮询。
        #   Windows 默认定时器粒度 15.6ms，sleep(1ms) 实测睡 15.2ms → 每帧白等 ~15ms；
        #   Event.wait 走内核可等待对象，回调 set 的瞬间唤醒（亚毫秒级），且等待期间不占 CPU。
        self._result_event = threading.Event()

        # 绘制样式（与模式 A 保持一致观感）
        self._pt_color = (255, 0, 255)
        self._line_color = (0, 255, 0)
        self._pt_radius = 3
        self._pt_thickness = 2
        self._line_thickness = 2

        # 推理内部缩放尺寸：MediaPipe 模型只需要 ≈224px 的输入，
        # 1280×720 全尺寸送入只会放大 cvtColor + mp.Image 构造开销，
        # 对准确率几乎没帮助（归一化坐标本身就和像素无关）。
        # 缩到短边 480px（保持宽高比）→ 推理速度 +50%~100%。
        self._infer_short = 480

        # ✨ 2026-10-02 逆光补偿（对面强光 → AE 锁定亮背景 → 手部呈近黑剪影，
        #   模型大量漏检，表现为 None/丢帧）：
        #   仅对【送入模型的推理帧】的 L 通道做 gamma 阴影拉升 + 轻度 CLAHE——
        #   暗部（手，L≈20~60）抬到 ~60~90，亮背景基本不动（实测 200→207）；
        #   预览画面、归一化坐标、鼠标/点击链路全部不受影响。
        #   实测 480p 单帧耗时 ~5ms（推理等待 35~48ms，占比可接受）。
        self._ENH_GAMMA = 0.70
        self._enh_lut = np.array(
            [255.0 * ((i / 255.0) ** self._ENH_GAMMA) for i in range(256)],
            dtype=np.uint8)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._enh_on = False  # 增强开关（带迟滞，避免阈值附近每帧开关抖动）
        self._ENH_DARK_LUMA = 55.0    # L<55 计为暗像素（手部剪影典型落在 20~50）
        self._ENH_DARK_ON = 0.22      # 暗像素占比 ≥22% → 开
        self._ENH_DARK_OFF = 0.13     # 迟滞：回落 ≤13% 才关
        self._ENH_DEAD_MEAN = 10.0    # 全帧均值<10=相机纯黑帧，无细节可恢复，不增强（只放大噪声）

        # 预热 OpenCV LAB/LUT/CLAHE 内核：首次调用有 ~140ms 一次性初始化开销，
        # 在启动期吃掉，避免运行中第一次遇到逆光时单帧卡顿
        _warm_l = cv2.split(cv2.cvtColor(np.full((32, 32, 3), 40, np.uint8),
                                         cv2.COLOR_BGR2LAB))[0]
        self._clahe.apply(cv2.LUT(_warm_l, self._enh_lut))

    def _enhance_inference_bgr(self, bgr):
        """逆光自适应补偿（只处理模型输入帧）。

        返回 (out_bgr, enhanced: bool, dark_frac: float)。
        判定基于灰度暗像素占比：逆光场景 = 大面积暗区（手）+ 小面积强光背景共存。
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        dark_frac = float(np.count_nonzero(gray < self._ENH_DARK_LUMA)) / float(gray.size)
        if dark_frac > self._det_diag["dark_max"]:
            self._det_diag["dark_max"] = dark_frac
        # 迟滞开关
        if dark_frac >= self._ENH_DARK_ON:
            self._enh_on = True
        elif dark_frac <= self._ENH_DARK_OFF:
            self._enh_on = False
        if not self._enh_on or float(gray.mean()) < self._ENH_DEAD_MEAN:
            return bgr, False, dark_frac
        # L 通道：gamma 全局拉暗影调（平色块也能抬起）+ CLAHE 局部对比（恢复手部纹理）
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_ch = self._clahe.apply(cv2.LUT(l_ch, self._enh_lut))
        out = cv2.cvtColor(cv2.merge((l_ch, a_ch, b_ch)), cv2.COLOR_LAB2BGR)
        self._det_diag["enh"] += 1
        return out, True, dark_frac

    # ---------------- Tasks 回调 ----------------
    def _on_result(self, result, output_image, timestamp_ms):
        """LIVE_STREAM 模式下通过回调获取结果。"""
        self._pending_result = result
        self._result_event.set()

    # ---------------- 主接口 ----------------
    def detect(self, frame):
        h, w = frame.shape[:2]

        # ⚡ 推理尺寸降级：MediaPipe 内部模型只需要 ~224px
        #   1280×720 → 缩到短边 480px：推理像素量进一步减少
        #   landmarks 仍是 [0,1] 归一化坐标，对业务完全透明，准确率无损。
        s = self._infer_short
        if min(h, w) > s + 4:
            scale = s / min(h, w)
            nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
            infer_bgr = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        else:
            infer_bgr = frame
        # ✨ 逆光补偿：仅作用于模型输入（frame 本身/预览/坐标不变）
        infer_bgr, _enh, _dark_frac = self._enhance_inference_bgr(infer_bgr)
        rgb = cv2.cvtColor(infer_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Tasks API 要求时间戳严格单调递增（毫秒）
        self._frame_timestamp_ms += 33
        self._pending_result = None
        self._result_event.clear()
        self._det_diag["infer"] += 1
        self._detector.detect_async(mp_img, self._frame_timestamp_ms)

        # ⭐ 2026-09-20：Event 内核等待替代 sleep(1ms) 轮询。
        #   Windows 默认定时器粒度 15.6ms，旧写法 time.sleep(0.001) 实测每次睡 15.2ms，
        #   推理 40ms 的帧平白多等 1~2 个定时片（detect 实测 55ms→修复后~40ms）。
        #   Event.wait 在回调 set 的瞬间由操作系统直接唤醒，精度亚毫秒；80ms 死线保留
        #   （极端卡顿下回调缺失时返回 None，与旧逻辑行为一致，不会卡死主循环）。
        _tw0 = time.perf_counter()
        _got = self._result_event.wait(timeout=0.080)
        if _got:
            self._det_diag["lat_ms"].append((time.perf_counter() - _tw0) * 1000.0)
        else:
            self._det_diag["timeout"] += 1
        result = self._pending_result

        hands_data = []
        if result is not None and result.hand_landmarks:
            self._det_diag["hand1"] += 1
            for i, lms_norm in enumerate(result.hand_landmarks):
                # handedness: tasks 返回分类列表，取 Top-1
                hlabel = "Right"
                hscore = 0.5
                if result.handedness and i < len(result.handedness):
                    cats = result.handedness[i]
                    if cats:
                        # tasks 的 category.display_name / category_name 二选一
                        top = max(cats, key=lambda c: c.score)
                        hlabel = (getattr(top, "display_name", None)
                                  or getattr(top, "category_name", None)
                                  or "Right")
                        hscore = float(top.score)

                # 包装成与 solutions 一致的 21 个 wrapper 列表
                wrapped = [_LmWrapper(p.x, p.y, getattr(p, "z", 0.0)) for p in lms_norm]
                # 米制 3D 点（hand_world_landmarks：单位米，原点腕部，身体坐标系与视角无关）
                _wrapped_world = None
                _wlms_raw = getattr(result, "hand_world_landmarks", None)
                if _wlms_raw and i < len(_wlms_raw) and _wlms_raw[i]:
                    _wrapped_world = [_LmWrapper(float(p.x), float(p.y), float(p.z))
                                     for p in _wlms_raw[i]]
                hands_data.append(
                    {
                        "label": hlabel,
                        "landmarks": wrapped,
                        "world_landmarks": _wrapped_world,
                        "score": hscore,
                    }
                )
                # 自行绘制关键点与连线（tasks 无 drawing_utils）
                self._draw_on_frame(frame, wrapped)
        elif result is not None:
            # 回调正常回来但本帧无手（区分于 80ms 超时）
            self._det_diag["empty"] += 1

        return hands_data, frame

    def take_det_diag(self) -> dict:
        """取出自上次取数以来的检测层诊断窗口并清零（纯观测，不影响检测行为）。
        返回 infer/timeout/empty/hand1 + 等待延迟 avg/p90(ms)。"""
        d = self._det_diag
        lat = d["lat_ms"]
        n = len(lat)
        avg = sum(lat) / n if n else 0.0
        p90 = sorted(lat)[min(n - 1, int(round(0.9 * (n - 1))))] if n else 0.0
        snap = {"infer": d["infer"], "timeout": d["timeout"],
                "empty": d["empty"], "hand1": d["hand1"],
                "lat_avg_ms": round(avg, 1), "lat_p90_ms": round(p90, 1),
                "enh": d["enh"], "dark_max": round(float(d["dark_max"]), 3)}
        d["infer"] = d["timeout"] = d["empty"] = d["hand1"] = 0
        d["lat_ms"] = []
        d["enh"] = 0
        d["dark_max"] = 0.0
        return snap

    # ---------------- 自定义绘制（与模式 A 观感一致）----------------
    def _draw_on_frame(self, frame, wrapped_landmarks):
        h, w = frame.shape[:2]
        pts = []
        for lm in wrapped_landmarks:
            px, py = int(lm.x * w), int(lm.y * h)
            pts.append((px, py))
            cv2.circle(frame, (px, py), self._pt_radius,
                       self._pt_color, self._pt_thickness)
        for a, b in HAND_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(frame, pts[a], pts[b],
                         self._line_color, self._line_thickness)

    def close(self):
        if hasattr(self, "_detector") and self._detector is not None:
            self._detector.close()
            self._detector = None
