import os
import time
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
            "label":    "Left" | "Right",   # MP 原始标签（调用端再反转为用户视角）
            "landmarks": list[_LmLike],     # 21 个点，每个支持 .x/.y/.z ∈ [0,1]
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

    # ---------------- 主接口 ----------------
    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        hands_data = []
        if results.multi_hand_landmarks and results.multi_handedness:
            for lms_wrapper, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                label = handedness.classification[0].label
                score = handedness.classification[0].score
                hands_data.append(
                    {
                        "label": label,
                        "landmarks": lms_wrapper.landmark,
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
            mode: "palm" = (WRIST_0 + MIDDLE_MCP_9)/2，掌心锚点，食指屈伸不飘；
                  "index_tip" = INDEX_FINGER_TIP_8，原食指指尖锚点（兼容）
        Returns:
            (x_norm, y_norm) 在 0~1 内
        """
        if mode == "index_tip":
            return float(lms[8].x), float(lms[8].y)
        # 默认 palm：掌心（腕部 0 + 中指掌指关节 9 的中点，业界标准）
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

    # ---------------- Tasks 回调 ----------------
    def _on_result(self, result, output_image, timestamp_ms):
        """LIVE_STREAM 模式下通过回调获取结果。"""
        self._pending_result = result

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
            rgb = cv2.cvtColor(cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR),
                               cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Tasks API 要求时间戳严格单调递增（毫秒）
        self._frame_timestamp_ms += 33
        self._pending_result = None
        self._detector.detect_async(mp_img, self._frame_timestamp_ms)

        # 🔥 等待回调结果：用 time.sleep(1ms) + 80ms 死线
        #   之前用 "for _ in range(5): cv2.waitKey(1)" 有严重问题：
        #     cv2.waitKey 在无窗口焦点 / 无GUI事件时可能立即返回（根本不等1ms），
        #     5 次循环实际总共等了 <0.5ms，而 1280×720 高分辨率帧的 MediaPipe
        #     回调通常需要 4~18ms → 结果几乎永远拿不到 → hands_data 永远空 →
        #     用户观感"只偶尔出现关节连线，手势完全无法识别"。
        #   修复：time.sleep(0.001) 严格等待，最多等 80ms（远大于最坏耗时，又不会卡住）。
        _deadline = time.perf_counter() + 0.080
        while time.perf_counter() < _deadline:
            if self._pending_result is not None:
                break
            time.sleep(0.001)
        result = self._pending_result

        hands_data = []
        if result is not None and result.hand_landmarks:
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
                hands_data.append(
                    {
                        "label": hlabel,
                        "landmarks": wrapped,
                        "score": hscore,
                    }
                )
                # 自行绘制关键点与连线（tasks 无 drawing_utils）
                self._draw_on_frame(frame, wrapped)

        return hands_data, frame

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
