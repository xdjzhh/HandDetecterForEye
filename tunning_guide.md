# HandDetecter 稳定方案 & 调参指南 (tunning_guide)

> 适用项目：`d:\TraeProj\HandDetecter`（MediaPipe 双手手势 → 鼠标 / 窗口控制）
> 核心目标：**大范围跟手 + 小范围不晃 + 点击不漂移 + 拉伸/移动不中断**

---

## 1. 系统稳定架构总览（3 大模块 × N 层流水线）

```
MediaPipe Hands 21 关键点
      │
      ├─ 手势识别 ──► gesture_recognizer.recognize_gesture()
      │               (OPEN/PINCH/PEACE/POINT/FIST… 阈值在 §6)
      │
      ├─ 点击检测 ──► finger_tap_click_detector (Tap-Click 状态机, §4)
      │               (食指屈伸=点击, 不依赖手掌运动)
      │
      └─ 鼠标映射 ──► mouse_controller.MouseController
                        │
                        ├─ ROI 映射（§8）：掌心 0~1 → 屏幕绝对像素（=输入 raw_x/raw_y）
                        │
                        ├─ CURSOR_MODE_SMOOTH   → AR 悬停 5 层流水线（§2）
                        └─ CURSOR_MODE_ABSOLUTE → 方案 A 3+1 层流水线（§3）
```

---

## 2. SMOOTH 模式：AR 眼镜 5 层悬停流水线
适用：追求**极致稳定**（AR 眼镜光标悬停体验），但有轻微相位滞后。
入口：`ARSmoother.filter(mode=smooth)` [mouse_controller.py L269-L411](file:///d:/TraeProj/HandDetecter/mouse_controller.py#L269-L411)

| Layer | 机制 | 核心参数 | 作用 |
|---|---|---|---|
| 第 0 层（前置）| 点击前位置冻结 | `CLICK_FREEZE_FRAMES=3` | 点击手势（TAP/PINCH/PEACE）触发瞬间 **锁光标 3 帧**，手指运动不拉动光标 |
| 1 | 滑动中值滤波 | `MEDIAN_WINDOW=5` | 杀 1~2 帧 MediaPipe ±30px 尖峰飞点（中值对极端离群点 100% 过滤）|
| 2 | 静止方差锁 | `LOCK_WINDOW=8`<br>`LOCK_THRESH_HOVER=0.80`<br>`LOCK_THRESH_BROWSE=3.0`<br>`LOCK_THRESH_DRAG=99999` | 近 8 帧位移 σ 低于阈值 → 光标硬锁死（悬停最稳 0.8px²；拖拽不锁=跟手）|
| 3 | 1€ Filter（速度自适应平滑） | `MODE_PARAMS[*] = (fc_min_Hz, beta, deadzone, response_k)`<br>POINT:`(0.30, 0.0040, 2.8, 3.0)`<br>OPEN :`(0.80, 0.0100, 1.2, 1.6)`<br>FIST :`(1.50, 0.0250, 0.0, 1.2)` | 慢动超平滑（fc_min 小）+ 快动超跟手（beta 大）；deadzone 单独控制小位移死区 |
| 4 | Sigmoid 非线性响应 | `response_k`（上面 1€ 参数第 4 位） | 大动灵敏（Δ→∞ 增益=1.0）+ 小动不动（Δ 小 增益≈0.22，5 倍阻尼） |
| 5 | 手势模式切换 | 按手势名切换上面 `MODE_PARAMS` 整套参数 | POINT 悬停=最稳 / OPEN 浏览=平衡 / FIST 拖拽=最跟手零死区 / TAP/PINCH=中等 |

**调用方式**：
```bash
python main.py --cursor-mode smooth    # 启用 AR 5 层
```

---

## 3. ABSOLUTE 模式：方案 A（3+1 层）= 大范围跟手 + 小范围不晃 ✅ 推荐
适用：**像触摸屏指哪打哪**（大范围 0 滞后），同时对准小按钮不晃（小范围压制噪声）。
入口：`ARSmoother.filter(mode=absolute)` [mouse_controller.py L270-L360](file:///d:/TraeProj/HandDetecter/mouse_controller.py#L270-L360)

### 3.1 Layer 定义（串行流水线，输入 raw_x/raw_y = 屏幕绝对像素）
| Layer | 机制 | 核心参数 | 解决什么问题 | 典型位移表现 |
|---|---|---|---|---|
| 第 0 层（前置）| 点击前位置冻结 | `CLICK_FREEZE_FRAMES=3` | 同 SMOOTH：点击瞬间锁死光标 |
| 1 | 轻量 3 帧中值 | `ABS_MEDIAN_WINDOW=3` | 杀 1 帧尖峰跳点（手瞬间识别错位跳到角落） | 1 帧 30px 飞点 → 输出几乎不变 |
| **2** | **Sigmoid 微移增益压缩** ✨ | `ABS_SIGMOID_K=0.35`<br>`ABS_SIGMOID_MIN_G=0.20`<br>**Sigmoid 拐点=11.0px**（用户要求「压缩到 5px」） | 1~5px 的 MediaPipe 纯噪声 → 光标晃动 | **Δ≤5px 全压**：增益∈[0.22, 0.29]（阻尼≥3.5 倍，1px 抖=0.22px 输出）<br>Δ=11px：增益≈0.60（拐点中点）<br>**Δ>25px：增益>0.99（完全不衰减=跟手）** |
| **3** | **静止方差锁（钉住光标）** | `ABS_LOCK_WINDOW=10`<br>`ABS_LOCK_THRESH=2.5px`<br>解锁阈值=`2.5×1.6 = 4.0px` | 对准 16×16 px 小按钮时「最后 1px 对不准」 | 连续 10 帧（≈333ms）σ<2.5px → **硬锁死在均值位置（0px 漂移）**<br>手一动 >4px → 1 帧立刻解锁（响应真实动作） |

### 3.2 Layer 2 Sigmoid 增益数值对照表（拐点=11，K=0.35，MIN_G=0.20）
| 位移 Δ (px) | 增益 g(Δ) | 等效阻尼 | 状态 |
|---|---|---|---|
| 0.5 / 1.0 | 0.220 / 0.223 | 4.6× / 4.5× | ✅ 5px 以内全压 |
| 3.0 / 4.0 | 0.246 / 0.264 | 4.1× / 3.8× | ✅ 5px 以内全压 |
| 5.0 | 0.287 | 3.5× | ✅ 5px 以内全压（刚好卡边界） |
| 11.0 | 0.600 | 1.7× | 📍 拐点（S 形中点） |
| 20.0 / 25.0 | 0.967 / 0.994 | 接近 0 衰减 | ✅ 大范围跟手 |
| 30.0 | 0.999 | 0 衰减 | ✅ 大范围跟手 |

**拐点计算公式**（想改"压缩到 N px"直接套）：
```
K = ABS_SIGMOID_K （当前=0.35）
贴底点 ≈ 拐点 - (2 / K)
⇒ 拐点 = 目标压缩边界(5px) + (2 / 0.35) ≈ 5 + 5.71 = 11.0 px ✅
```

### 3.3 调用方式
```bash
python main.py --cursor-mode absolute    # 启用方案 A 3+1 层（推荐）
```

---

## 4. 点击稳定：Tap-Click 状态机（食指屈伸=点击，手掌不动）
入口：`FingerTapClickDetector.update(bend)` [finger_tap_click_detector.py L42-L90](file:///d:/TraeProj/HandDetecter/finger_tap_click_detector.py#L42-L90)

### 4.1 核心阈值（已整体下移 0.10，避免手指自然弯曲误触）
| 参数（默认值） | 含义 | 数值范围（按 hardness ∈ [0,1]） |
|---|---|---|
| `ENTER_BEND_RATIO` | 进入弯曲判定阈值（bend < baseline × ENTER） | `0.62 ~ 0.70`（旧 0.72~0.80 整体下移 0.10） |
| `EXIT_BEND_RATIO` | 回弹判定阈值（形成滞回，ENTER+0.06） | `0.68 ~ 0.76`（整体同步下移 0.10） |
| `MIN_BEND_DROP` | 相对下降幅度门槛（必须从历史高值掉这么多才认弯）| `0.10 ~ 0.26`（防自然弯曲误触） |
| `LONG_PRESS_FRAMES` | 长按判定（弯曲持续多少帧=按住左键拖拽）| 默认=22 帧≈730ms |
| `TAP_MAX_FRAMES` | **快速短按优先级**：从弯曲到回弹≤此值→无论什么都按 TAP（禁止变成长按）| 默认=18 |
| `BEND_STABLE_THRESH` | 弯曲稳定判据（弯曲必须稳定 N 帧才视为真的弯了，防单帧毛刺）| 默认=0.010（配合 MIN_HOLD=2）|
| `COOLDOWN_FRAMES` | 点击后冷却（防止第二下连点卡死）| 用户要求保留=4 帧 |
| `BEND_SMOOTH_ALPHA` | 弯曲值 EMA 平滑（越大=跟手越快=越容易抖）| 默认=0.75（被 speed 参数覆盖：0.55 ~ 0.95） |

### 4.2 用户友好参数（CLI 直接调，不用算上面那些）
```bash
python main.py --tap-hardness 0.6   # 0~1：越大=要弯得越狠才触发（防误触）
               --tap-speed    0.6   # 0~1：越大=响应越快（灵敏度，触发快+短按时间窗宽）
```

### 4.3 状态机核心规则（防单点变长按）
```
STRAIGHT_IDLE → BEND_ARMED(1帧确认毛刺) → BEND_HOLD
   ├─ 弯曲≤TAP_MAX 且 回弹 → TAP_CLICK(单击) ✅ 快速短按优先级最高
   ├─ 弯曲>LONG_PRESS 且 稳定 → PRESSED_HOLD(长按/拖拽开始)
   └─ 回弹释放 → COOLDOWN(4帧) → IDLE
```

---

## 5. 拉伸/移动窗口稳定（双手手势 + 顶层窗口判定）
入口：`DiagonalStretchManager.update(left/right_px, frame_size)` [diagonal_stretch.py L195-L1020](file:///d:/TraeProj/HandDetecter/diagonal_stretch.py#L195-L1020)

### 5.1 手势互斥与判定（双手场景）
| 双手状态 | 对应模式 | 判定规则 |
|---|---|---|
| 两只手 **五指都张开（OPEN）** | **MOVE** = 移动窗口（平移，不改 size） | OPEN 严格判定 3 条同时满足：<br>① 四段共线角度：四指<30° / 拇指<35°<br>② 指尖散开度 ≥ 0.35 掌宽<br>③ 指尖距离 比 掌指关节距离 ≥ 1.15（指尖确实在外/伸直了）|
| 两只手 **非 OPEN（=任何其他姿态：握拳/PEACE/捏合/单指伸…）** | **SCALE** = 拉伸缩放（改 size，不改位置）| ✨ 规则：「只要不是 OPEN 就视为捏合」，避免捏合阈值太严识别不出 |
| 一只手 OPEN（单手）| **MOVE** = 移动窗口（平移）| 同上面 OPEN 严格判定 |

**UI 预判定反馈解耦**：`ui_pinch_active` / `ui_move_active` 只要满足条件 **立刻 True 画连线**，和状态机确认帧逻辑解耦，视觉先反馈不滞后。

### 5.2 SCALE 拉伸模式判定（按双手连线夹角 θ 区间）
| 双手连线与水平线夹角 \|θ\|（取绝对值 0~90°）| 拉伸模式 | 对应行为 |
|---|---|---|
| [0°, 20°] | HORIZONTAL（水平） | 只拉宽 / 缩窄窗口宽度 |
| (20°, 70°) | DIAGONAL（对角） | 等比放缩 / 对角拉伸（宽高一起变）|
| [70°, 90°] | VERTICAL（垂直） | 只拉高 / 拉低窗口高度 |

阈值：`min_hold_frames=3`（双手出现就默认 3 帧确认），`min_diag_px=120`（两手最近距离门槛，比 220 松很多）。

### 5.3 防「指令中断/重新识别」：双层防抖
| 层级 | 机制 | 阈值 | 解决什么 |
|---|---|---|---|
| 源头判定稳定 | OPEN/PINCH 判定连续 2 帧一致才切换 | `OPEN_STABLE_N=2` | 单帧 OPEN/PINCH 抖一下 → 输出完全不变（避免混杂态）|
| ACTIVE 容忍帧 | 进入 SCALING / MOVE_ACTIVE 后，不满足条件**先等 N 帧再退**，期间所有快照（hwnd/w0h0/scale_ratio/参考坐标）**100% 保留**，UI 继续画 | `SCALE_LOST_MAX=4`<br>`MOVE_LOST_MAX=4`（≈30fps 下 133ms） | 覆盖 MediaPipe 偶发漏检 / 距离瞬时抖一下 / OPEN 单帧抖动 → 之前 1 帧不满足就退 IDLE=「中断」，现在 4 帧内都视为「暂时遮挡，手会回来」→ 完全连续，不会重新识别 |

### 5.4 顶层主窗口判定（解决 WPS/Word/Chrome 只移动标签页子控件的坑）
命中点 HWND 必须经过 **4 层强校验**（`_promote_hit_hwnd_to_top_level` [diagonal_stretch.py L409-L436](file:///d:/TraeProj/HandDetecter/diagonal_stretch.py#L409-L436)）：
```
WindowFromPoint(掌心命中点)
  → ① GA_ROOTOWNER（主路，追溯到顶层所有者窗口=应用主窗口，跳过所有子控件）
  → ② GA_ROOT（兜底，爬到最顶层父窗口）
  → ③ GetWindowLongPtrW(GWL_STYLE) 强校验：
        ✗ 带 WS_CHILD=0x40000000 → 子控件/标签页=直接丢弃（100% 不允许移动子控件）
        ✓ 必须带 WS_CAPTION（有标题栏=正常可拖动窗口）或 WS_POPUP
  → ④ 尺寸校验：宽<80 / 高<60 = 极小幽灵窗口丢弃
```
前台窗口兜底也强制走一遍同样校验（避免前台 HWND 本身是子控件的极端情况）。

**结果**：无论掌心落在 Chrome 标签栏 / WPS 功能区 / Word 目录面板（都是子控件），最终选到的 HWND **都是带标题栏的应用主窗口** → 移动时整个应用一起动，绝不会只移一个标签页。

---

## 6. 参数总表（集中位置，一眼找到改哪里）
| 类别 | 参数 | 默认 | 文件:位置 |
|---|---|---|---|
| **ROI 映射** | `DEFAULT_ROI_SCALE` | **0.70**（中心 70% 映射全屏，每边留白 15%）| [mouse_controller.py L34](file:///d:/TraeProj/HandDetecter/mouse_controller.py#L34) / [main.py L222](file:///d:/TraeProj/HandDetecter/main.py#L222) CLI 默认 |
| | `ROI_SCALE_MIN / MAX` | 0.30 / 1.00 | [mouse_controller.py L35-L36](file:///d:/TraeProj/HandDetecter/mouse_controller.py#L35-L36) |
| **SMOOTH 模式** | `MEDIAN_WINDOW` | 5 | [mouse_controller.py L40](file:///d:/TraeProj/HandDetecter/mouse_controller.py#L40) |
| | `LOCK_WINDOW` | 8 | [mouse_controller.py L42](file:///d:/TraeProj/HandDetecter/mouse_controller.py#L42) |
| | `LOCK_THRESH_HOVER / BROWSE / DRAG` | 0.80 / 3.0 / 99999 | [mouse_controller.py L44-L46](file:///d:/TraeProj/HandDetecter/mouse_controller.py#L44-L46) |
| | `MODE_PARAMS`（1€ 4 元组）| POINT/OPEN/FIST/PEACE/PINCH/OTHER | [mouse_controller.py L50-L60](file:///d:/TraeProj/HandDetecter/mouse_controller.py#L50-L60) |
| **ABSOLUTE 方案 A** | `ABS_MEDIAN_WINDOW`（Layer 1）| 3 | [mouse_controller.py L71](file:///d:/TraeProj/HandDetecter/mouse_controller.py#L71) |
| | `ABS_SIGMOID_K`（Layer 2 曲线陡度）| 0.35 | [mouse_controller.py L77](file:///d:/TraeProj/HandDetecter/mouse_controller.py#L77) |
| | `ABS_SIGMOID_MIN_G`（Layer 2 最小增益）| 0.20 | [mouse_controller.py L78](file:///d:/TraeProj/HandDetecter/mouse_controller.py#L78) |
| | Sigmoid 拐点（代码内 INFLECTION_PX，Layer2） | **11.0**（压缩到 5px）| [mouse_controller.py L269](file:///d:/TraeProj/HandDetecter/mouse_controller.py#L269) |
| | `ABS_LOCK_WINDOW`（Layer 3 锁死窗口）| 10 | [mouse_controller.py L82](file:///d:/TraeProj/HandDetecter/mouse_controller.py#L82) |
| | `ABS_LOCK_THRESH`（Layer 3 静止阈值 px）| 2.5 | [mouse_controller.py L83](file:///d:/TraeProj/HandDetecter/mouse_controller.py#L83) |
| **共享** | `CLICK_FREEZE_FRAMES` | 3 | [mouse_controller.py L64](file:///d:/TraeProj/HandDetecter/mouse_controller.py#L64) |
| **点击** | `ENTER_BEND_RATIO` 档位起点（整体下移 0.10）| 0.62 | [finger_tap_click_detector.py L87](file:///d:/TraeProj/HandDetecter/finger_tap_click_detector.py#L87) |
| | `LONG_PRESS_FRAMES_DEFAULT` | 22 | [finger_tap_click_detector.py L49](file:///d:/TraeProj/HandDetecter/finger_tap_click_detector.py#L49) |
| | `TAP_MAX_FRAMES_DEFAULT`（快速短按优先级） | 18 | [finger_tap_click_detector.py L51](file:///d:/TraeProj/HandDetecter/finger_tap_click_detector.py#L51) |
| | `COOLDOWN_FRAMES` | 4 | [finger_tap_click_detector.py L58](file:///d:/TraeProj/HandDetecter/finger_tap_click_detector.py#L58) |
| | `BEND_STABLE_THRESH_DEFAULT` | 0.010 | [finger_tap_click_detector.py L53](file:///d:/TraeProj/HandDetecter/finger_tap_click_detector.py#L53) |
| **拉伸/移动** | `min_hold_frames`（双手出现默认 3 帧确认）| 8→3（双手出现时切到 3） | [diagonal_stretch.py L202-L222](file:///d:/TraeProj/HandDetecter/diagonal_stretch.py#L202-L222) |
| | `min_diag_px`（双手最小距离门槛）| 120（原 220 放宽）| [diagonal_stretch.py L204](file:///d:/TraeProj/HandDetecter/diagonal_stretch.py#L204) |
| | `SCALE_LOST_MAX / MOVE_LOST_MAX`（ACTIVE 容忍帧） | 4 / 4 | [diagonal_stretch.py L198-L199](file:///d:/TraeProj/HandDetecter/diagonal_stretch.py#L198-L199) |
| | `OPEN_STABLE_N`（OPEN 滞回帧）| 2 | [diagonal_stretch.py L200](file:///d:/TraeProj/HandDetecter/diagonal_stretch.py#L200) |
| | OPEN 角度阈值（四指 / 拇指）| 30° / 35° | [diagonal_stretch.py L369/L372](file:///d:/TraeProj/HandDetecter/diagonal_stretch.py#L369) |
| | OPEN 散开度下限 | ≥0.35 掌宽 | [diagonal_stretch.py L375+](file:///d:/TraeProj/HandDetecter/diagonal_stretch.py#L375) |
| | 拉伸模式角度区间 θ（水平 / 对角 / 垂直） | 20° / 70°（±ABS）| diagonal_stretch.py SCALING 判定 |
| **手势识别** | `PINCH` 距离阈值（`pinch_r < 0.25 或 <0.40 且非(食+中)伸直`） | 0.25 / 0.40 | [gesture_recognizer.py L74](file:///d:/TraeProj/HandDetecter/gesture_recognizer.py#L74) |
| | PEACE 二次校验阈值（食/中伸直比 ≥0.70，无/小弯曲 ≤0.55）| 0.70 / 0.55 | [gesture_recognizer.py 代码中段](file:///d:/TraeProj/HandDetecter/gesture_recognizer.py) |

---

## 7. 推荐调参路径（按常见诉求顺序）
### 路径 A：想要「更跟手」（大范围拖动更爽）
1. CLI 切 `--cursor-mode absolute`（方案 A，比 smooth 跟手很多）
2. 如果还嫌不够 → 把 `ABS_SIGMOID_K` 从 0.35 → **0.30**（拐点变松：12px 之后更顺滑）
3. 还不行 → 把 `ABS_SIGMOID_MIN_G` 从 0.20 → **0.30**（整体阻尼减少）

### 路径 B：想要「更稳不晃」（对准 16px 小按钮不抖）
1. CLI 切 `--cursor-mode absolute`（方案 A 的 Layer 3 静止锁就够了）
2. 如果还抖 → `ABS_LOCK_THRESH` 2.5 → **2.0**（对静止要求更严）
3. 再抖 → `ABS_LOCK_WINDOW` 10 → **14**（需要静止更久才锁，但锁得更准）
4. 极端追求悬停稳 → 切 `--cursor-mode smooth` 并多用 POINT 手势（POINT 有最强平滑+2.8px 死区+0.8px² 锁）

### 路径 C：想要「点击更灵敏 / 更少误触」
- 更灵敏：`--tap-speed 0.8`（响应更快，短按时间窗更宽）
- 更少误触：`--tap-hardness 0.85`（需要手指弯得更狠才触发，避免自然弯曲误触）
- 经常「单点变成长按」→ `TAP_MAX_FRAMES_DEFAULT` 从 18 → **24**（快速短按优先级更宽容），或 `--tap-speed 0.9`（speed=1 → TAP_MAX=28）

### 路径 D：想要「拉伸/移动指令不中断」
- 默认已经是 `SCALE_LOST_MAX=4`，够了
- 还会断（比如手移出镜头 1 秒就放回来）→ `SCALE_LOST_MAX / MOVE_LOST_MAX` 从 4 → **8**（容忍更长漏检）
- OPEN 判定太严（五指张开老是识别不出）→ 共线角度放宽 30/35 → **40/45**，散开度 0.35 → **0.30**（但要注意：放宽后「手指并拢状态」可能会误判成 OPEN → 所以只放一点点就够）

### 路径 E：想要「Sigmoid 压缩边界改到 N px」（比如压缩到 8px）
直接改 `_abs_sigmoid_gain` 里的 `INFLECTION_PX`：
```
INFLECTION_PX = 目标边界 + (2 / ABS_SIGMOID_K)
例：想压到 8px  →  8 + (2/0.35) ≈ 8 + 5.71 = 13.71 → 取 14.0
```

---

## 8. ROI 映射原理（为什么选 0.70 = 中心 70% 映射全屏）
```
 摄像头画面 640×480 (100% 区域)
┌──────────────────────────────────────┐
│   ▲ 7.5% 留白 ▲                      │
│ ┌──────────────────────────────┐ ◀───┐
│ │  ROI = 70% × 70% 中心区域    │     │ 留白总宽 = 30% / 2 = 15%
│ │  → 1:1 映射 1920×1080 屏幕   │     │
│ │  (掌心到这里=屏幕边缘)       │     │
│ └──────────────────────────────┘ ◀───┘
│   ▼ 7.5% 留白 ▼                      │
└──────────────────────────────────────┘
  ✅ 当掌心映射到屏幕边缘时，整只手仍然在画面 85%~100% 区域，
     指尖/腕部完全不碰镜头边缘 → MediaPipe 100% 能识别完整 21 个关键点 → 不会「无法识别到掌心」
  ✅ 之前 0.85（留白 7.5%）：掌心到屏幕边缘时，指尖已经跑出画面 → 识别失败；
     现 0.70（留白 15%）：彻底解决。
```
CLI 调：
```bash
python main.py --roi-scale 0.70   # 默认（最平衡）
python main.py --roi-scale 0.60   # 再往内收（留白 20%），手活动空间更大，但掌心小范围映射大光标
python main.py --roi-scale 0.85   # 往外放（留白 7.5%，最跟手但易碰边）
```

---

## 9. 常见现象 → 只改 1 个参数（快速索引）
| 用户现象 / 抱怨 | 原因推测 | 改哪个参数 | 调大/调小 | 预期效果 |
|---|---|---|---|---|
| 大范围移动光标**拖尾滞后**，不像指哪打哪 | SMOOTH 模式 1€ 的 fc_min 太小导致慢 | 切 `--cursor-mode absolute`（方案 A）| 切换 | 0 滞后指哪打哪 |
| 小范围**抖得点不准小按钮**（16×16 关闭键） | Layer 2/3 不够严，噪声漏过 | `ABS_LOCK_THRESH` | 2.5→2.0（调小）| 更早进入锁死，钉死光标 |
| 手一停**半天锁不住**（一直晃）| 方差锁窗口太短 / 阈值太严 | `ABS_LOCK_WINDOW` | 10→14（调大）| 需要静止更久才锁，但锁上就特别稳 |
| 手一晃**立刻误锁**（锁得太早/太灵敏） | Layer 3 锁阈值太松 | `ABS_LOCK_THRESH` | 2.5→3.5（调大）| 更难进入锁死（允许 3.5px 微调不锁）|
| Sigmoid 压缩范围太小（3px 就不压了）→ 用户要求「压缩到 5px」| 拐点太靠左（之前 5→7→最终定 11）| `INFLECTION_PX` in `_abs_sigmoid_gain` | 7→11（调大）| Δ≤5px 增益∈[0.22,0.29] 全部压下去 |
| 大范围移动时觉得「有阻尼拖不动」 | Sigmoid 衰减区太大（拐点太靠右）| `INFLECTION_PX` | 11→9（调小）| 拐点左移 15px 处增益≈0.99 |
| **点击需要弯太狠**（要掰手指才触发）| hardness 太高 / ENTER 太严 | `--tap-hardness` | 0.6→0.3（调小）| ENTER=0.644（松）|
| **手指自然放松就误点**（还没弯就触发）| hardness 太低，ENTER 太松 | `--tap-hardness` | 0.6→0.9（调大）| ENTER=0.692（严；整体已下移 0.10）|
| 单击一下就被识别成**长按**（单点变长按）| TAP_MAX 太短（快速短按优先级不够宽容） | `--tap-speed` | 0.6→0.9（调大）| TAP_MAX=26~28（更宽容）|
| 连点第二下**卡住没反应** | COOLDOWN 太长 | 已改到 4 帧（若仍不行改 3）| 略小 | 冷却更短，第二下响应快 |
| 双手拉伸 / 移动**老是中断重来** | LOST_MAX 容忍帧太少（漏检就退 IDLE）| `SCALE_LOST_MAX / MOVE_LOST_MAX` | 4→8（调大）| 连续 8 帧不满足才真退 |
| 五指张开移动窗口**老是识别不出来** | OPEN 共线度 / 散开度太严 | `30°→40°, 35°→45°, 0.35→0.30` | 稍微放宽（谨慎！）| 更容易识别 OPEN，注意并拢可能误判 |
| 非五指张开**也被当成 OPEN**（捏合/拉伸不触发）| OPEN 判定太松，指尖并拢也判成张开 | 上面三个阈值往回缩：30→25 / 0.35→0.38 | 调严 | OPEN 判定更严格=非 OPEN=捏合更准 |
| WPS/Word/Chrome 移动时**只移动了一个标签页**（整个窗口不动）| WindowFromPoint 命中了子控件 HWND | `_promote_hit_hwnd_to_top_level`（默认已开，无需改）| — | GA_ROOTOWNER 追溯到顶层主窗口=整个应用一起动（§5.4）|
| 掌心到达屏幕边缘时**MediaPipe 直接识别不到手**（=丢手心） | ROI 太大，留白不够 → 手跑出画面 | `--roi-scale` 或 `DEFAULT_ROI_SCALE` | 0.85→0.70→0.60（调小=往内收）| 留白更多，掌心到边时整只手还在画面 |

---

## 10. 推荐默认启动命令（95% 场景最优体验）
```bash
# 方案 A 绝对坐标（大范围跟手 + 小范围不晃）+ ROI 70%（掌心不跑边）+ 中等点击硬度 + 略快响应
python main.py --use-tasks --cursor-mode absolute --roi-scale 0.70 --tap-hardness 0.6 --tap-speed 0.7
```
