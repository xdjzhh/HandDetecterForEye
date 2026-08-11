# HandDetecter 体感鼠标 调参指南（TUNING GUIDE）
> 适用：双手体感鼠标项目 `python main.py`（含 Tasks 本地模型 / 默认 Solutions 两种模式）
> 最后更新：2026-08-11

---

## 一、快速启动：4 种场景「开箱即用」命令
> 直接复制粘贴即可，参数含义见第二、三章

| # | 场景 | 命令 | 适合谁 |
|---|------|------|--------|
| A | **新人入门（稳优先）** | `py -3.10 main.py --use-tasks --anchor palm --cursor-mode smooth --roi-scale 0.85 --tap-hardness 0.70 --tap-speed 0.65` | 第一次用，容易误触，**ENTER 整体平移 -0.10 + H=0.7 → 需弯 32%**，自然放松的弯曲不会触发 |
| B | **极致跟手（指哪打哪）** | `py -3.10 main.py --use-tasks --anchor palm --cursor-mode absolute --roi-scale 0.80 --tap-hardness 0.75 --tap-speed 0.90` | 进阶用户：光标跟手、掌心绝对坐标，H=0.75 → 需弯 32%，响应快速 |
| C | **办公连点 + 拖拽（默认推荐）** ✨ | `py -3.10 main.py --use-tasks --anchor palm --cursor-mode smooth --roi-scale 0.85 --tap-hardness 0.70 --tap-speed 0.80` | 日常办公：H=0.7 弯 32% 过滤自然放松弯，响应 S=0.80 快 |
| D | **大屏 / 手小 / 离摄像头远** | `py -3.10 main.py --use-tasks --anchor palm --cursor-mode absolute --roi-scale 0.55 --tap-hardness 0.80 --tap-speed 0.70` | 摄像头距离远，H=0.8 → 需弯 31.6%≈32%，弯幅门槛一致 |

**注意**：所有启动命令均以 `py -3.10` 启动（本项目 MediaPipe 仅兼容 3.10），3.13 会报 `ModuleNotFoundError: mediapipe`。

---

## 二、所有 CLI 参数总表（按功能分类）

### 2.1 核心模式 / 硬件 / 模型
| 参数 | 默认 | 取值 | 作用 | 常见调节建议 | 源码位置 |
|------|------|------|------|-------------|----------|
| `--use-tasks` | 关（布尔开关） | `--use-tasks`=开启；不加=默认 Solutions | 使用 **本地模型**（`assets/hand_landmarker.task`）离线运行，不依赖网络；**强烈建议始终加上**。 | 第一次使用必须加；无此参数时启动的是 MediaPipe 内置模型，需要联网下模型。 | [main.py L202-L205](file:///d:/TraeProj/HandDetecter/main.py#L202-L205) |
| `--model` | `None`（=自动找 assets 下的 task 文件） | 任意文件路径 string | `--use-tasks` 专用：手动指定 `.task` 文件的绝对/相对路径（你把模型放在其他位置时用）。 | 一般不用改；若提示"找不到 hand_landmarker.task"就填完整路径。 | [main.py L206-L208](file:///d:/TraeProj/HandDetecter/main.py#L206-L208) + [L242-L250](file:///d:/TraeProj/HandDetecter/main.py#L242-L250) |
| `--max-hands` | `2` | 整数 `1 ~ 2` | MediaPipe 最多同时识别几只手。 | 单只手使用场景可以改 `1` 省一点算力；双手组合手势必须 `2`。 | [main.py L210](file:///d:/TraeProj/HandDetecter/main.py#L210) |

### 2.2 鼠标光标锚点 / 移动策略 / 屏幕映射（最重要 3 个参数）
| 参数 | 默认 | 取值 | 作用 | 常见调节建议 | 源码位置 |
|------|------|------|------|-------------|----------|
| `--anchor` | `palm` ✨推荐 | `palm` \| `index_tip` | **光标对应手的哪里**：`palm`=掌心（腕部 0 + 中指 MCP 9 中点，**食指点击时不漂**）；`index_tip`=食指尖（兼容旧逻辑，但点击时光标会随食指弯曲移动=误点）。 | **永远用 `palm`（默认）**，不要改回 `index_tip`。 | [main.py L213-L216](file:///d:/TraeProj/HandDetecter/main.py#L213-L216) + [hand_detector.py cursor_anchor_from_landmarks](file:///d:/TraeProj/HandDetecter/hand_detector.py#L107-L117) |
| `--cursor-mode` | `smooth` | `smooth`（AR 5 层悬停平滑）\| `absolute`（掌心→屏幕绝对坐标直接跳） | **光标怎么动**：`smooth` 是慢速跟手但极稳（不抖）；`absolute` 是指哪跳哪极跟手但 MediaPipe 尖峰会漂一点（已加 3 帧中值滤波杀尖峰）。 | 办公/连点用 `smooth`（默认）；跟手要求高/手移动慢/大屏用 `absolute`。 | [main.py L217-L220](file:///d:/TraeProj/HandDetecter/main.py#L217-L220) + [mouse_controller.py ARSmoother.filter](file:///d:/TraeProj/HandDetecter/mouse_controller.py#L233-L330) |
| `--roi-scale` | `0.85` | 浮点数 `0.30 ~ 1.00`（硬约束） | **摄像头画面映射到屏幕的比例**：`1.00`=全屏映射（手必须移到摄像头画面边缘才能到屏幕边缘）；`0.55`=画面中心 55% 区域即覆盖全屏（手不移远也能碰屏幕四边，四周留手指伸展空间）。 | 手在屏幕中间挪不动光标/碰不到边角就**往下调**（0.85→0.70→0.55）；画面边缘手抖就**往上调**到 0.90。 | [main.py L221-L224](file:///d:/TraeProj/HandDetecter/main.py#L221-L224) + [L235](file:///d:/TraeProj/HandDetecter/main.py#L235) 范围夹取 + [mouse_controller.py _norm_to_screen](file:///d:/TraeProj/HandDetecter/mouse_controller.py#L374-L382) |

### 2.3 食指点击（Tap/Long）两个新手友好参数 ✨（第三章讲内部映射）
| 参数 | 默认 | 取值 | 作用 | 常见调节建议 | 源码位置 |
|------|------|------|------|-------------|----------|
| `--tap-hardness` | `0.70` ✨=你要求的"小幅度不触发" | 浮点数 `0.00 ~ 1.00`（硬约束） | **点击需要弯多狠**：值越大→**必须弯得越深 + 相对降幅越大**才能触发点击。✨本轮明确调整：**ENTER_BEND_RATIO 整体平移 -0.10（所有档位同时降 0.10，H0~H1 间距 0.08 不变）**，默认 H=0.7 需弯 32% 才能触发，避免"手指自然放松的一点点弯曲就被误判为点击"。 | 自然弯曲仍误触就把 H 调到 `0.85` 甚至 `1.0`；刻意弯到底还没反应就降到 `0.50`。 | [main.py L226-L229](file:///d:/TraeProj/HandDetecter/main.py#L226-L229) / [L237-L238](file:///d:/TraeProj/HandDetecter/main.py#L237-L238)（范围夹取） |
| `--tap-speed` | `0.80` ✨=你要求的"更敏感更快" | 浮点数 `0.00 ~ 1.00`（硬约束） | **点击识别响应速度**：值越大→bend 平滑α 更高（更灵敏）、弯曲确认帧更少、短按优先级时间窗更宽容（更不容易误转长按）。 | 单点总被识别成长按就调 `1.0`；太灵敏抖动乱触发就降 `0.50`。 | [main.py L230-L233](file:///d:/TraeProj/HandDetecter/main.py#L230-L233) / [L239-L240](file:///d:/TraeProj/HandDetecter/main.py#L239-L240)（范围夹取） |

---

## 三、点击参数映射详解（H/S → 8 个内部阈值，避免手改 8 个文件级常量）
> 解析函数：[finger_tap_click_detector.py resolve_tap_params()](file:///d:/TraeProj/HandDetecter/finger_tap_click_detector.py#L65-L112)
> 注入构造：[FingerTapClickDetector.__init__  hardness/speed → self._p](file:///d:/TraeProj/HandDetecter/finger_tap_click_detector.py#L130-L161)
> 状态机引用：`self._p["ENTER_BEND_RATIO"]` 等（全状态机替换完成，不再依赖模块级常量）
> 构造注入 main.py：[FingerTapClickDetector(hardness=args.tap_hardness, speed=args.tap_speed)](file:///d:/TraeProj/HandDetecter/main.py#L270)

### 3.1 `--tap-hardness`（弯曲幅度门槛）→ 3 个"严格度"阈值
> ✨**用户本轮明确调整**：ENTER_BEND_RATIO **整体平移 -0.10**（=旧 0.72~0.80 整个区间线性下移 0.10，H0~H1 间距 0.08 保持不变）。目的：让"点击判定的初始弯曲度"整体增大 10%，**避免手指自然放松状态下有一点点弯曲就被误判为点击**。MIN_BEND_DROP / EXIT 滞回差 0.06 均保持旧设计不变。

| 内部阈值 | 含义 | H=0.00（最松=旧 0.72 减 0.10） | H=0.70（默认推荐=自然弯曲不触发） | H=1.00（最严=旧 0.80 减 0.10） |
|---------|------|---------|--------------------------|----------|
| **ENTER_BEND_RATIO** | 弯入线（弯到此线以下=开始算"在弯曲"），实际 enter_t = 基线 × ENTER_BEND_RATIO；**越小越严（必须弯更低）** | **0.62 → 必须至少弯下去 38%** ✨整体下移 0.10 | **0.62 + 0.7×0.08 = 0.676≈0.68 → 必须弯下去 32%** ✨自然放松状态通常只有 15%~20% 弯曲度，永远达不到这个门槛 → 彻底杜绝误判 | 0.70 → 必须弯 30%（旧 0.80 下移 0.10，间距仍=0.08） |
| **MIN_BEND_DROP** | 相对近期伸直高值的**最低降幅**（第二道硬门槛：`(max_recent - bend) / max_recent` 必须 > 此值）；**越大越严** | 0.10 → 降 10% | **0.212≈0.21 → 必须相对高值弯下去 21%** ✨双重保险：哪怕 enter_t 过了但降幅不够也不触发 | 0.26 → 降 26%（用户未改，公式不变） |
| **EXIT_BEND_RATIO** | 伸直线（回弹到此线以上=释放/单击完成），与 ENTER 形成滞回区间，避免来回抖动；实际 exit_t = 基线 × EXIT | ENTER + 0.06 = **0.68** ✨ENTER 下移 EXIT 同步下移，更早识别回弹释放 | ENTER + 0.06 = **0.736≈0.74** ✨回弹阈值也整体下移 0.10，单击响应更快 | ENTER + 0.06 = 0.76（滞回差 0.06 保持，封顶 0.92） |

### 3.2 `--tap-speed`（响应速度）→ 5 个"响应性/稳定性"参数
| 内部阈值 | 含义 | S=0.00（最稳最慢） | S=0.80（默认推荐=敏感快速） | S=1.00（极速响应） |
|---------|------|---------|--------------------------|----------|
| **BEND_SMOOTH_ALPHA** | bend EMA 平滑系数；**越大=越跟随 raw bend（响应快），越小=越平滑（稳但滞后）** | 0.55 → 慢稳 | **0.87** ✨ 兼顾快和平滑，用户觉得敏感够用 | 0.95 → 几乎直接跟随 raw bend |
| **MIN_BEND_HOLD** | 进入弯曲的最小确认帧数（避免单帧尖峰）；**越少=越快响应**，0=单帧立刻确认 | 2（至少连续2帧弯才算） | **0** ✨ 用户要"速度"，单帧即确认 | 0 |
| **TAP_MAX_FRAMES** | ✨**快速短按优先级总帧数**（用户做"弯→弹"的总帧数 ≤ 此值 → 无论中间发生什么一律算单击 TAP，绝不可能变长按/拖拽）；**越大=短按动作越宽松，越难误成长按** | 14 → 极短时间窗 | **25** ✨ 充裕的单击时间窗（0.8~0.9秒），解决"单点变长按"痛点 | 28 → 最宽松 |
| **BEND_STABLE_THRESH** | "弯曲深度是否稳定"判据（用于长按计数冻结）；**越小=越宽松=更容易判定为"停住了"→长按帧数更快累计** | 0.020（2%，严=慢稳） | **0.008（0.8%）** ✨ 小抖动也算稳定，正常单击稳定帧足够累计但不会太快 | 0.005（最宽松） |
| **BEND_STABLE_MIN_HOLD** | 连续多少帧稳定才算"真的稳定"（防抖），不受 speed 影响，固定=2 | 2 | 2 | 2 |

### 3.3 固定阈值（不随 H/S 变化，用户明确要求保持）
| 阈值 | 值 | 说明 | 源码位置 |
|------|----|------|----------|
| **LONG_PRESS_FRAMES** | **22 帧**（≈30FPS 下 730ms） | 稳定帧累计 ≥ 23 帧 → 触发 LONG_DOWN 拖拽。用户要求恢复此默认值，不参与 H/S 映射。 | [resolve_tap_params L96/L105](file:///d:/TraeProj/HandDetecter/finger_tap_click_detector.py#L96-L105) |
| **COOLDOWN_FRAMES** | **4 帧**（≈130ms） | 每一次 TAP / LONG_UP 之后的冷却窗口（解决用户说"连点第二下卡死"痛点）。固定值，如需微调改代码 `COOLDOWN_FRAMES = 4` 这一行。 | [L58](file:///d:/TraeProj/HandDetecter/finger_tap_click_detector.py#L58) / [L108](file:///d:/TraeProj/HandDetecter/finger_tap_click_detector.py#L108) / [L319](file:///d:/TraeProj/HandDetecter/finger_tap_click_detector.py#L319) |

---

## 四、场景化调参（按你遇到的现象 → 直接改什么参数）

### 4.1 食指点击类问题
| 你遇到的现象 | 根因 | 推荐改法 |
|-------------|------|---------|
| 🟠 小幅度食指轻微颤动，就误触发了"单击" | hardness 太低 → ENTER/DROP 两道门槛都太松 | `↑ --tap-hardness`：默认 0.70 → 先试 `0.85`，仍有问题拉满 `1.0`；✨本轮已整体平移-0.10，默认H=0.7需弯32%，自然放松（15%~20%）已经不会触发了 |
| 🟠 刻意弯到底了还是完全没反应，右上角状态一直是"伸直待机" | hardness 太高 / speed 太低 → 永远达不到 DROP 门槛或 MIN_HOLD 太慢 | `↓ --tap-hardness` 0.70→`0.50`，同时 `↑ --tap-speed` 到 `0.9` |
| 🔴 单点经常被识别为**长按拖拽**，经过其他按钮时就误点了 | ① TAP_MAX 时间窗太短，用户动作慢了一点就超；② EXIT 检测慢（α 太低）导致回弹 3~5 帧后才满足 exit_t | `↑ --tap-speed` 到 **0.90~1.0**（TAP_MAX 从 25→28，α 变快回弹立刻感知）；如果仍有问题再 `↑ --tap-hardness` 0.05~0.1（把弯幅门槛抬一点，慢弯过程不会被计入） |
| 🟠 连点第二下按不出来（点了两次只响应 1 次） | COOLDOWN 冷却太长，或者 TAP_MAX 窗太短第二次弯曲被第一次冷却挡掉 | `↑ --tap-speed` 先到 `0.9`；若仍不行，**改代码** [finger_tap_click_detector.py L58](file:///d:/TraeProj/HandDetecter/finger_tap_click_detector.py#L58) 的 `COOLDOWN_FRAMES = 4 → 2` |
| 🔴 长按拖拽要按住好久才触发（or 根本触发不了） | ① stable_thresh 太小，稍微抖就不算稳定；② H 太高弯幅达不到一直停在 BEND_HOLD 不累计 | `↓ --tap-speed` 到 `0.5`（stable_thresh=0.0125 更严但稳定帧仍会累计，给足够时间达 22 帧）；同时 `↓ --tap-hardness` 0.7→0.6，先让食指更容易进弯曲状态 |
| 🟠 相反：不小心就进入了长按模式（一弯就拖） | `↓ --tap-speed` 到 `0.5`，MIN_BEND_HOLD 变 1 + TAP_MAX 缩 21，短按优先级更早触发 TAP | 同左；如果连 SHIFT 键都要按，再配合 `↑ --tap-hardness 0.85` 提高弯幅门槛 |

### 4.2 鼠标移动 / 光标类问题
| 你遇到的现象 | 根因 | 推荐改法 |
|-------------|------|---------|
| 🟠 手掌在画面里移到极限了，光标还只在屏幕中央 50% 区域，**碰不到边角** | ROI 太大 → 画面 100% 映射屏幕，手得非常大的动作才能覆盖全屏；或者你用的是 smooth 模式（经过中值/1€/Sigmoid 阻尼=位移被吃掉一截） | ① **切 `--cursor-mode absolute`**（绝对坐标无阻尼）；② 同时 `↓ --roi-scale`：0.85→`0.70`→`0.55`。每次降 0.1 试，直到画面边缘掌心正好对应屏幕边缘 |
| 🟠 光标**跟着食指弯一起漂**，点下去歪了 | 你用了 `--anchor index_tip`（点击时光标锚点=正在弯的食指尖，必然漂） | 必须切回默认 `--anchor palm`（掌心锚点，食指弯曲不会改变掌心位置） |
| 🟠 光标太**抖/Jitter**，悬停不到一个小按钮上 | 你用了 `absolute` 模式但指尖小抖动就直接跳像素；或者 ROI 太小（0.55 把画面抖动放大了 1.8 倍） | ① 切 `--cursor-mode smooth`（5 层流水线 AR 悬停级平滑）；② `↑ --roi-scale` 到 0.80 以上（缩放系数小=抖动不被放大） |
| 🟠 光标移动**太滞后/不跟手**，手停了光标还在动 | smooth 模式的 1€ Filter + Sigmoid 响应曲线 = 慢动超平滑但有 100~200ms 延迟 | 切 `--cursor-mode absolute` + `--roi-scale 0.75`（绝对坐标直接跳，只有 3 帧中值滤波杀尖峰=延迟 ~100ms） |
| 🟠 手**移出摄像头画面后，鼠标只能停在中间不远的位置** | smooth 模式下的边缘死区 + 多层平滑衰减 | 同上：切 `absolute` 模式 + 降 ROI 到 0.70 |

### 4.3 双手 / 手势识别类问题
| 现象 | 根因 | 改法 |
|------|------|------|
| V 字手势（PEACE）识别不出来 | 伸直阈值 / 二次校验太严 | 改 [gesture_recognizer.py PEACE 分支阈值](file:///d:/TraeProj/HandDetecter/gesture_recognizer.py#L119-L125)：index_ratio>0.70 / middle_ratio>0.70 → 下调到 0.65；ring/pinky < 0.55 → 放宽到 0.60 |
| 左手一直识别不到（只识别了一只） | max_hands 被改成 1 了 | 确认参数 `--max-hands 2`（默认就是） |

---

## 五、UI 可视化识读（画面右上角食指条 6 行详细说明）
> 绘制函数：[main.py _draw_tap_bar](file:///d:/TraeProj/HandDetecter/main.py#L82-L194)
> 所有识别状态 + 进度 + 参数档位全部可视化，不需要开 Debug 日志也能知道"当前系统认为你在做什么"。

| 行号 | 内容 | 怎么看 / 调参参考 |
|------|------|------------------|
| 第 1 行 | 食指伸直度大进度条 + 三角形标记 | 三角=你当前 bend 值；**白色竖线=用户真实伸直基线**；**红线=ENTER 弯入线（H 越高越靠右）**；**黄线=EXIT 伸直线**。点击过程：三角从右→左穿过红线→保持左→弹回穿过黄线=一次 TAP。 |
| 第 2 行 | 中文状态文字 | `伸直待机` = 基线学习中；`食指弯曲确认中` = 刚进入弯曲（BEND_ARMED）；`弯曲保持 X 帧` = 正在判定 TAP / LONG；`✅ TAP 单击触发` = 刚发了 click；`🔒 按住拖拽中` = 已经发 LONG_DOWN，现在移动手=拖；`冷却 X 帧` = 冷却窗内，防止连发。 |
| 第 3 行 | 左：短按窗口进度条（绿→黄→红）+ 右：长按稳定计数进度条（蓝→橙→红） | **左短按条：<70%绿=安全单击区；<100%黄=仍在短按优先级；变红=超过 TAP_MAX，再回弹就可能变长按**。你做单击动作时：**确保左条变红前三角已经弹回穿过黄线 = 100% 不会变拖拽**。<br>**右长按条：满红=下帧就发 LONG_DOWN 拖拽**，你想单击的话必须让它永远不到 100%。 |
| 第 4 行 | `bend=X.XX 基线=X.XX` | 数值化看第 1 行的两个关键值。基线如果一直在降（<0.8），说明你没有完全伸直食指，基线学歪了 → 手放开伸直到初始姿势 2 秒让基线恢复。 |
| 第 5 行 | `弯入线=X.XX(红) 伸直线=X.XX(黄) 单击×N 长按×N` | 显示当前生效的 ENTER/EXIT 阈值（随 H 变化）。累计次数帮你观察"单击成功率 vs 长按误触发率"。 |
| **第 6 行（本轮新增蓝字）** | `硬度 H=NN% 速度 S=NN%  弯幅≥XX%+降XX%` | **一眼知道当前参数档位**：硬度百分比就是你传的 --tap-hardness × 100，速度百分比同理。`弯幅≥XX%` 就是「你必须至少弯下去 XX%」才会触发的第一道门槛（ENTER 折算），`+降XX%` 就是 MIN_DROP 第二道门槛。 |

---

## 六、参数生效 / 调试技巧

1. **改完参数必须重启**：本项目不支持运行期热重载参数，必须关窗口再启动命令。
2. **想"回默认"只需删除该参数**：比如你不确定 hardness 应该多少，就去掉 `--tap-hardness` 整个参数，系统就用默认 0.70。
3. **极端参数测试法**：
   - 想快速确认「hardness 生效」→ 开 `--tap-hardness 1.0`，轻微弯食指看状态是否一直停在伸直待机（不进弯曲=生效）。
   - 想快速确认「speed 生效」→ 开 `--tap-speed 1.0`，点完立刻看右上角状态是否立刻弹 TAP 而不是停冷却很久。
4. **最终参数记录**：当你找到一套非常顺手的组合后，把完整命令粘在 Windows 的快捷方式目标栏（或做一个 .bat 脚本），下次直接双击即可。
   - 例（.bat 脚本，保存为 `启动-办公模式.bat`）：
     ```bat
     @echo off
     cd /d d:\TraeProj\HandDetecter
     py -3.10 main.py --use-tasks --anchor palm --cursor-mode smooth --roi-scale 0.85 --tap-hardness 0.70 --tap-speed 0.80
     pause
     ```
