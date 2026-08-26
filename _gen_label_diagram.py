"""生成 五指张开 · MediaPipe Hands 21 Landmark 全指节标示图 hand_joints_label.png
标准命名（每根手指 从指根→指尖 的通用关节名缩写）：
  0     WRIST               腕部（公共）
  ├─ 拇指 Thumb：
  │     1 CMC(腕掌)  2 MCP(掌指)  3 IP(指间)   4 TIP(指尖)
  ├─ 食指 Index：
  │     5 MCP(掌指)  6 PIP(近指间)  7 DIP(远指间)  8 TIP(指尖)
  ├─ 中指 Middle：
  │     9 MCP        10 PIP         11 DIP        12 TIP
  ├─ 无名指 Ring：
  │    13 MCP        14 PIP         15 DIP        16 TIP
  └─ 小指 Pinky：
       17 MCP        18 PIP         19 DIP        20 TIP
说明：PIP=近端指间关节（靠近掌的一节），DIP=远端指间关节（靠指尖的一节）；拇指只有 2 节，所以用 IP（指间）代替 PIP+DIP。
"""
from PIL import Image, ImageDraw, ImageFont
import os

W, H = 1600, 1100
img = Image.new("RGB", (W, H), (248, 250, 252))
draw = ImageDraw.Draw(img)

# ---------- 字体 ----------
FONT_PATHS = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\arial.ttf",
]
fn14, fn18, fn22, fn28, fn36, fn48 = [None]*6
for fp in FONT_PATHS:
    if os.path.exists(fp):
        try:
            fn14 = ImageFont.truetype(fp, 14)
            fn18 = ImageFont.truetype(fp, 18)
            fn22 = ImageFont.truetype(fp, 22)
            fn28 = ImageFont.truetype(fp, 28)
            fn36 = ImageFont.truetype(fp, 36)
            fn48 = ImageFont.truetype(fp, 48)
            break
        except Exception:
            continue
if fn14 is None:
    fn14=fn18=fn22=fn28=fn36=fn48 = ImageFont.load_default()

# ---------- 标题 ----------
draw.text((W//2, 45),  "MediaPipe Hands 21 Landmark · 五指张开全指节标识",
          fill=(20, 40, 80), font=fn48, anchor="ma")
draw.text((W//2, 95),  "通用关节名：MCP(掌指关节) · PIP(近端指间) · DIP(远端指间) · TIP(指尖) · 拇指特殊=CMC/IP",
          fill=(70, 90, 130), font=fn28, anchor="ma")
draw.text((W//2, 130), "掌心锚点(鼠标光标) = (#0 WRIST + #9 MIDDLE_MCP) / 2",
          fill=(210, 110, 0), font=fn28, anchor="ma")

# ======================================================================
# 「手掌张开」手型示意图的 21 个 landmark 像素坐标（手绘拓扑）
# 画布偏左下，留顶部标题和右侧说明面板
#   掌根中心 CX=650, CY=680
CX, CY = 620, 700
LM = {}
LM[0] = (CX,        CY + 240)           # 0  WRIST 腕底

# 拇指（从腕部向左上方展开，约45°外展）——4节
LM[1] = (CX - 130,  CY + 160)          # 1  THUMB_CMC
LM[2] = (CX - 240,  CY +  60)          # 2  THUMB_MCP
LM[3] = (CX - 330,  CY -  30)          # 3  THUMB_IP
LM[4] = (CX - 390,  CY - 120)          # 4  THUMB_TIP

# 食指（竖直偏右一点）——4节
LM[5]  = (CX -  30,  CY +  80)         # 5  INDEX_MCP
LM[6]  = (CX -  40,  CY -  70)         # 6  INDEX_PIP
LM[7]  = (CX -  55,  CY - 210)         # 7  INDEX_DIP
LM[8]  = (CX -  70,  CY - 335)         # 8  INDEX_TIP

# 中指（正上方，最长）——4节
LM[9]  = (CX +  60,  CY +  90)         # 9  MIDDLE_MCP（=掌心中心锚点端点）
LM[10] = (CX +  70,  CY -  90)         # 10 MIDDLE_PIP
LM[11] = (CX +  85,  CY - 250)         # 11 MIDDLE_DIP
LM[12] = (CX + 100,  CY - 390)         # 12 MIDDLE_TIP

# 无名指（竖直偏右，略短于中指）——4节
LM[13] = (CX + 155,  CY +  78)         # 13 RING_MCP
LM[14] = (CX + 180,  CY -  80)         # 14 RING_PIP
LM[15] = (CX + 200,  CY - 230)         # 15 RING_DIP
LM[16] = (CX + 215,  CY - 355)         # 16 RING_TIP

# 小指（最右，最短，外展一点）——4节
LM[17] = (CX + 245,  CY +  50)         # 17 PINKY_MCP（=palm_n_raw 分母端点 A）
LM[18] = (CX + 280,  CY -  30)         # 18 PINKY_PIP
LM[19] = (CX + 305,  CY - 140)         # 19 PINKY_DIP
LM[20] = (CX + 320,  CY - 230)         # 20 PINKY_TIP

# ======================================================================
# 画「骨骼线」：MediaPipe 标准连接拓扑 SKELETON
SKELETON = [
    (0,1),(1,2),(2,3),(3,4),                                        # 拇指
    (0,5),(5,6),(6,7),(7,8),                                        # 食指
    (5,9), (9,10),(10,11),(11,12),                                  # 掌纹→中指（掌纹连接 #5-#9）
    (9,13),(13,14),(14,15),(15,16),                                 # 掌纹→无名指
    (13,17),(0,17), (17,18),(18,19),(19,20),                        # 掌缘 + 小指
]
for (a, b) in SKELETON:
    pa, pb = LM[a], LM[b]
    # 每根手指的颜色：拇指(橙)/食(粉)/中(蓝)/无名(绿)/小指(紫)
    if a in (0,1,2,3) or b in (1,2,3,4):   col = (255, 170, 60)
    elif a in (5,6,7) or b in (6,7,8):     col = (244, 114, 182)
    elif a in (9,10,11) or b in (10,11,12):col = (92, 172, 238)
    elif a in (13,14,15) or b in (14,15,16):col= (46, 204, 113)
    elif a in (17,18,19) or b in (18,19,20):col=(155, 89, 182)
    else:                                   col = (150, 165, 185)  # 掌缘/掌纹灰
    w = 18 if (a not in (0,5,9,13,17) and b not in (0,5,9,13,17)) else 20
    draw.line([pa, pb], fill=col, width=w, joint="curve")

# ======================================================================
# 画每根手指 MCP/PIP/DIP 关节类型的背景色块（清晰看出"哪节叫什么名"）
# 拇指特殊：CMC=MCP(掌侧) / MCP=普通MCP / 只有一个 IP（无 PIP/DIP）
JOINT_TYPE_OF = {
    # landmark_idx -> ("通用关节名", 背景填充色)
    0: ("WRIST",          (220, 230, 245)),
    # 拇指
    1: ("THUMB_CMC",      (255, 225, 180)),
    2: ("THUMB_MCP",      (255, 225, 180)),
    3: ("THUMB_IP",       (255, 200, 130)),
    4: ("THUMB_TIP",      (255, 180,  90)),
    # 食指
    5: ("INDEX_MCP",      (255, 220, 240)),
    6: ("INDEX_PIP",      (255, 190, 225)),
    7: ("INDEX_DIP",      (255, 160, 205)),
    8: ("INDEX_TIP",      (245, 110, 180)),
    # 中指
    9: ("MIDDLE_MCP",     (215, 230, 250)),
    10:("MIDDLE_PIP",     (180, 215, 245)),
    11:("MIDDLE_DIP",     (140, 195, 240)),
    12:("MIDDLE_TIP",     ( 70, 150, 225)),
    # 无名指
    13:("RING_MCP",       (210, 245, 220)),
    14:("RING_PIP",       (170, 235, 195)),
    15:("RING_DIP",       (130, 225, 170)),
    16:("RING_TIP",       ( 40, 195, 100)),
    # 小指
    17:("PINKY_MCP",      (230, 215, 245)),
    18:("PINKY_PIP",      (205, 180, 230)),
    19:("PINKY_DIP",      (180, 145, 215)),
    20:("PINKY_TIP",      (140,  80, 180)),
}

# 掌心锚点高亮 (#0+#9)/2
ax = (LM[0][0] + LM[9][0]) // 2
ay = (LM[0][1] + LM[9][1]) // 2
draw.ellipse([ax-22, ay-22, ax+22, ay+22], outline=(255,140,0), width=4)
draw.ellipse([ax-12, ay-12, ax+12, ay+12], fill=(255, 180, 40))
draw.text((ax, ay - 36), "掌心锚点 = (#0+#9)/2", fill=(210,110,0), font=fn22, anchor="mb")

# 画每个 landmark 点 + #号 + 关节名标签
R_DOT = 15
def _draw_point(idx):
    x, y = LM[idx]
    jname, fill_bg = JOINT_TYPE_OF[idx]
    # 外框 + 彩色背景圆
    draw.ellipse([x-R_DOT-5, y-R_DOT-5, x+R_DOT+5, y+R_DOT+5], fill=(255,255,255))
    draw.ellipse([x-R_DOT,   y-R_DOT,   x+R_DOT,   y+R_DOT],   fill=fill_bg, outline=(80,80,80), width=2)
    # 白色 #号（在点中央）
    draw.text((x, y), f"#{idx}", fill=(20,30,60), font=fn22, anchor="mm")

    # 标签放到点的外侧（根据相对掌心的方向）
    if idx == 0:
        lx, ly = x, y + 45
        anchor = "ma"
    elif idx in (1,2,3,4):  # 拇指：标签放左上侧
        lx, ly = x - 80, y - 5
        anchor = "rm"
    elif idx in (5,6,7,8):  # 食指：标签放左侧
        lx, ly = x - 95, y
        anchor = "rm"
    elif idx in (9,10,11,12):# 中指：标签放正上方
        lx, ly = x + 20, y - 18
        anchor = "lm"
    elif idx in (13,14,15,16):# 无名指：标签放正上方
        lx, ly = x + 20, y - 18
        anchor = "lm"
    elif idx in (17,18,19,20):# 小指：标签放右侧
        lx, ly = x + 25, y
        anchor = "lm"
    else:
        lx, ly = x + 30, y
        anchor = "lm"
    # 画两行：第一行关节名（大字），第二行 landmark编号（小字灰色）
    _, fill_col = JOINT_TYPE_OF[idx]
    # 取一个深点的文字色（与点背景对比）
    text_col = (140, 70, 10) if idx<5 else \
               (170, 50, 120) if idx<9 else \
               (20, 80, 170) if idx<13 else \
               (20, 120, 60) if idx<17 else \
               (100, 40, 150)
    draw.text((lx, ly - 9), jname, fill=text_col, font=fn22, anchor=anchor)
    draw.text((lx, ly + 14), f"landmark #{idx}", fill=(110,120,140), font=fn14, anchor=anchor)

for i in range(21):
    _draw_point(i)

# ======================================================================
# 右上角：通用关节名词典（MCP/PIP/DIP 含义卡片）
DX = W - 520
DY = 160
draw.rounded_rectangle([DX - 20, DY - 25, W - 40, DY + 295], radius=18, fill=(255,255,255), outline=(210,220,230), width=2)
draw.text((DX + 230, DY + 5), "通用关节名缩写 · 说明", fill=(20,40,80), font=fn28, anchor="ma")
glossary = [
    ("WRIST",   "#0",          "整个手的腕部（手掌最底部）"),
    ("CMC",     "#1 (仅拇指)","腕掌关节：腕部 ↔ 拇指第1节"),
    ("MCP",     "#2/#5/#9/#13/#17","掌指关节：手掌边缘 ↔ 第1指节（每指1个）"),
    ("PIP",     "#6/#10/#14/#18",  "近端指间关节：第1节 ↔ 第2指节（靠近掌心的那节·拇指无）"),
    ("DIP",     "#7/#11/#15/#19",  "远端指间关节：第2节 ↔ 第3指节（靠近指尖的那节·拇指无）"),
    ("IP",      "#3 (仅拇指)","指间关节：拇指只有2节，所以合并 PIP+DIP 叫 IP"),
    ("TIP",     "#4/#8/#12/#16/#20","指尖：每根手指最末端的点"),
]
gy = DY + 55
for (abbr, lms, desc) in glossary:
    draw.rounded_rectangle([DX, gy, DX+110, gy+40], radius=10, fill=(235,242,252), outline=(180,200,220))
    draw.text((DX+55, gy+20), abbr, fill=(20,40,80), font=fn22, anchor="mm")
    draw.text((DX+125, gy+10), lms, fill=(10,80,170), font=fn18, anchor="la")
    draw.text((DX+125, gy+30), desc, fill=(80,90,110), font=fn14, anchor="la")
    gy += 42

# ======================================================================
# 右下角：项目本项目用到的几个关键点（摘要）卡片
DX2 = W - 520
DY2 = 500
draw.rounded_rectangle([DX2 - 20, DY2 - 25, W - 40, DY2 + 225], radius=18, fill=(255,245,235), outline=(250,210,170), width=2)
draw.text((DX2 + 230, DY2 + 5), "本项目实际使用的点（摘要）", fill=(210,110,0), font=fn28, anchor="ma")
uses = [
    ("#0 + #9",             "掌心锚点（=鼠标光标位置，二者平均）",           (210,110,0)),
    ("#5 → #17",            "palm_n_raw = dist(#5,#17) 掌心宽度分母",        (90, 60, 180)),
    ("#5-#6-#7",            "∠PIP 角（食指 PIP 辅角 25% 权重）",            (220, 80, 160)),
    ("#6-#7-#8",            "★∠DIP 角（食指 DIP 主角 60% 权重=识别点击）", (10, 100, 200)),
    ("#8 (TIP) - #5 (MCP)", "dist_bend_fallback 长度比值（仅占 5% 兜底）",   (120,120,120)),
]
uy = DY2 + 55
for (title, desc, col) in uses:
    draw.text((DX2, uy+5), title, fill=col, font=fn22)
    draw.text((DX2+245, uy+12), desc, fill=(60,60,80), font=fn14, anchor="la")
    uy += 38

# ======================================================================
# 底部：五指张开标准图标题
draw.text((W//2, H - 35), "以上为 MediaPipe Hands 官方标准 21 个 landmark 的五指张开命名与编号 · 每根手指从 指根→指尖 统一顺序",
          fill=(20, 50, 100), font=fn22, anchor="ma")

out_path = r"d:\TraeProj\HandDetecter\hand_joints_label.png"
img.save(out_path, "PNG")
print(f"SAVED: {out_path}  size={os.path.getsize(out_path)} bytes")
