"""
cv2 中文文本绘制工具
====================
OpenCV 原生 cv2.putText 不支持中文/CJK字符，会显示为方块乱码。
解决方案：用 PIL.ImageDraw 渲染 TrueType 字体后转换回 OpenCV BGR 数组。

Windows 字体候选（按优先级）：微软雅黑 → 黑体 → 宋体 → 系统默认
字体缺失时自动降级为英文提示+cv2.putText，保证主循环不崩。
"""
import os
import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False
    ImageFont = None  # type: ignore

# 候选字体路径（Windows 10/11 自带）
_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",     # 微软雅黑（推荐）
    r"C:\Windows\Fonts\msyhbd.ttc",   # 微软雅黑粗体
    r"C:\Windows\Fonts\simhei.ttf",   # 黑体
    r"C:\Windows\Fonts\simsun.ttc",   # 宋体
    r"C:\Windows\Fonts\simkai.ttf",   # 楷体
]

# 不同字号的字体缓存，避免每帧重复加载
_FONT_CACHE = {}


def _get_font(size: int):
    """按字号获取可用字体对象（带缓存）。

    Returns: (font_obj, is_chinese_font)
    """
    if not PIL_AVAILABLE:
        return None, False
    key = int(size)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]

    loaded = None
    for path in _FONT_CANDIDATES:
        try:
            if os.path.isfile(path):
                loaded = ImageFont.truetype(path, key)
                break
        except Exception:
            continue
    if loaded is None:
        # 找不到任何中文字体，退回 PIL 默认字体（只支持英文）
        try:
            loaded = ImageFont.load_default()
            _FONT_CACHE[key] = (loaded, False)
        except Exception:
            _FONT_CACHE[key] = (None, False)
    else:
        _FONT_CACHE[key] = (loaded, True)
    return _FONT_CACHE[key]


def put_text_cn(
    frame: np.ndarray,
    text: str,
    position,
    font_scale: float = 0.6,
    color=(255, 255, 255),
    thickness: int = 2,
    bg_color=None,
):
    """在 OpenCV 画面上绘制支持中文的文本。

    Args:
        frame:       原始 BGR 图像（会被直接修改）
        text:        任意字符串（支持中英文混合）
        position:    (x, y) 文字左上角坐标
        font_scale:  相对字号（0.5=小 0.6=正常 0.8=大 1.0=特大）
        color:       (B, G, R) 文字颜色
        thickness:   字重（实际用于加粗偏移）
        bg_color:    (B, G, R) 可选，背景矩形底色（提升可读性）
    Returns:
        绘制完成的 frame（与原对象同一引用）
    """
    # 字号 → PIL 像素高度：经验公式 (font_scale * 30，近似 cv2 字号观感
    px_size = max(10, int(font_scale * 33))
    font, is_cjk = _get_font(px_size)

    x0, y0 = int(position[0]), int(position[1])

    # ---------- 路径 1：PIL 不可用 或 纯 ASCII → 退回到 cv2.putText ----------
    if not PIL_AVAILABLE or font is None:
        return _fallback_ascii_put(frame, text, (x0, y0), font_scale, color, thickness)

    # ---------- 路径 2：PIL 可用 + 非CJK字体但文本含中文 ----------
    has_cjk = any(ord(c) > 127 for c in text)
    if has_cjk and not is_cjk:
        # 没加载到中文字体但文本含中文 → 先加【CN?】标识提示缺字体
        prefix = "[FALLBACK] "
        return _fallback_ascii_put(frame, prefix + text, (x0, y0),
                                    font_scale, color, thickness)

    # ---------- 路径 3：正常中文渲染 ----------
    # 估算画布大小（先估算文本尺寸决定ROI，避免整帧→PIL 转换到需要时才做转换只绘制的 ROI）
    try:
        # Pillow 9.2+ 用 textbbox；旧版 textsize
        dummy_img = Image.new("RGB", (1, 1))
        dummy_draw = ImageDraw.Draw(dummy_img)
        try:
            bbox = dummy_draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = dummy_draw.textsize(text, font=font)
    except Exception:
        tw, th = len(text) * px_size, px_size

    # 加 4 像素内边距，避免裁剪
    pad = 4
    tw += pad * 2
    th += pad * 2

    h, w = frame.shape[:2]
    # 裁剪 ROI（超出边界的话做裁剪，避免越界）
    x1 = min(w - 1, max(0, x0))
    y1 = min(h - 1, max(0, y0))
    x2 = min(w, x1 + tw)
    y2 = min(h, y1 + th)
    rw, rh = x2 - x1, y2 - y1
    if rw < 4 or rh < 4:
        return frame

    # ROI → PIL RGB
    roi_bgr = frame[y1:y2, x1:x2].copy()
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(roi_rgb)
    draw = ImageDraw.Draw(pil_img)

    # 可选底色
    if bg_color is not None:
        br, bg, bb = int(bg_color[0]), int(bg_color[1]), int(bg_color[2])
        draw.rectangle([(0, 0), (rw - 1, rh - 1)], fill=(bb, bg, br))

    # 绘制多次模拟粗体（PIL没有thickness，用偏移叠加近似）
    r, g, b = int(color[2]), int(color[1]), int(color[0])  # BGR -> RGB
    fill_color = (r, g, b)
    if thickness >= 2:
        draw.text((pad - 1, pad), text, font=font, fill=fill_color)
        draw.text((pad + 1, pad), text, font=font, fill=fill_color)
        draw.text((pad, pad - 1), text, font=font, fill=fill_color)
        draw.text((pad, pad + 1), text, font=font, fill=fill_color)
    draw.text((pad, pad), text, font=font, fill=fill_color)

    # 粘贴回 frame
    out_rgb = np.array(pil_img)
    frame[y1:y2, x1:x2] = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    return frame


def _fallback_ascii_put(frame, text, pos, font_scale, color, thickness):
    """降级：只用 cv2.putText 绘制。遇到非 ASCII 字符时用 '?' 代替。"""
    safe = "".join(c if ord(c) < 128 else "?" for c in text)
    cv2.putText(frame, safe, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    return frame
