"""
cv2 中文文本绘制工具
====================
OpenCV 原生 cv2.putText 不支持中文/CJK字符，会显示为方块乱码。
解决方案：用 PIL.ImageDraw 渲染 TrueType 字体后转换回 OpenCV BGR 数组。

Windows 字体候选（按优先级）：微软雅黑 → 黑体 → 宋体 → 系统默认
字体缺失时自动降级为英文提示+cv2.putText，保证主循环不崩。

⭐ 2026-09-20 性能重构（主循环 8fps→修帧率瓶颈）：
  · 文本渲染结果（灰度蒙版=字形抗锯齿覆盖率）按 (text, 字号, 粗体) 缓存，
    主循环里的手标签/状态文案/数字条绝大多数是重复串 → 命中缓存后每处文字
    只剩一次小 ROI 的 numpy alpha 混合（实测 ~0.05ms，旧版 2~6ms/处）。
  · 蒙版与颜色解耦：白色字形渲染一次，任意颜色在合成时套用，缓存键不含颜色。
  · 粗体不再用"同一文字偏移画 5 遍"模拟，直接加载微软雅黑粗体（msyhbd.ttc）画 1 遍。
"""
import os
from collections import OrderedDict
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
# 粗体优先用真粗体字重，缺失时退化到黑体（本身偏粗），再退回常规候选
_FONT_BOLD_CANDIDATES = [
    r"C:\Windows\Fonts\msyhbd.ttc",   # 微软雅黑粗体
    r"C:\Windows\Fonts\simhei.ttf",   # 黑体
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simsun.ttc",
    r"C:\Windows\Fonts\simkai.ttf",
]

# 不同字号的字体缓存，避免每帧重复加载
_FONT_CACHE = {}

# ⭐ 文本蒙版 LRU 缓存：key=(text, px_size, is_bold) → (alpha(HxW uint8), W, H)
#    alpha=白色字形在黑底上的灰度值=抗锯齿覆盖率；颜色在 blit 时才套用。
_GLYPH_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_GLYPH_CACHE_MAX = 512

# 字形 ROI 内边距（避免裁剪）
_PAD = 4


def _get_font(size: int, bold: bool = False):
    """按字号获取可用字体对象（带缓存）。

    Returns: (font_obj, is_chinese_font)
    """
    if not PIL_AVAILABLE:
        return None, False
    key = (int(size), bool(bold))
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]

    loaded = None
    candidates = _FONT_BOLD_CANDIDATES if bold else _FONT_CANDIDATES
    for path in candidates:
        try:
            if os.path.isfile(path):
                loaded = ImageFont.truetype(path, key[0])
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


def _render_glyph_mask(text: str, px_size: int, bold: bool):
    """渲染并缓存文字灰度蒙版。返回 (alpha[H,W]uint8, W, H)。

    白色 255=字形完全覆盖，0=背景，中间值=抗锯齿。形状/尺寸与颜色无关，
    故同一段文字换色不重渲染；只在首次出现（或数字滚动产生新串）时付一次 PIL 成本。
    """
    key = (text, int(px_size), bool(bold))
    cached = _GLYPH_CACHE.get(key)
    if cached is not None:
        _GLYPH_CACHE.move_to_end(key)
        return cached

    font, _is_cjk = _get_font(px_size, bold)
    # 量测墨迹边界
    dummy = Image.new("L", (1, 1), 0)
    dd = ImageDraw.Draw(dummy)
    try:
        bbox = dd.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        ox, oy = bbox[0], bbox[1]
    except Exception:
        tw, th = dd.textsize(text, font=font)
        ox, oy = 0, 0
    W, H = tw + _PAD * 2, th + _PAD * 2
    mask_img = Image.new("L", (max(1, W), max(1, H)), 0)
    md = ImageDraw.Draw(mask_img)
    # 粗体由粗体字重本身承担，只画 1 遍（旧版偏移画 5 遍=5 倍 PIL 成本）
    md.text((_PAD - ox, _PAD - oy), text, font=font, fill=255)
    alpha = np.asarray(mask_img, dtype=np.uint8)

    _GLYPH_CACHE[key] = (alpha, W, H)
    _GLYPH_CACHE.move_to_end(key)
    if len(_GLYPH_CACHE) > _GLYPH_CACHE_MAX:
        _GLYPH_CACHE.popitem(last=False)
    return alpha, W, H


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
        thickness:   >=2 时使用粗体字重（真粗体字体，单遍渲染并缓存）
        bg_color:    (B, G, R) 可选，背景矩形底色（提升可读性）
    Returns:
        绘制完成的 frame（与原对象同一引用）
    """
    # 字号 → PIL 像素高度：经验公式 (font_scale * 33，近似 cv2 字号观感
    px_size = max(10, int(font_scale * 33))
    font, is_cjk = _get_font(px_size, bold=(thickness >= 2))

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

    # ---------- 路径 3：缓存蒙版 + ROI alpha 混合 ----------
    alpha, tw, th = _render_glyph_mask(text, px_size, bold=(thickness >= 2))

    h, w = frame.shape[:2]
    # 裁剪 ROI（超出边界做裁剪，避免越界）
    x1 = min(w - 1, max(0, x0))
    y1 = min(h - 1, max(0, y0))
    x2 = min(w, x1 + tw)
    y2 = min(h, y1 + th)
    rw, rh = x2 - x1, y2 - y1
    if rw < 4 or rh < 4:
        return frame

    # 可选底色：整块 ROI 填充（cv2 矩形，亚毫秒）
    if bg_color is not None:
        frame[y1:y2, x1:x2] = (
            int(bg_color[0]), int(bg_color[1]), int(bg_color[2]))

    roi = frame[y1:y2, x1:x2]
    a = alpha[:rh, :rw].astype(np.float32) * (1.0 / 255.0)
    txt_col = np.array([int(color[0]), int(color[1]), int(color[2])],
                       dtype=np.float32)
    # out = roi*(1-a) + 字色*a（a 升一维广播到 BGR 三通道）
    blended = roi.astype(np.float32) * (1.0 - a[:, :, None]) \
        + txt_col[None, None, :] * a[:, :, None]
    frame[y1:y2, x1:x2] = blended.astype(np.uint8)
    return frame


def _fallback_ascii_put(frame, text, pos, font_scale, color, thickness):
    """降级：只用 cv2.putText 绘制。遇到非 ASCII 字符时用 '?' 代替。"""
    safe = "".join(c if ord(c) < 128 else "?" for c in text)
    cv2.putText(frame, safe, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    return frame
