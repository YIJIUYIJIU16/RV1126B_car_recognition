import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PLATE_CHARS = [
    "#", "京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
    "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "学",
    "警", "港", "澳", "挂", "使", "领", "民", "航", "危", "0", "1",
    "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D",
    "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R",
    "S", "T", "U", "V", "W", "X", "Y", "Z", "险", "品",
]

LABEL_TEXTS = [
    "单层车牌",
    "双层车牌",
    "汽车",
    "黑色",
    "蓝色",
    "绿色",
    "白色",
    "黄色",
    "棕色",
    "灰色",
    "橙色",
    "粉色",
    "紫色",
    "红色",
    "车辆颜色:",
    "双层",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate PNG glyphs from platech.ttf for C++ rendering.")
    parser.add_argument(
        "--font",
        default="/data1/CYC/RV/Car_recognition/fonts/platech.ttf",
        help="Path to platech.ttf",
    )
    parser.add_argument(
        "--output-dir",
        default="../model/platech_glyphs",
        help="Directory where generated glyph PNGs will be saved.",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=64,
        help="Base font size for glyph rasterization.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=8,
        help="Padding around glyphs before cropping.",
    )
    return parser.parse_args()


def collect_chars():
    chars = set(PLATE_CHARS)
    for text in LABEL_TEXTS:
        chars.update(text)
    return sorted(chars)


def codepoint_name(ch: str) -> str:
    return f"u{ord(ch):04X}.png"


def render_glyph(font: ImageFont.FreeTypeFont, ch: str, padding: int) -> Image.Image:
    dummy = Image.new("L", (256, 256), 0)
    draw = ImageDraw.Draw(dummy)
    draw.text((padding, padding), ch, fill=255, font=font)
    bbox = dummy.getbbox()
    if bbox is None:
        return Image.new("RGBA", (padding * 2, padding * 2), (0, 0, 0, 0))

    cropped = dummy.crop(bbox)
    rgba = Image.new("RGBA", cropped.size, (255, 255, 255, 0))
    rgba.putalpha(cropped)
    return rgba


def main():
    args = parse_args()
    font_path = Path(args.font).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    font = ImageFont.truetype(str(font_path), args.font_size, encoding="utf-8")
    chars = collect_chars()

    generated = 0
    for ch in chars:
        glyph = render_glyph(font, ch, args.padding)
        save_path = output_dir / codepoint_name(ch)
        glyph.save(save_path)
        generated += 1

    print(f"Generated {generated} glyph PNGs in: {output_dir}")


if __name__ == "__main__":
    main()
