r"""一张图看懂"深度中心化"：形状不变，只把整体亮度拉平。

左：原始深度（整体偏亮，平均值约 181）
中：中心化之后（整体被拉到 128，手和背景的相对关系不变）
右：两者的平均值对比条

用法：
    .venv\Scripts\python.exe show_centering.py
"""
from __future__ import annotations
import paths


import argparse
import glob
import os
from pathlib import Path

import cv2
import numpy as np

SRC_DIR = str(paths.DEFAULT_DATASET / "rgbd" / "images" / "test")
CENTER_DIR = str(paths.DATASETS["hand_common_center"] / "rgbd" / "images" / "test")
OUTPUT = r"H:\YJJ\Yolo_RGBD\MultiYoloRGB\runs\centering_view.png"
PANEL = (520, 380)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="中心化前后对比图")
    parser.add_argument("--image", default=None)
    parser.add_argument("--output", default=OUTPUT)
    return parser.parse_args()


def label(panel: np.ndarray, text: str, font) -> np.ndarray:
    canvas = np.zeros((panel.shape[0] + 34, panel.shape[1], 3), dtype=np.uint8)
    canvas[34:] = panel
    if font is not None:
        from PIL import Image, ImageDraw
        image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        ImageDraw.Draw(image).text((6, 5), text, font=font, fill=(255, 255, 255))
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    cv2.putText(canvas, text, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return canvas


def main() -> None:
    args = parse_args()
    try:
        from PIL import ImageFont
        font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 18)
    except Exception:
        font = None

    name = args.image or os.path.basename(sorted(glob.glob(os.path.join(SRC_DIR, "*.png")))[0])
    original = cv2.imread(os.path.join(SRC_DIR, name), cv2.IMREAD_UNCHANGED)
    centered = cv2.imread(os.path.join(CENTER_DIR, name), cv2.IMREAD_UNCHANGED)
    if original is None or centered is None:
        raise FileNotFoundError(f"找不到这张图: {name}")

    o = original[:, :, 3]
    c = centered[:, :, 3]
    valid = o > 0
    mean_o = float(o[valid].mean())
    mean_c = float(c[valid].mean())
    median_o = float(np.median(o[valid]))

    panels = [
        label(cv2.cvtColor(cv2.resize(o, PANEL, interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR),
              f"原样：整体偏亮  平均 {mean_o:.0f}  中位数 {median_o:.0f}", font),
        label(cv2.cvtColor(cv2.resize(c, PANEL, interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR),
              f"中心化后：整体被拉到 128  平均 {mean_c:.0f}", font),
    ]

    # 第三栏：两根平均值柱，直观看出"整体亮度被拉平"
    bars = np.full((PANEL[1], PANEL[0], 3), 30, dtype=np.uint8)
    for index, (value, colour, text) in enumerate([(mean_o, (0, 200, 255), f"orig {mean_o:.0f}"),
                                                  (mean_c, (255, 120, 255), f"center {mean_c:.0f}")]):
        height = int(value / 255 * (PANEL[1] - 70))
        x1 = 120 + index * 200
        cv2.rectangle(bars, (x1, PANEL[1] - 40 - height), (x1 + 90, PANEL[1] - 40), colour, -1)
        # 柱子下的文字用英文，避免 cv2 画不出中文
        cv2.putText(bars, text, (x1 - 26, PANEL[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    overall = label(bars, "整体亮度（平均值）：左边两个数字的对比", font)

    combined = np.hstack([panels[0], panels[1], overall])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), combined)
    print(f"样本: {name}")
    print(f"原样  平均 {mean_o:.1f}  中位数 {median_o:.1f}")
    print(f"中心化 平均 {mean_c:.1f}")
    print(f"已保存: {output}")


if __name__ == "__main__":
    main()
