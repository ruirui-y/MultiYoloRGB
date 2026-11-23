r"""把一张 4 通道图的四个通道拆开看：RGB 三个通道 + 第 4 通道（深度）。

产物是一张四栏图：
    第 1 栏 RGB（三个通道合起来）
    第 2 栏 第 4 通道原样（亮 = 远、暗 = 近）
    第 3 栏 第 4 通道反转后（255 − 原值）
    第 4 栏 第 4 通道的数值直方图

用法：
    .venv\Scripts\python.exe show_depth_channel.py
    .venv\Scripts\python.exe show_depth_channel.py --image 路径.png
"""
from __future__ import annotations
import paths


import argparse
import glob
import os
from pathlib import Path

import cv2
import numpy as np

DEFAULT_DIR = str(paths.DEFAULT_DATASET / "rgbd" / "images" / "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="拆开看第 4 通道")
    parser.add_argument("--image", default=None, help="不给就自动取目录里第一张")
    parser.add_argument("--dir", default=DEFAULT_DIR)
    parser.add_argument("--output", default=r"H:\YJJ\Yolo_RGBD\MultiYoloRGB\runs\depth_channel_view.png")
    return parser.parse_args()


def label(panel: np.ndarray, text: str) -> np.ndarray:
    """在图上加一条黑底标题。"""
    canvas = panel.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(canvas, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return canvas


def main() -> None:
    args = parse_args()
    path = Path(args.image) if args.image else Path(sorted(glob.glob(os.path.join(args.dir, "*.png")))[0])
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.shape[2] != 4:
        raise ValueError(f"不是 4 通道图: {path}")
    rgb = image[:, :, :3]
    depth = image[:, :, 3]
    valid = depth > 0
    inverted = np.where(valid, 255 - depth, 0).astype(np.uint8)

    # 直方图：只统计有效像素，横轴 0~255，纵轴像素数
    hist_height, hist_width = 240, 480
    hist = np.zeros((hist_height, hist_width, 3), dtype=np.uint8)
    counts, _ = np.histogram(depth[valid], bins=64, range=(0, 255))
    if counts.max() > 0:
        scaled = (counts / counts.max() * (hist_height - 20)).astype(int)
        bar_width = hist_width // 64
        for index, value in enumerate(scaled):
            x1 = index * bar_width + 1
            cv2.rectangle(hist, (x1, hist_height - value), (x1 + bar_width - 2, hist_height), (200, 200, 200), -1)
    mean_value = float(depth[valid].mean()) if valid.any() else 0.0
    cv2.putText(hist, f"depth values (mean={mean_value:.0f})", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(hist, "left=1 (near)      right=255 (far)", (10, hist_height - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

    panels = [
        label(rgb, "1) RGB (channels 1-3)"),
        label(cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR), "2) channel 4 = depth: bright=far, dark=near"),
        label(cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR), "3) channel 4 inverted (255 - value)"),
        label(hist, "4) how many pixels at each depth value"),
    ]
    height = max(p.shape[0] for p in panels)
    padded = []
    for panel in panels:
        if panel.shape[0] < height:
            pad = np.zeros((height - panel.shape[0], panel.shape[1], 3), dtype=np.uint8)
            panel = np.vstack([panel, pad])
        padded.append(panel)
    combined = np.hstack(padded)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), combined)
    print(f"输入: {path}")
    print(f"有效像素比例: {valid.mean() * 100:.1f}%   深度平均值: {mean_value:.1f}   最小/最大: "
          f"{int(depth[valid].min())}/{int(depth[valid].max())}")
    print(f"已保存: {output}")


if __name__ == "__main__":
    main()
