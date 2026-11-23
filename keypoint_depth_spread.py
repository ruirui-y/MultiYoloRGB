r"""7 个标注点上的深度差到底有多大？

如果深度要在"7 个点之间"提供区分信息，那这 7 个点的深度值必须明显不同。
这个脚本直接把标注点上第 4 通道的数值取出来，算每张图里 7 个点的最大差值。

第 4 通道编码：1 + (clip(d,1100,1850) − 1100) × 254/750，
所以 1 个通道值 ≈ 2.95mm。

用法：
    .venv\Scripts\python.exe keypoint_depth_spread.py
"""
from __future__ import annotations
import paths


import argparse
import csv
import glob
import os
from pathlib import Path

import cv2
import numpy as np

DEFAULT_ROOT = str(paths.DEFAULT_DATASET)
MM_PER_CHANNEL = 750.0 / 254.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="7 个标注点的深度差")
    parser.add_argument("--dataset-root", default=DEFAULT_ROOT)
    parser.add_argument("--split", default="test")
    parser.add_argument("--radius", type=int, default=2, help="每个点周围取多少像素做中位数")
    parser.add_argument("--output", default=r"H:\YJJ\Yolo_RGBD\MultiYoloRGB\runs\keypoint_depth_spread.csv")
    return parser.parse_args()


def sample(channel: np.ndarray, x: float, y: float, radius: int) -> float | None:
    """在 (x,y) 附近取一小块的中位数，避开单个像素的噪声。"""
    height, width = channel.shape
    x0, x1 = max(0, int(x) - radius), min(width, int(x) + radius + 1)
    y0, y1 = max(0, int(y) - radius), min(height, int(y) + radius + 1)
    patch = channel[y0:y1, x0:x1]
    valid = patch[patch > 0]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root)
    images = root / "rgbd" / "images" / args.split
    label_dir = root / "rgb" / "labels" / args.split
    files = sorted(glob.glob(str(images / "*.png")))

    rows = []
    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        label_path = label_dir / f"{stem}.txt"
        if not label_path.is_file():
            continue
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        height, width = image.shape[:2]
        parts = list(map(float, label_path.read_text(encoding="utf-8").split()))
        points = np.asarray(parts[5:], dtype=np.float64).reshape(-1, 3)
        values = [sample(image[:, :, 3], px * width, py * height, args.radius)
                  for px, py, _visibility in points]
        values = [v for v in values if v is not None]
        if len(values) < 2:
            continue
        spread = max(values) - min(values)
        rows.append({
            "image": stem,
            "points": len(values),
            "min": round(min(values), 1),
            "max": round(max(values), 1),
            "spread_channel": round(spread, 1),
            "spread_mm": round(spread * MM_PER_CHANNEL, 1),
            "median": round(float(np.median(values)), 1),
        })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    spreads = np.array([r["spread_channel"] for r in rows])
    spreads_mm = np.array([r["spread_mm"] for r in rows])
    medians = np.array([r["median"] for r in rows])
    print("=" * 88)
    print(f"图片 {len(rows)} 张，每张取 7 个标注点上的深度值（{args.radius * 2 + 1}×{args.radius * 2 + 1} 中位数）")
    print("=" * 88)
    print(f"7 个点的深度差（最大 − 最小）：")
    print(f"  中位数 {np.median(spreads):.1f} 个通道值  ≈ {np.median(spreads_mm):.0f} mm")
    print(f"  平均   {spreads.mean():.1f} 个通道值  ≈ {spreads_mm.mean():.0f} mm")
    print(f"  最小   {spreads.min():.1f}     最大 {spreads.max():.1f}")
    print(f"  分位数 25% {np.percentile(spreads, 25):.1f} / 50% {np.median(spreads):.1f} / "
          f"75% {np.percentile(spreads, 75):.1f}")
    print()
    print(f"这些点本身的深度值（中位数）：平均 {medians.mean():.1f}，"
          f"范围 {medians.min():.1f} ~ {medians.max():.1f}")
    print()
    print("对照：一个通道值 ≈ 2.95mm；手的整体厚度通常 20~40mm")
    print(f"已保存: {out}")


if __name__ == "__main__":
    main()
