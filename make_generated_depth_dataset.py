r"""用 YOLO26 的单目深度模型给每张彩色图生成深度，做成 4 通道数据集。

做的事：
    彩色图 → yolo26n-depth.pt → 相对深度（0~1 的小数）
    → 按分位数裁剪后归一化到 1~255（1=近、255=远，与传感器编码方向一致）
    → 与原彩色图拼成 4 通道 PNG，存成新数据集

产出：
    dataset/hand_gen           训练 + 验证（与 hand 同样的划分）
    dataset/hand_common_gen    跨人测试
    runs/generated_depth_stats.csv   每张图的统计（生成深度的范围、与真实深度的相似度）
    docs/fig-13-gen-vs-real.png      真实深度 vs 生成深度 的对比图

用法：
    .venv\Scripts\python.exe make_generated_depth_dataset.py --force
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

DATASET_ROOT = Path(r"H:\YJJ\Yolo_RGBD\Resource\session1_200146\dataset")
TASKS = [("hand", DATASET_ROOT / "hand", DATASET_ROOT / "hand_gen"),
         ("hand_common", DATASET_ROOT / "hand_common", DATASET_ROOT / "hand_common_gen")]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
DOCS = Path(r"H:\YJJ\Yolo_RGBD\MultiYoloRGB\docs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成深度数据集")
    parser.add_argument("--depth-weights", default=r"yolo26n-depth.pt")
    parser.add_argument("--low-percentile", type=float, default=1.0)
    parser.add_argument("--high-percentile", type=float, default=99.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stats", default=str(Path(r"H:\YJJ\Yolo_RGBD\MultiYoloRGB\runs\generated_depth_stats.csv")))
    parser.add_argument("--preview", default=str(DOCS / "fig-13-gen-vs-real.png"))
    return parser.parse_args()


def to_channel(depth: np.ndarray, low: float, high: float) -> np.ndarray:
    """相对深度 → 1~255。低分位对应 1（近），高分位对应 255（远）。"""
    low_value = float(np.percentile(depth, low))
    high_value = float(np.percentile(depth, high))
    if high_value - low_value < 1e-6:
        return np.full(depth.shape, 128, dtype=np.uint8)
    normalized = (depth - low_value) / (high_value - low_value)
    return np.clip(np.round(1 + normalized * 254), 1, 255).astype(np.uint8)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """两幅图的秩相关：越接近 1 说明"哪里近哪里远"越一致。"""
    if a.size < 10:
        return float("nan")
    rank_a = np.argsort(np.argsort(a.ravel()))
    rank_b = np.argsort(np.argsort(b.ravel()))
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def label(panel: np.ndarray, text: str, font) -> np.ndarray:
    canvas = np.zeros((panel.shape[0] + 32, panel.shape[1], 3), dtype=np.uint8)
    canvas[32:] = panel
    if font is not None:
        from PIL import Image, ImageDraw
        image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        ImageDraw.Draw(image).text((6, 4), text, font=font, fill=(255, 255, 255))
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    cv2.putText(canvas, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return canvas


def main() -> None:
    args = parse_args()
    try:
        from PIL import ImageFont
        font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 18)
    except Exception:
        font = None

    depth_model = YOLO(args.depth_weights)
    stats: list[dict[str, Any]] = []
    preview_rows: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []

    for name, src_root, dst_root in TASKS:
        if dst_root.exists():
            if not args.force:
                raise FileExistsError(f"{dst_root} 已存在；要重建请加 --force")
            shutil.rmtree(dst_root)
        rgb_images = src_root / "rgb" / "images"
        if not rgb_images.is_dir():
            print(f"跳过 {name}：没有 {rgb_images}")
            continue
        print("=" * 92)
        print(f"{name}: {src_root} → {dst_root}")
        for split_dir in sorted(rgb_images.iterdir()):
            if not split_dir.is_dir():
                continue
            split = split_dir.name
            files = sorted(p for p in split_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
            out_images = dst_root / "rgbd" / "images" / split
            out_images.mkdir(parents=True, exist_ok=True)
            # 标签原样复制
            label_src = src_root / "rgb" / "labels" / split
            if label_src.is_dir():
                shutil.copytree(label_src, dst_root / "rgbd" / "labels" / split, dirs_exist_ok=True)
            for path in files:
                rgb = cv2.imread(str(path), cv2.IMREAD_COLOR)
                result = depth_model.predict(str(path), verbose=False)[0]
                generated = np.asarray(result.depth.data.cpu()).squeeze().astype(np.float32)
                if generated.shape != rgb.shape[:2]:
                    generated = cv2.resize(generated, (rgb.shape[1], rgb.shape[0]),
                                           interpolation=cv2.INTER_LINEAR)
                channel = to_channel(generated, args.low_percentile, args.high_percentile)
                four_channel = np.dstack([rgb, channel])
                cv2.imwrite(str(out_images / f"{path.stem}.png"), four_channel)

                record = {"dataset": name, "split": split, "image": path.stem,
                          "gen_min": round(float(generated.min()), 4),
                          "gen_max": round(float(generated.max()), 4),
                          "gen_mean": round(float(generated.mean()), 4),
                          "ch4_min": int(channel.min()), "ch4_max": int(channel.max()),
                          "ch4_mean": round(float(channel.mean()), 1),
                          "spearman_vs_real": ""}
                # 有真实深度就比一下结构相似度
                real_path = src_root / "rgbd" / "images" / split / f"{path.stem}.png"
                if real_path.is_file():
                    real = cv2.imread(str(real_path), cv2.IMREAD_UNCHANGED)
                    if real is not None and real.ndim == 3 and real.shape[2] == 4:
                        valid = real[:, :, 3] > 0
                        if valid.any() and valid.shape == channel.shape:
                            record["spearman_vs_real"] = round(
                                spearman(channel[valid], real[:, :, 3][valid]), 4)
                stats.append(record)
                if len(preview_rows) < 2 and split == "test":
                    real = cv2.imread(str(real_path), cv2.IMREAD_UNCHANGED) if real_path.is_file() else None
                    preview_rows.append((path.stem, rgb, channel, real))
            print(f"  {split}: {len(files)} 张")

        (dst_root / "rgbd").mkdir(parents=True, exist_ok=True)
        (dst_root / "rgbd" / "data_rgbd_gen.yaml").write_text(
            "# 生成深度版 RGBD 数据集（由 make_generated_depth_dataset.py 生成）\n"
            f"path: {(dst_root / 'rgbd').as_posix()}\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            "nc: 1\n"
            "names:\n"
            "  0: hand\n"
            "channels: 4\n"
            "kpt_shape: [7, 3]\n",
            encoding="utf-8")

    stats_path = Path(args.stats)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stats[0]))
        writer.writeheader()
        writer.writerows(stats)

    # 结构相似度统计（只看测试集）
    similarities = [row["spearman_vs_real"] for row in stats
                    if row["split"] == "test" and row["spearman_vs_real"] not in ("", None)]
    if similarities:
        values = np.array([float(value) for value in similarities])
        print(f"\n与真实深度的结构相似度（测试集 {len(values)} 张）："
              f"中位数 {np.median(values):.3f}，最小 {values.min():.3f}，最大 {values.max():.3f}")

    # 预览图：RGB / 真实深度 / 生成深度
    if preview_rows:
        panels = []
        for stem, rgb, channel, real in preview_rows:
            row = [label(rgb, f"RGB  {stem[-13:]}", font),
                   label(cv2.cvtColor(channel, cv2.COLOR_GRAY2BGR),
                         "26 生成深度（亮=远）", font)]
            if real is not None and real.ndim == 3 and real.shape[2] == 4:
                row.append(label(cv2.cvtColor(real[:, :, 3], cv2.COLOR_GRAY2BGR),
                                 "传感器真实深度", font))
            height = min(panel.shape[0] for panel in row)
            row = [cv2.resize(panel, (int(panel.shape[1] * height / panel.shape[0]), height))
                   for panel in row]
            panels.append(np.hstack(row))
        width = min(panel.shape[1] for panel in panels)
        panels = [cv2.resize(panel, (width, int(panel.shape[0] * width / panel.shape[1])))
                  for panel in panels]
        preview_path = Path(args.preview)
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(preview_path), np.vstack(panels))
        print(f"预览图: {preview_path}")

    print(f"统计: {stats_path}")


if __name__ == "__main__":
    main()
