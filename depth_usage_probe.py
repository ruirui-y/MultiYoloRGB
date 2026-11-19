r"""深度用法探针：把第 4 通道改造成三种形态，看模型用到的是哪一部分信息。

四种形态（在 4 通道输入上原地改造第 4 通道，模型与权重完全不动）：
    true_depth      原始深度
    shift_near      深度整张 −150mm（第 4 通道 −51），空间起伏不变，只改远近
    shift_far       深度整张 +150mm（第 4 通道 +51）
    binarize_otsu   按 Otsu 阈值二值化：近处（手）255、远处 1、无效 0
    gray_as_depth   第 4 通道换成原图灰度

第 4 通道编码（与 prepare_pose_rgbd_dataset.py 一致）：
    无效 0；有效 1 + (clip(d,1100,1850) − 1100) × 254/750，即 1~255，越大越远。
    因此 150mm ≙ 150 × 254/750 ≈ 51 个通道值。

用法：
    .venv\Scripts\python.exe depth_usage_probe.py
    .venv\Scripts\python.exe depth_usage_probe.py --models yolo11n_rgbd_4ch --variants true_depth binarize_otsu
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import ablate_c2psa as ac
from ablate_layers import load_items

DEPTH_LOW = 1100
DEPTH_HIGH = 1850
CHANNEL_PER_MM = 254.0 / (DEPTH_HIGH - DEPTH_LOW)
DEFAULT_MODELS = ["yolo11n_rgbd_4ch", "yolov8n_rgbd_4ch"]
VARIANTS = ["true_depth", "shift_near", "shift_far", "binarize_otsu", "gray_as_depth"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="深度用法探针（第 4 通道形态改造）")
    parser.add_argument("--dataset-root", default=str(ac._DEFAULT_DATASET))
    parser.add_argument("--weights-dir", default=str(ac._REPO / "train_weight"))
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--variants", nargs="+", default=None,
                        help="默认跑 VARIANTS 全集；也可只指定若干")
    parser.add_argument("--shifts-channels", nargs="+", type=int, default=None,
                        help="额外的第 4 通道平移量（通道值），会追加成 shift+N 变体")
    parser.add_argument("--split", default="test")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--shift-mm", type=float, default=150.0)
    parser.add_argument("--expected-count", type=int, default=100)
    parser.add_argument("--output-dir", default=str(ac._REPO / "runs" / "depth_usage_probe"))
    return parser.parse_args()


def shift_channel(image: np.ndarray, delta_channels: int) -> np.ndarray:
    """第 4 通道整体加常数：无效像素（0）保持 0，有效像素夹在 1~255。"""
    out = image.copy()
    channel = out[:, :, 3]
    valid = channel > 0
    shifted = channel.astype(np.int16) + delta_channels
    channel[valid] = np.clip(shifted[valid], 1, 255).astype(np.uint8)
    return out


def binarize_channel(image: np.ndarray) -> np.ndarray:
    """按 Otsu 阈值二值化：近处=255、远处=1、无效=0（保留无效与远处的区分）。"""
    out = image.copy()
    channel = out[:, :, 3]
    valid = channel > 0
    if not np.any(valid):
        return out
    values = channel[valid]
    threshold, _ = cv2.threshold(values.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    near = valid & (channel <= threshold)
    far = valid & (channel > threshold)
    channel[near] = 255
    channel[far] = 1
    return out


def gray_as_depth(image: np.ndarray) -> np.ndarray:
    """第 4 通道换成原图灰度（BGR 取灰度），数值范围同样是 uint8 0~255。"""
    out = image.copy()
    out[:, :, 3] = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    return out


def apply_variant(image: np.ndarray, variant: str, shift_channels: int) -> np.ndarray:
    if variant == "true_depth":
        return image
    if variant == "shift_near":
        return shift_channel(image, -shift_channels)
    if variant == "shift_far":
        return shift_channel(image, shift_channels)
    if variant == "binarize_otsu":
        return binarize_channel(image)
    if variant == "gray_as_depth":
        return gray_as_depth(image)
    dynamic = re.fullmatch(r"shift([+-]\d+)", variant)
    if dynamic:
        return shift_channel(image, int(dynamic.group(1)))
    raise ValueError(f"未知变体: {variant}")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root).expanduser()
    weights_dir = Path(args.weights_dir).expanduser()
    out_dir = Path(args.output_dir).expanduser()
    shift_channels = int(round(args.shift_mm * CHANNEL_PER_MM))

    rgbd = ac.stem_map(root / "rgbd" / "images" / args.split)
    rgb = ac.stem_map(root / "rgb" / "images" / args.split)
    label_dir = root / "rgb" / "labels" / args.split
    if not label_dir.is_dir():
        label_dir = root / "rgbd" / "labels" / args.split
    stems = sorted(s for s in rgbd.keys() & rgb.keys() if (label_dir / f"{s}.txt").is_file())
    if args.expected_count and len(stems) != args.expected_count:
        raise RuntimeError(f"期望 {args.expected_count} 张，实际 {len(stems)} 张")

    base_items = load_items(root, args.split, 4, stems)

    print("=" * 104)
    print("深度用法探针：第 4 通道形态改造")
    print(f"数据: {root}  split={args.split}  图片数={len(stems)}")
    print(f"口径: conf={args.conf} imgsz={args.imgsz} match_iou={args.match_iou}")
    print(f"平移: ±{args.shift_mm:.0f}mm ≙ 第 4 通道 ±{shift_channels}（1mm ≙ {CHANNEL_PER_MM:.4f} 通道值）")

    variants = list(args.variants) if args.variants else list(VARIANTS)
    if args.shifts_channels:
        variants += [f"shift{value:+d}" for value in args.shifts_channels]
    print(f"变体: {variants}")
    print("=" * 104)

    summary: list[dict[str, Any]] = []
    for model_name in args.models:
        weight = weights_dir / f"{model_name}.pt"
        if not weight.is_file():
            raise FileNotFoundError(weight)
        for variant in variants:
            model = ac.YOLO(str(weight))
            channels = int(model.model.model[0].conv.weight.shape[1])
            if channels != 4:
                raise RuntimeError(f"{model_name} 不是 4 通道模型（首层 {channels} 通道）")
            items = [(stem, apply_variant(image, variant, shift_channels), gt)
                     for stem, image, gt in base_items]
            started = time.perf_counter()
            try:
                rows = ac.run_condition(model, items, args.conf, args.imgsz, args.match_iou)
            finally:
                del model
            hits = [r for r in rows if r["nme"] is not None]
            entry = {
                "model": model_name,
                "variant": variant,
                "frames": len(rows),
                "hits": len(hits),
                "nme": ac.mean_of([r["nme"] for r in hits]),
                "box_iou": ac.mean_of([r["iou"] for r in hits]),
                "elapsed_s": round(time.perf_counter() - started, 2),
            }
            for k in range(ac.KEYPOINT_COUNT):
                entry[f"k{k}_px"] = ac.mean_of([r[f"k{k}_px"] for r in hits])
            summary.append(entry)
            print(f"{model_name:<22} {variant:<15} hits {entry['hits']:>3}/{len(rows)} "
                  f"NME={ac.fmt(entry['nme'])} IoU={ac.fmt(entry['box_iou'], 3)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["model", "variant", "frames", "hits", "nme", "box_iou", "elapsed_s"] + \
             [f"k{k}_px" for k in range(ac.KEYPOINT_COUNT)]
    write_csv(out_dir / "summary.csv", summary, fields)
    config = {
        "dataset_root": str(root), "split": args.split, "stems": len(stems),
        "conf": args.conf, "imgsz": args.imgsz, "match_iou": args.match_iou,
        "ultralytics": ac._LOCKED_VERSION, "eval_impl": ac._EVAL_SCRIPT.name,
        "models": args.models, "variants": variants,
        "depth_encoding": {"low_mm": DEPTH_LOW, "high_mm": DEPTH_HIGH,
                           "channel_per_mm": CHANNEL_PER_MM, "invalid": 0},
        "shift_mm": args.shift_mm, "shift_channels": shift_channels,
    }
    (out_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n相对原始深度的变化（同一模型内比）")
    print(f"{'model':<22}{'variant':<15}{'hits':>6}{'NME':>10}{'Δ%':>9}{'IoU':>8}")
    for model_name in args.models:
        base = next((r for r in summary if r["model"] == model_name and r["variant"] == "true_depth"), None)
        for row in [r for r in summary if r["model"] == model_name]:
            delta = None
            if base and base["nme"] and row["nme"]:
                delta = (row["nme"] / base["nme"] - 1) * 100
            print(f"{row['model']:<22}{row['variant']:<15}{row['hits']:>6}"
                  f"{ac.fmt(row['nme']):>10}{ac.fmt(delta, 2):>9}{ac.fmt(row['box_iou'], 3):>8}")
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
