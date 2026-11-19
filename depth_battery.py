r"""深度敏感性测试矩阵：把能想到的输入改动一次性跑完。

每条测试只改输入（默认改第 4 通道），权重、结构、推理参数全部不动。
另有一组"改 RGB 三个通道"的对照，用来判断某个效应是深度独有还是所有输入都有。
最后一组是把第 1 层输出乘 0.5 与几种深度改动做交叉。

用法：
    .venv\Scripts\python.exe depth_battery.py                      :: 11 全矩阵
    .venv\Scripts\python.exe depth_battery.py --model yolov8n_rgbd_4ch --preset core
    .venv\Scripts\python.exe depth_battery.py --only invert global_shift+102

第 4 通道编码：无效 = 0；有效 = 1~255，数值越大越远（1mm ≙ 0.3387）。
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

import ablate_c2psa as ac
from ablate_layers import LayerAttenuator, load_items

CHANNEL_PER_MM = 254.0 / 750.0


@dataclass(frozen=True)
class Condition:
    name: str
    description: str
    transform: Callable[[np.ndarray], np.ndarray]
    l1_keep: float = 1.0


# ---------------------------------------------------------------- 变换函数
def _valid(image: np.ndarray) -> np.ndarray:
    return image[:, :, 3] > 0


def invert(image: np.ndarray) -> np.ndarray:
    """深度整体反转：255 − 原值，无效像素保持 0。"""
    out = image.copy()
    channel = out[:, :, 3]
    mask = channel > 0
    channel[mask] = 255 - channel[mask]
    return out


def shift_by(delta: int) -> Callable[[np.ndarray], np.ndarray]:
    def apply(image: np.ndarray) -> np.ndarray:
        out = image.copy()
        channel = out[:, :, 3]
        mask = channel > 0
        channel[mask] = np.clip(channel[mask].astype(np.int16) + delta, 1, 255).astype(np.uint8)
        return out
    return apply


def constant(value: int) -> Callable[[np.ndarray], np.ndarray]:
    def apply(image: np.ndarray) -> np.ndarray:
        out = image.copy()
        channel = out[:, :, 3]
        channel[channel > 0] = value
        return out
    return apply


def flat_valid_mean(image: np.ndarray) -> np.ndarray:
    out = image.copy()
    channel = out[:, :, 3]
    mask = channel > 0
    if mask.any():
        channel[mask] = int(np.clip(round(float(channel[mask].mean())), 1, 255))
    return out


def blur(kernel: int, sigma: float = 0.0) -> Callable[[np.ndarray], np.ndarray]:
    def apply(image: np.ndarray) -> np.ndarray:
        out = image.copy()
        channel = out[:, :, 3]
        blurred = cv2.GaussianBlur(channel, (kernel, kernel), sigma)
        invalid = channel == 0
        result = blurred.copy()
        result[invalid] = 0
        out[:, :, 3] = result
        return out
    return apply


def spatial_shuffle(seed: int = 20261011) -> Callable[[np.ndarray], np.ndarray]:
    def apply(image: np.ndarray) -> np.ndarray:
        out = image.copy()
        channel = out[:, :, 3]
        rng = np.random.default_rng(seed)
        flat = channel.ravel()
        out[:, :, 3] = flat[rng.permutation(flat.size)].reshape(channel.shape)
        return out
    return apply


def mirror(axis: int) -> Callable[[np.ndarray], np.ndarray]:
    def apply(image: np.ndarray) -> np.ndarray:
        out = image.copy()
        channel = out[:, :, 3]
        out[:, :, 3] = cv2.flip(channel, axis)
        return out
    return apply


def roll(pixels: int) -> Callable[[np.ndarray], np.ndarray]:
    def apply(image: np.ndarray) -> np.ndarray:
        out = image.copy()
        channel = out[:, :, 3]
        out[:, :, 3] = np.roll(channel, pixels, axis=1)
        return out
    return apply


def sobel_edges(image: np.ndarray) -> np.ndarray:
    out = image.copy()
    channel = out[:, :, 3].astype(np.float32)
    gx = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    magnitude = np.clip(magnitude, 0, 255)
    magnitude[image[:, :, 3] == 0] = 0
    out[:, :, 3] = magnitude.astype(np.uint8)
    return out


def add_noise(sigma: float = 8.0, seed: int = 20261011) -> Callable[[np.ndarray], np.ndarray]:
    def apply(image: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(seed)
        out = image.copy()
        channel = out[:, :, 3].astype(np.float32)
        noisy = channel + rng.normal(0, sigma, channel.shape)
        noisy = np.clip(noisy, 0, 255)
        noisy[image[:, :, 3] == 0] = 0
        out[:, :, 3] = noisy.astype(np.uint8)
        return out
    return apply


def quantize(levels: int = 4) -> Callable[[np.ndarray], np.ndarray]:
    def apply(image: np.ndarray) -> np.ndarray:
        out = image.copy()
        channel = out[:, :, 3]
        mask = channel > 0
        step = 255.0 / (levels - 1)
        quantized = np.round(channel[mask].astype(np.float32) / step) * step
        channel[mask] = np.clip(quantized, 1, 255).astype(np.uint8)
        return out
    return apply


def gamma(power: float) -> Callable[[np.ndarray], np.ndarray]:
    def apply(image: np.ndarray) -> np.ndarray:
        out = image.copy()
        channel = out[:, :, 3]
        mask = channel > 0
        scaled = (channel[mask].astype(np.float32) / 255.0) ** power * 255.0
        channel[mask] = np.clip(np.round(scaled), 1, 255).astype(np.uint8)
        return out
    return apply


def replace_with(channel_index: int | str) -> Callable[[np.ndarray], np.ndarray]:
    def apply(image: np.ndarray) -> np.ndarray:
        out = image.copy()
        rgb = image[:, :, :3]
        if channel_index == "gray":
            replacement = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        elif channel_index == "mean":
            replacement = rgb.astype(np.float32).mean(axis=2).astype(np.uint8)
        else:
            replacement = rgb[:, :, int(channel_index)]
        out[:, :, 3] = replacement
        return out
    return apply


def rgb_shift(delta: int) -> Callable[[np.ndarray], np.ndarray]:
    def apply(image: np.ndarray) -> np.ndarray:
        out = image.copy()
        rgb = out[:, :, :3].astype(np.int16)
        out[:, :, :3] = np.clip(rgb + delta, 0, 255).astype(np.uint8)
        return out
    return apply


def rgb_invert(image: np.ndarray) -> np.ndarray:
    out = image.copy()
    out[:, :, :3] = 255 - out[:, :, :3]
    return out


def identity(image: np.ndarray) -> np.ndarray:
    return image


# ---------------------------------------------------------------- 条件表
def build_conditions() -> list[Condition]:
    items: list[Condition] = [
        Condition("baseline", "原始输入（基准）", identity),
        Condition("depth_invert", "深度整体反转：255−原值（无标签、可部署）", invert),
        Condition("depth_invert_shift+102", "先反转再抬电平", lambda im: shift_by(102)(invert(im))),
        Condition("depth_shift+102", "深度数值 +102（≈+300mm）", shift_by(102)),
        Condition("depth_shift+51", "深度数值 +51（≈+150mm）", shift_by(51)),
        Condition("depth_shift-51", "深度数值 −51（≈−150mm）", shift_by(-51)),
        Condition("depth_zero", "深度清零（无效语义）", constant(0)),
        Condition("depth_const1", "有效区全填 1（最远）", constant(1)),
        Condition("depth_const128", "有效区全填 128", constant(128)),
        Condition("depth_const255", "有效区全填 255（最近？编码上最远）", constant(255)),
        Condition("depth_flat_mean", "有效区全填该图均值", flat_valid_mean),
        Condition("depth_blur31", "深度高斯模糊 31", blur(31)),
        Condition("depth_blur15", "深度高斯模糊 15", blur(15)),
        Condition("depth_shuffle", "深度像素整体打乱", spatial_shuffle()),
        Condition("depth_mirror_lr", "深度左右镜像", mirror(1)),
        Condition("depth_mirror_ud", "深度上下镜像", mirror(0)),
        Condition("depth_roll16", "深度右移 16 像素", roll(16)),
        Condition("depth_roll64", "深度右移 64 像素", roll(64)),
        Condition("depth_edges", "只留深度边缘（Sobel）", sobel_edges),
        Condition("depth_noise8", "深度加高斯噪声（σ=8）", add_noise(8.0)),
        Condition("depth_quant4", "深度只留 4 个等级", quantize(4)),
        Condition("depth_gamma0.5", "深度提亮（γ=0.5）", gamma(0.5)),
        Condition("depth_gamma2.0", "深度压暗（γ=2.0）", gamma(2.0)),
        Condition("depth_as_gray", "第 4 通道换成原图黑白版", replace_with("gray")),
        Condition("depth_as_rgbmean", "第 4 通道换成 RGB 平均", replace_with("mean")),
        Condition("rgb_invert", "RGB 三通道反转（深度不变）", rgb_invert),
        Condition("rgb_shift+51", "RGB 三通道 +51（深度不变）", rgb_shift(51)),
        Condition("l1x0.5_only", "第 1 层输出 ×0.5（不改输入）", identity, 0.5),
        Condition("l1x0.5+depth_invert", "第 1 层 ×0.5 + 深度反转", invert, 0.5),
        Condition("l1x0.5+depth_zero", "第 1 层 ×0.5 + 深度清零", constant(0), 0.5),
        Condition("l1x0.5+depth_as_gray", "第 1 层 ×0.5 + 深度换黑白", replace_with("gray"), 0.5),
    ]
    return items


CORE = {
    "baseline", "depth_invert", "depth_shift+102", "depth_zero", "depth_flat_mean",
    "depth_shuffle", "depth_as_gray", "rgb_invert", "l1x0.5_only", "l1x0.5+depth_invert",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="深度敏感性测试矩阵")
    parser.add_argument("--dataset-root", default=str(ac._DEFAULT_DATASET))
    parser.add_argument("--weights-dir", default=str(ac._REPO / "train_weight"))
    parser.add_argument("--model", default="yolo11n_rgbd_4ch")
    parser.add_argument("--split", default="test")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--expected-count", type=int, default=100)
    parser.add_argument("--preset", choices=["all", "core"], default="all")
    parser.add_argument("--only", nargs="+", default=None, help="只跑这些条件名")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root).expanduser()
    weights_dir = Path(args.weights_dir).expanduser()
    weight = weights_dir / f"{args.model}.pt"
    if not weight.is_file():
        raise FileNotFoundError(weight)
    out_dir = Path(args.output_dir) if args.output_dir else ac._REPO / "runs" / "depth_battery" / args.model
    out_dir = Path(out_dir).expanduser()

    rgb = ac.stem_map(root / "rgb" / "images" / args.split)
    rgbd = ac.stem_map(root / "rgbd" / "images" / args.split)
    label_dir = root / "rgb" / "labels" / args.split
    if not label_dir.is_dir():
        label_dir = root / "rgbd" / "labels" / args.split
    stems = sorted(s for s in rgb.keys() & rgbd.keys() if (label_dir / f"{s}.txt").is_file())
    if args.expected_count and len(stems) != args.expected_count:
        raise RuntimeError(f"期望 {args.expected_count} 张，实际 {len(stems)} 张")
    base_items = load_items(root, args.split, 4, stems)

    conditions = build_conditions()
    if args.only:
        wanted = set(args.only)
        conditions = [c for c in conditions if c.name in wanted]
    elif args.preset == "core":
        conditions = [c for c in conditions if c.name in CORE]

    print("=" * 108)
    print(f"深度敏感性测试矩阵  模型={args.model}  图片={len(stems)}  conf={args.conf}")
    print(f"条件数={len(conditions)}")
    print("=" * 108)

    summary: list[dict[str, Any]] = []
    current_keep = None
    model = None
    for spec in conditions:
        if spec.l1_keep != current_keep or model is None:
            if model is not None:
                del model
            model = ac.YOLO(str(weight))
            if spec.l1_keep != 1.0:
                model.model.model[1] = LayerAttenuator(model.model.model[1], spec.l1_keep)
            current_keep = spec.l1_keep
        items = [(stem, spec.transform(image), gt) for stem, image, gt in base_items]
        started = time.perf_counter()
        rows = ac.run_condition(model, items, args.conf, args.imgsz, args.match_iou)
        hits = [r for r in rows if r["nme"] is not None]
        entry = {
            "condition": spec.name,
            "description": spec.description,
            "l1_keep": spec.l1_keep,
            "frames": len(rows),
            "hits": len(hits),
            "nme": ac.mean_of([r["nme"] for r in hits]),
            "box_iou": ac.mean_of([r["iou"] for r in hits]),
            "seconds": round(time.perf_counter() - started, 2),
        }
        summary.append(entry)
        print(f"{spec.name:<26} hits {entry['hits']:>3}/{len(rows)} "
              f"NME={ac.fmt(entry['nme'])} IoU={ac.fmt(entry['box_iou'], 3)}")
    del model

    base = next((r for r in summary if r["condition"] == "baseline"), None)
    for row in summary:
        if base and base["nme"] and row["nme"]:
            row["delta_pct"] = round((row["nme"] / base["nme"] - 1) * 100, 2)
        else:
            row["delta_pct"] = None

    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["condition", "description", "l1_keep", "frames", "hits", "nme",
              "delta_pct", "box_iou", "seconds"]
    with (out_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)
    (out_dir / "run_config.json").write_text(json.dumps({
        "dataset_root": str(root), "split": args.split, "stems": len(stems),
        "model": args.model, "conf": args.conf, "imgsz": args.imgsz,
        "match_iou": args.match_iou, "preset": args.preset,
        "conditions": [c.name for c in conditions],
        "ultralytics": ac._LOCKED_VERSION, "eval_impl": ac._EVAL_SCRIPT.name,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n按效果排序（负数 = 误差变小 = 变好）")
    ordered = sorted([r for r in summary if r["delta_pct"] is not None], key=lambda r: r["delta_pct"])
    print(f"{'变化':>8}  {'命中':>6}  {'NME':>9}  条件")
    for row in ordered:
        print(f"{row['delta_pct']:>+7.2f}%  {row['hits']:>6}  {ac.fmt(row['nme']):>9}  "
              f"{row['condition']:<26} {row['description']}")
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
