r"""第 1 层输出幅度剖面：3 通道模型 vs 4 通道模型，以及深度电平改变后的变化。

只读激活，不改权重、不改结构。第 1 层输出就是第 2 层的输入，
它的幅度决定后面所有层的信号尺度。

统计量（每张图取一次，再对 100 张取均值）：
    abs_mean      32 个通道激活绝对值的平均（整体"有多亮"）
    ch_abs_mean   每个通道自己的绝对值平均
    ch_spread     32 个通道之间 abs_mean 的离散度（越大越"挑通道"）

用法：
    .venv\Scripts\python.exe l1_energy_profile.py
    .venv\Scripts\python.exe l1_energy_profile.py --variants true_depth shift+102
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

import ablate_c2psa as ac
from ablate_layers import load_items
from depth_usage_probe import apply_variant
from l1_channel_attribution import L1Recorder, resolve_active_layer

LAYER_INDEX = 1
DEFAULT_MODELS = ["yolo11n_rgb_3ch", "yolo11n_rgbd_4ch"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="第 1 层输出幅度剖面")
    parser.add_argument("--dataset-root", default=str(ac._DEFAULT_DATASET))
    parser.add_argument("--weights-dir", default=str(ac._REPO / "train_weight"))
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--variants", nargs="+", default=["true_depth", "shift+102"],
                        help="只有 4 通道模型会应用深度变体")
    parser.add_argument("--split", default="test")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--expected-count", type=int, default=100)
    parser.add_argument("--output-dir", default=str(ac._REPO / "runs" / "l1_energy_profile"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root).expanduser()
    weights_dir = Path(args.weights_dir).expanduser()
    out_dir = Path(args.output_dir).expanduser()

    stems = ac.stem_map(root / "rgbd" / "images" / args.split)
    label_dir = root / "rgb" / "labels" / args.split
    if not label_dir.is_dir():
        label_dir = root / "rgbd" / "labels" / args.split
    stems = sorted(s for s in stems if (label_dir / f"{s}.txt").is_file())
    if args.expected_count and len(stems) != args.expected_count:
        raise RuntimeError(f"期望 {args.expected_count} 张，实际 {len(stems)} 张")
    items_4ch = load_items(root, args.split, 4, stems)
    items_3ch = load_items(root, args.split, 3, stems)

    print("=" * 100)
    print("第 1 层输出幅度剖面")
    print(f"数据: {root}  split={args.split}  图片数={len(stems)}")
    print("=" * 100)

    summary: list[dict[str, Any]] = []
    per_channel: list[dict[str, Any]] = []
    for model_name in args.models:
        weight = weights_dir / f"{model_name}.pt"
        if not weight.is_file():
            raise FileNotFoundError(weight)
        model = ac.YOLO(str(weight))
        channels_in = int(model.model.model[0].conv.weight.shape[1])
        model.model.model[LAYER_INDEX] = L1Recorder(model.model.model[LAYER_INDEX])
        variants = args.variants if channels_in == 4 else ["true_depth"]
        items = items_4ch if channels_in == 4 else items_3ch
        for variant in variants:
            if variant == "true_depth":
                probe_items = items
            else:
                # shift+N 由 apply_variant 的动态分支解析，第三个参数这里不起作用
                probe_items = [(stem, apply_variant(image, variant, 0), gt)
                               for stem, image, gt in items]
            abs_means: list[np.ndarray] = []
            overall: list[float] = []
            for stem, image, gt in probe_items:
                ac.ev.run_pose_model(model, image, args.conf, args.imgsz)
                recorder = resolve_active_layer(model)
                if recorder is None or recorder.last is None:
                    raise RuntimeError(f"{model_name}: 没有读到第 1 层激活")
                activation = np.abs(recorder.last.astype(np.float32))
                channel_mean = activation.mean(axis=(1, 2))
                abs_means.append(channel_mean)
                overall.append(float(channel_mean.mean()))
            stacked = np.stack(abs_means)             # (N, C)
            mean_per_channel = stacked.mean(axis=0)
            entry = {
                "model": model_name,
                "channels_in": channels_in,
                "variant": variant,
                "l1_channels": int(stacked.shape[1]),
                "abs_mean": float(np.mean(overall)),
                "per_image_std": float(np.std(overall)),
                "channel_mean_spread": float(mean_per_channel.std()),
                "top_channel_ratio": float(mean_per_channel.max() / max(mean_per_channel.mean(), 1e-9)),
            }
            summary.append(entry)
            for index, value in enumerate(mean_per_channel, start=1):
                per_channel.append({"model": model_name, "variant": variant,
                                    "channel_1based": index, "abs_mean": round(float(value), 6)})
            print(f"{model_name:<22}{variant:<12} 输入{channels_in}ch  "
                  f"L1 输出 {entry['l1_channels']} 通道  abs_mean={entry['abs_mean']:.4f}  "
                  f"通道间离散={entry['channel_mean_spread']:.4f}")
        del model

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with (out_dir / "per_channel.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_channel[0]))
        writer.writeheader()
        writer.writerows(per_channel)
    (out_dir / "run_config.json").write_text(json.dumps({
        "dataset_root": str(root), "split": args.split, "stems": len(stems),
        "models": args.models, "variants": args.variants,
        "layer_index": LAYER_INDEX, "conf": args.conf, "imgsz": args.imgsz,
        "ultralytics": ac._LOCKED_VERSION, "eval_impl": ac._EVAL_SCRIPT.name,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n对比")
    base = next((r for r in summary if r["model"] == "yolo11n_rgb_3ch"), None)
    for row in summary:
        delta = None
        if base:
            delta = (row["abs_mean"] / base["abs_mean"] - 1) * 100
        print(f"{row['model']:<22}{row['variant']:<12} abs_mean={row['abs_mean']:.4f}  "
              f"相对 3 通道 {ac.fmt(delta, 1):>7}%")
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
