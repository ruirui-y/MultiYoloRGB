r"""第 1 层第 4 个输入通道的权重缩放实验。

目的：检验"深度只通过第 1 层第 4 个输入通道起作用"这个说法。
做法：直接改第 1 层卷积权重里对应第 4 个输入通道的那一块（weight[:, 3]），
      其它权重、BN、结构都不动。然后把"原始深度"和"反转深度"各跑一遍。

预算出的预测：
    权重置零 → 深度怎么改都不影响结果（两种深度结果必须完全相同）
    权重放大 → 反转带来的收益应该更大
    权重缩小 → 反转带来的收益应该更小

用法：
    .venv\Scripts\python.exe depth_weight_probe.py
    .venv\Scripts\python.exe depth_weight_probe.py --scales 0 0.5 1 2 --conf 0.01
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import ablate_c2psa as ac
from ablate_layers import load_items

LAYER_INDEX = 0     # 深度真正进入网络的是第 0 层（4→16 的卷积）；第 1 层是 16→32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="第 1 层第 4 输入通道权重缩放")
    parser.add_argument("--dataset-root", default=str(ac._DEFAULT_DATASET))
    parser.add_argument("--inverted-root",
                        default=r"H:\YJJ\Yolo_RGBD\Resource\session1_200146\dataset\hand_common_inv")
    parser.add_argument("--weights-dir", default=str(ac._REPO / "train_weight"))
    parser.add_argument("--model", default="yolo11n_rgbd_4ch")
    parser.add_argument("--scales", nargs="+", type=float, default=[1.0, 0.0, 0.5, 2.0])
    parser.add_argument("--depths", nargs="+", default=["original", "inverted"],
                        choices=["original", "inverted"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--conf", type=float, default=0.01)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--expected-count", type=int, default=100)
    parser.add_argument("--output-dir", default=str(ac._REPO / "runs" / "depth_weight_probe"))
    return parser.parse_args()


def scale_depth_input_weight(model: Any, scale: float) -> float:
    """把第 1 层卷积对应第 4 个输入通道的权重整体乘 scale，返回缩放前的范数。"""
    layer = model.model.model[LAYER_INDEX]
    weight = layer.conv.weight
    if weight.shape[1] != 4:
        raise RuntimeError(f"第 {LAYER_INDEX} 层输入通道不是 4，而是 {weight.shape[1]}")
    before = float(weight.data[:, 3].norm())
    with torch.no_grad():
        weight.data[:, 3] *= scale
    after = float(weight.data[:, 3].norm())
    if scale > 0 and abs(after / before - scale) > 1e-4:
        raise RuntimeError(f"缩放异常: {after}/{before} != {scale}")
    return before


def main() -> None:
    args = parse_args()
    weight_path = Path(args.weights_dir) / f"{args.model}.pt"
    if not weight_path.is_file():
        raise FileNotFoundError(weight_path)
    out_dir = Path(args.output_dir).expanduser()

    root = Path(args.dataset_root)
    inv_root = Path(args.inverted_root)
    label_dir = root / "rgb" / "labels" / args.split
    if not label_dir.is_dir():
        label_dir = root / "rgbd" / "labels" / args.split
    stems = sorted(s for s in ac.stem_map(root / "rgbd" / "images" / args.split)
                   if (label_dir / f"{s}.txt").is_file())
    if args.expected_count and len(stems) != args.expected_count:
        raise RuntimeError(f"期望 {args.expected_count} 张，实际 {len(stems)} 张")

    items = {
        "original": load_items(root, args.split, 4, stems),
        "inverted": load_items(inv_root, args.split, 4, stems),
    }

    print("=" * 100)
    print(f"第 1 层第 4 输入通道权重缩放  模型={args.model}  图片={len(stems)}  conf={args.conf}")
    print(f"缩放系数={args.scales}   深度输入={args.depths}")
    print("=" * 100)

    summary: list[dict[str, Any]] = []
    for scale in args.scales:
        for depth in args.depths:
            model = ac.YOLO(str(weight_path))
            before_norm = scale_depth_input_weight(model, scale)
            started = time.perf_counter()
            try:
                rows = ac.run_condition(model, items[depth], args.conf, args.imgsz, args.match_iou)
            finally:
                del model
            hits = [r for r in rows if r["nme"] is not None]
            entry = {
                "model": args.model,
                "depth": depth,
                "scale": scale,
                "weight_norm_before": round(before_norm, 6),
                "weight_norm_after": round(before_norm * scale, 6),
                "frames": len(rows),
                "hits": len(hits),
                "nme": ac.mean_of([r["nme"] for r in hits]),
                "box_iou": ac.mean_of([r["iou"] for r in hits]),
                "seconds": round(time.perf_counter() - started, 2),
            }
            summary.append(entry)
            print(f"scale={scale:<4} depth={depth:<9} hits {entry['hits']:>3}/{len(rows)} "
                  f"NME={ac.fmt(entry['nme'])} IoU={ac.fmt(entry['box_iou'], 3)}")

    # 每个缩放档位下，反转相对原始的变化
    print("\n每个档位下：反转深度相对原始深度")
    print(f"{'scale':>6}  {'原始':>10}  {'反转':>10}  {'变化':>9}")
    for scale in args.scales:
        orig = next((r for r in summary if r["scale"] == scale and r["depth"] == "original"), None)
        inv = next((r for r in summary if r["scale"] == scale and r["depth"] == "inverted"), None)
        if not orig or not inv or not orig["nme"] or not inv["nme"]:
            continue
        delta = (inv["nme"] / orig["nme"] - 1) * 100
        print(f"{scale:>6}  {orig['nme']:>10.6f}  {inv['nme']:>10.6f}  {delta:>+8.2f}%")

    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["model", "depth", "scale", "weight_norm_before", "weight_norm_after",
              "frames", "hits", "nme", "box_iou", "seconds"]
    with (out_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)
    (out_dir / "run_config.json").write_text(json.dumps({
        "dataset_root": str(root), "inverted_root": str(inv_root), "split": args.split,
        "stems": len(stems), "model": args.model, "scales": args.scales,
        "depths": args.depths, "conf": args.conf, "imgsz": args.imgsz,
        "match_iou": args.match_iou, "layer_index": LAYER_INDEX,
        "intervention": "weight[:, 3] *= scale（只改第 1 层第 4 个输入通道）",
        "ultralytics": ac._LOCKED_VERSION, "eval_impl": ac._EVAL_SCRIPT.name,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
