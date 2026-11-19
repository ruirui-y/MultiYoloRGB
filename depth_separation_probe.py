r"""深度"分离信息"探针：把手/背景压成两个常数，彻底去掉深度值。

三个变体（都只改第 4 通道，权重与结构不动）：
    gt_near      用 GT 框做掩膜：框内=255、框外=1、无效=0   ← 用了标签，只作诊断上限
    gt_far       反过来：框内=1、框外=255                   ← 方向对照
    pred_near    用第一趟推理出的最高分框做掩膜（两趟推理，不用标签，可部署）

判据：
    如果 gt_near / pred_near 明显变好 → 模型只需要"手在哪"这一层分离，
    精细深度值是噪声（"分离假说"成立）
    如果不变或变差 → 之前的 flat_valid 测量或者分离假说有误会

用法：
    .venv\Scripts\python.exe depth_separation_probe.py
    .venv\Scripts\python.exe depth_separation_probe.py --models yolo11n_rgbd_4ch
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import ablate_c2psa as ac
from ablate_layers import load_items

VARIANTS = ["true_depth", "gt_near", "gt_far", "pred_near",
            "swap_means", "hand_high_only", "bg_low_only"]
DEFAULT_MODELS = ["yolo11n_rgbd_4ch", "yolov8n_rgbd_4ch"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="深度分离信息探针")
    parser.add_argument("--dataset-root", default=str(ac._DEFAULT_DATASET))
    parser.add_argument("--weights-dir", default=str(ac._REPO / "train_weight"))
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    parser.add_argument("--split", default="test")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--expected-count", type=int, default=100)
    parser.add_argument("--output-dir", default=str(ac._REPO / "runs" / "depth_separation_probe"))
    return parser.parse_args()


def box_mask(shape: tuple[int, int], box: np.ndarray) -> np.ndarray:
    """把 (x1,y1,x2,y2) 像素框画成布尔掩膜。"""
    height, width = shape
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    x1, x2 = sorted((int(round(x1)), int(round(x2))))
    y1, y2 = sorted((int(round(y1)), int(round(y2))))
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    mask = np.zeros((height, width), dtype=bool)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = True
    return mask


def two_level_depth(image: np.ndarray, mask: np.ndarray, inside: int, outside: int) -> np.ndarray:
    """框内/框外各填一个常数；原本无效（0）的像素保持 0，避免改变无效语义。"""
    out = image.copy()
    channel = out[:, :, 3]
    valid = channel > 0
    channel[valid & mask] = inside
    channel[valid & ~mask] = outside
    return out


def region_average(channel: np.ndarray, mask: np.ndarray, valid: np.ndarray) -> float | None:
    """算某个区域里有效像素的平均值；没有有效像素就返回 None。"""
    selected = channel[valid & mask]
    if selected.size == 0:
        return None
    return float(selected.mean())


def swap_region_means(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """把手上和背景的平均值对调：高低差不变，只把方向反过来。"""
    out = image.copy()
    channel = out[:, :, 3]
    valid = channel > 0
    hand_mean = region_average(channel, mask, valid)
    background_mean = region_average(channel, ~mask, valid)
    if hand_mean is None or background_mean is None:
        return out
    channel[valid & mask] = int(round(background_mean))
    channel[valid & ~mask] = int(round(hand_mean))
    return out


def hand_high_only(image: np.ndarray, mask: np.ndarray, top: int = 254) -> np.ndarray:
    """只把手区域抬到最高，背景保持原值。"""
    out = image.copy()
    channel = out[:, :, 3]
    valid = channel > 0
    channel[valid & mask] = top
    return out


def background_low_only(image: np.ndarray, mask: np.ndarray, low: int = 1) -> np.ndarray:
    """只把背景压到最低，手保持原值。"""
    out = image.copy()
    channel = out[:, :, 3]
    valid = channel > 0
    channel[valid & ~mask] = low
    return out


def predicted_box(model: Any, image: np.ndarray, conf: float, imgsz: int) -> np.ndarray | None:
    """第一趟推理：取置信度最高的那个框，用于第二趟的掩膜。"""
    result = ac.ev.run_pose_model(model, image, conf, imgsz)
    if result.boxes is None or len(result.boxes) == 0:
        return None
    boxes = result.boxes
    best = int(np.argmax(boxes.conf.cpu().numpy()))
    return boxes.xyxy.cpu().numpy()[best]


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

    rgb = ac.stem_map(root / "rgb" / "images" / args.split)
    rgbd = ac.stem_map(root / "rgbd" / "images" / args.split)
    label_dir = root / "rgb" / "labels" / args.split
    if not label_dir.is_dir():
        label_dir = root / "rgbd" / "labels" / args.split
    stems = sorted(s for s in rgb.keys() & rgbd.keys() if (label_dir / f"{s}.txt").is_file())
    if args.expected_count and len(stems) != args.expected_count:
        raise RuntimeError(f"期望 {args.expected_count} 张，实际 {len(stems)} 张")
    items = load_items(root, args.split, 4, stems)

    print("=" * 100)
    print("深度分离信息探针：框内/框外各填一个常数")
    print(f"数据: {root}  split={args.split}  图片数={len(stems)}")
    print(f"口径: conf={args.conf} imgsz={args.imgsz} match_iou={args.match_iou}")
    print("=" * 100)

    summary: list[dict[str, Any]] = []
    for model_name in args.models:
        weight = weights_dir / f"{model_name}.pt"
        if not weight.is_file():
            raise FileNotFoundError(weight)
        for variant in args.variants:
            model = ac.YOLO(str(weight))
            channels = int(model.model.model[0].conv.weight.shape[1])
            if channels != 4:
                raise RuntimeError(f"{model_name} 不是 4 通道模型")
            started = time.perf_counter()
            missing = 0
            probe_items = []
            for stem, image, gt in items:
                if variant == "true_depth":
                    probe_items.append((stem, image, gt))
                    continue
                if variant == "pred_near":
                    box = predicted_box(model, image, args.conf, args.imgsz)
                    if box is None:
                        missing += 1
                        probe_items.append((stem, image, gt))
                        continue
                else:
                    box = np.asarray(gt["box"], dtype=np.float64)
                mask = box_mask(image.shape[:2], box)
                if variant == "gt_far":
                    transformed = two_level_depth(image, mask, inside=1, outside=255)
                elif variant == "swap_means":
                    transformed = swap_region_means(image, mask)
                elif variant == "hand_high_only":
                    transformed = hand_high_only(image, mask)
                elif variant == "bg_low_only":
                    transformed = background_low_only(image, mask)
                else:
                    transformed = two_level_depth(image, mask, inside=255, outside=1)
                probe_items.append((stem, transformed, gt))
            try:
                rows = ac.run_condition(model, probe_items, args.conf, args.imgsz, args.match_iou)
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
                "missing_pred_box": missing,
                "elapsed_s": round(time.perf_counter() - started, 2),
            }
            for k in range(ac.KEYPOINT_COUNT):
                entry[f"k{k}_px"] = ac.mean_of([r[f"k{k}_px"] for r in hits])
            summary.append(entry)
            print(f"{model_name:<22}{variant:<12} hits {entry['hits']:>3}/{len(rows)} "
                  f"NME={ac.fmt(entry['nme'])} IoU={ac.fmt(entry['box_iou'], 3)}"
                  f"{'  无预测框=' + str(missing) if missing else ''}")

    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["model", "variant", "frames", "hits", "nme", "box_iou",
              "missing_pred_box", "elapsed_s"] + [f"k{k}_px" for k in range(ac.KEYPOINT_COUNT)]
    write_csv(out_dir / "summary.csv", summary, fields)
    (out_dir / "run_config.json").write_text(json.dumps({
        "dataset_root": str(root), "split": args.split, "stems": len(stems),
        "models": args.models, "variants": args.variants,
        "conf": args.conf, "imgsz": args.imgsz, "match_iou": args.match_iou,
        "ultralytics": ac._LOCKED_VERSION, "eval_impl": ac._EVAL_SCRIPT.name,
        "note": "gt_near/gt_far 使用 GT 框，仅作诊断；pred_near 用第一趟推理框，不用标签",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n相对原始深度的变化")
    print(f"{'model':<22}{'variant':<12}{'hits':>6}{'NME':>10}{'Δ%':>9}")
    for model_name in args.models:
        base = next((r for r in summary if r["model"] == model_name and r["variant"] == "true_depth"), None)
        for row in [r for r in summary if r["model"] == model_name]:
            delta = (row["nme"] / base["nme"] - 1) * 100 if base and row["nme"] else None
            print(f"{row['model']:<22}{row['variant']:<12}{row['hits']:>6}"
                  f"{ac.fmt(row['nme']):>10}{ac.fmt(delta, 2):>9}")
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
