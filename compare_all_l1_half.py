r"""Unified 5-model YOLO Pose comparison: original vs layer-index-1 output * keep.

Place next to ablate_c2psa.py and ablate_layers.py, then run:
    .venv\Scripts\python.exe compare_all_l1_half.py

The intervention is OUTPUT SCALING of top-level module index 1 (Conv), not
weight editing and not module deletion. Each condition loads fresh weights.
All predictions, matching and NME reuse ablate_c2psa.py's evaluation path.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Any

import ablate_c2psa as ac
from ablate_layers import LayerAttenuator, load_items

MODEL_SPECS: tuple[tuple[str, str, int], ...] = (
    ("yolo11n_rgb_3ch", "YOLO11-RGB", 3),
    ("yolo11n_rgbd_4ch", "YOLO11-RGBD", 4),
    ("yolo26n_rgb_3ch", "YOLO26-RGB", 3),
    ("yolov8n_rgb_3ch", "YOLOv8-RGB", 3),
    ("yolov8n_rgbd_4ch", "YOLOv8-RGBD", 4),
)
LAYER_INDEX = 1  # Matches the user's previous L1 experiments, NOT layer index 0.
N_KPTS = ac.KEYPOINT_COUNT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Same 100 images, same evaluator: all five models original vs L1 output attenuation"
    )
    parser.add_argument("--dataset-root", default=str(ac._DEFAULT_DATASET))
    parser.add_argument("--weights-dir", default=str(ac._REPO / "train_weight"))
    parser.add_argument("--output-dir", default=str(ac._REPO / "runs" / "compare_all_l1_half"))
    parser.add_argument("--keep", type=float, default=0.50)
    parser.add_argument("--split", default="test")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--expected-count", type=int, default=100,
                        help="Fail if common RGB/RGBD test set size differs (0 disables check)")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def matched_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {r["image"]: r for r in rows if r["nme"] is not None}


def keypoint_means(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    return {f"k{i}_px": ac.mean_of([r[f"k{i}_px"] for r in rows])
            for i in range(N_KPTS)}


def apply_l1_scale(model: Any, keep: float) -> str:
    modules = model.model.model
    if len(modules) <= LAYER_INDEX:
        raise RuntimeError(f"Model contains only {len(modules)} modules; no layer index 1")
    original = modules[LAYER_INDEX]
    module_type = type(original).__name__
    if module_type != "Conv":
        raise RuntimeError(
            f"Expected index {LAYER_INDEX} to be Conv, got {module_type}. "
            "Stopped rather than modifying an unintended layer."
        )
    modules[LAYER_INDEX] = LayerAttenuator(original, keep)
    assert modules[LAYER_INDEX].keep == keep
    return module_type


def run() -> None:
    args = parse_args()
    if not 0 <= args.keep <= 1:
        raise ValueError("--keep must be in [0,1]")
    root = Path(args.dataset_root).expanduser()
    weights_dir = Path(args.weights_dir).expanduser()
    out_dir = Path(args.output_dir).expanduser()

    rgb_map = ac.stem_map(root / "rgb" / "images" / args.split)
    rgbd_map = ac.stem_map(root / "rgbd" / "images" / args.split)
    label_dir = root / "rgb" / "labels" / args.split
    if not label_dir.is_dir():
        label_dir = root / "rgbd" / "labels" / args.split
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")
    stems = sorted(s for s in rgb_map.keys() & rgbd_map.keys()
                   if (label_dir / (s + ".txt")).is_file())
    if not stems:
        raise RuntimeError("No common RGB / RGBD image stems with labels")
    if args.expected_count and len(stems) != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count} shared frames but found {len(stems)}. "
            "Check dataset/split before comparing models."
        )
    for model_name, _label, _ch in MODEL_SPECS:
        path = weights_dir / (model_name + ".pt")
        if not path.is_file():
            raise FileNotFoundError(f"Missing weight: {path}")

    print("=" * 106)
    print("FIVE-MODEL UNIFIED EVALUATION | original vs L1 (module index=1, Conv) output attenuation")
    print(f"Dataset={root}, split={args.split}, frames={len(stems)}, "
          f"conf={args.conf}, imgsz={args.imgsz}, match_iou={args.match_iou}")
    print(f"Layer index={LAYER_INDEX}, keep={args.keep:.3f}; output multiplier, NOT weight editing")
    print("Fresh .pt weights for every condition; same image stems; same evaluation helper")
    print("=" * 106)

    tested: dict[tuple[str, str], list[dict[str, Any]]] = {}
    summary: list[dict[str, Any]] = []
    per_image: list[dict[str, Any]] = []
    weight_paths: dict[str, str] = {}
    condition_names = ("original", "L1_half")

    for model_name, label, channels in MODEL_SPECS:
        path = weights_dir / (model_name + ".pt")
        weight_paths[model_name] = str(path)
        items = load_items(root, args.split, channels, stems)
        print(f"\n{label} ({channels}ch)")
        for condition in condition_names:
            model = ac.YOLO(str(path))  # Fresh weights; change graph BEFORE predictor creation.
            module_type = type(model.model.model[LAYER_INDEX]).__name__
            if condition == "L1_half":
                module_type = apply_l1_scale(model, args.keep)
            try:
                rows = ac.run_condition(model, items, args.conf, args.imgsz, args.match_iou)
            finally:
                del model
                gc.collect()
            tested[(model_name, condition)] = rows
            hits = len(matched_map(rows))
            record: dict[str, Any] = {
                "model": model_name,
                "label": label,
                "channels": channels,
                "condition": condition,
                "layer_index": LAYER_INDEX,
                "layer_type": module_type,
                "keep": 1.0 if condition == "original" else args.keep,
                "total": len(rows),
                "hits": hits,
                "boxes_per_image": ac.mean_of([r["n_boxes"] for r in rows]),
                "nme_matched_only": ac.mean_of([r["nme"] for r in rows]),
            }
            record.update(keypoint_means(rows))
            summary.append(record)
            per_image.extend({"model": model_name, "label": label, "channels": channels,
                              "condition": condition, "keep": record["keep"], **row}
                             for row in rows)
            print(f"  {condition:<9} hits {hits:3d}/{len(rows)} "
                  f"NME(matched)={ac.fmt(record['nme_matched_only'])} "
                  f"mean_boxes={record['boxes_per_image']:.2f}")

        base = matched_map(tested[(model_name, "original")])
        half = matched_map(tested[(model_name, "L1_half")])
        common = sorted(base.keys() & half.keys())
        paired_base = ac.mean_of([base[s]["nme"] for s in common])
        paired_half = ac.mean_of([half[s]["nme"] for s in common])
        paired_pct = (100 * (paired_half - paired_base) / paired_base
                      if paired_base is not None and paired_half is not None and paired_base else None)
        for record in summary[-2:]:
            record.update({
                "paired_images": len(common),
                "paired_original_nme": paired_base,
                "paired_l1_half_nme": paired_half,
                "paired_l1_change_pct": paired_pct,
            })
        print(f"  PAIRED    {len(common):3d}/{len(stems)} "
              f"original={ac.fmt(paired_base)} L1={ac.fmt(paired_half)} "
              f"change={(f'{paired_pct:+.2f}%' if paired_pct is not None else '-')}")

    # A model ranking based on ALL 10 conditions' common matched image subset.
    # In the ideal case all conditions hit 100/100 and the subset is all 100 frames.
    maps = {key: matched_map(rows) for key, rows in tested.items()}
    common_all = sorted(set.intersection(*(set(d) for d in maps.values())))
    ranking_same_frames: list[dict[str, Any]] = []
    for model_name, label, channels in MODEL_SPECS:
        for condition in condition_names:
            rows = [maps[(model_name, condition)][s] for s in common_all]
            score = {
                "model": model_name, "label": label, "channels": channels,
                "condition": condition, "same_frames": len(common_all),
                "nme_same_frames": ac.mean_of([r["nme"] for r in rows]),
            }
            score.update(keypoint_means(rows))
            ranking_same_frames.append(score)
    ranking_same_frames.sort(key=lambda r: (
        r["nme_same_frames"] is None,
        r["nme_same_frames"] if r["nme_same_frames"] is not None else float("inf")
    ))

    # The primary research question: rank the five ATTENUATED models, not a
    # mixture of original and attenuated models. If hits differ, compare only
    # the intersection of the five attenuated models' matched frames.
    common_adjusted = sorted(set.intersection(*(
        set(maps[(model_name, "L1_half")]) for model_name, _label, _ch in MODEL_SPECS
    )))
    ranking_adjusted_same_frames: list[dict[str, Any]] = []
    for model_name, label, channels in MODEL_SPECS:
        rows = [maps[(model_name, "L1_half")][s] for s in common_adjusted]
        result = {"model": model_name, "label": label, "channels": channels,
                  "condition": "L1_half", "same_frames": len(common_adjusted),
                  "nme_same_frames": ac.mean_of([r["nme"] for r in rows])}
        result.update(keypoint_means(rows))
        ranking_adjusted_same_frames.append(result)
    ranking_adjusted_same_frames.sort(key=lambda r: (
        r["nme_same_frames"] is None,
        r["nme_same_frames"] if r["nme_same_frames"] is not None else float("inf")
    ))
    adjusted_full_hit = sorted(
        (r for r in summary if r["condition"] == "L1_half" and r["hits"] == len(stems)),
        key=lambda r: r["nme_matched_only"]
    )

    full_hit = sorted((r for r in summary if r["hits"] == len(stems)),
                      key=lambda r: r["nme_matched_only"])
    print("\n" + "=" * 106)
    print(f"PRIMARY RANKING: adjusted models with FULL {len(stems)}/{len(stems)} hits")
    for rank, row in enumerate(adjusted_full_hit, 1):
        print(f"{rank:>2}. {row['label']:<15} NME={ac.fmt(row['nme_matched_only'])} "
              f"matched={row['hits']}/{row['total']}")
    print(f"\nAdjusted-only matched intersection: {len(common_adjusted)}/{len(stems)} frames")
    for rank, row in enumerate(ranking_adjusted_same_frames, 1):
        print(f"{rank:>2}. {row['label']:<15} NME={ac.fmt(row['nme_same_frames'])}")
    print("\nK0..K6 mean pixel errors (adjusted models, each model's own matched frames):")
    print(f"{'Model':<16} {'hits':>6} " + " ".join(f"K{i:>3}" for i in range(N_KPTS)))
    for model_name, label, channels in MODEL_SPECS:
        r = next(r for r in summary if r['model'] == model_name and r['condition'] == 'L1_half')
        cells = " ".join(f"{r[f'k{i}_px']:>4.1f}" if r[f'k{i}_px'] is not None else "   - "
                         for i in range(N_KPTS))
        print(f"{label:<16} {r['hits']:>3}/{r['total']:<3} {cells}")

    print("\n" + "=" * 106)
    print(f"RANKING: original + adjusted configurations with FULL {len(stems)}-frame hits")
    for rank, row in enumerate(full_hit, 1):
        print(f"{rank:>2}. {row['label']:<15} {row['condition']:<9} "
              f"NME={ac.fmt(row['nme_matched_only'])} matched={row['hits']}/{row['total']}")
    if not full_hit:
        print("  None matched every frame")

    print("\n" + "=" * 106)
    print(f"RANKING: exact SAME {len(common_all)} frames matched by ALL 10 conditions")
    if not common_all:
        print("  No common matches across all conditions; cannot rank by shared NME")
    for rank, row in enumerate(ranking_same_frames, 1):
        print(f"{rank:>2}. {row['label']:<15} {row['condition']:<9} "
              f"NME={ac.fmt(row['nme_same_frames'])}")

    out_dir.mkdir(parents=True, exist_ok=True)
    sum_fields = ["model", "label", "channels", "condition", "layer_index",
                  "layer_type", "keep", "total", "hits", "boxes_per_image",
                  "nme_matched_only", "paired_images", "paired_original_nme",
                  "paired_l1_half_nme", "paired_l1_change_pct"] + \
                 [f"k{i}_px" for i in range(N_KPTS)]
    write_csv(out_dir / "summary.csv", summary, sum_fields)
    per_fields = ["model", "label", "channels", "condition", "keep", "image",
                  "n_boxes", "matched", "iou", "nme"] + [f"k{i}_px" for i in range(N_KPTS)]
    write_csv(out_dir / "per_image.csv", per_image, per_fields)
    same_fields = ["model", "label", "channels", "condition", "same_frames",
                   "nme_same_frames"] + [f"k{i}_px" for i in range(N_KPTS)]
    write_csv(out_dir / "ranking_same_frames.csv", ranking_same_frames, same_fields)
    write_csv(out_dir / "ranking_full_hits.csv", full_hit, sum_fields)
    write_csv(out_dir / "ranking_adjusted_full_hits.csv", adjusted_full_hit, sum_fields)
    write_csv(out_dir / "ranking_adjusted_same_frames.csv", ranking_adjusted_same_frames, same_fields)
    (out_dir / "run_config.json").write_text(
        json.dumps({
            "dataset_root": str(root), "split": args.split, "stems": len(stems),
            "models": weight_paths, "layer_index": LAYER_INDEX,
            "intervention": "module_forward_output_multiplication",
            "keep": args.keep, "conditions": list(condition_names),
            "conf": args.conf, "imgsz": args.imgsz, "match_iou": args.match_iou,
            "ultralytics": ac._LOCKED_VERSION, "evaluator": ac._EVAL_SCRIPT.name,
            "common_all_condition_frames": len(common_all),
            "common_adjusted_model_frames": len(common_adjusted),
            "note": "Weights are NOT modified; L1=module index 1; "
                    "do not compare NME with different matched-image denominators.",
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\nSaved:", out_dir)
    print("summary.csv | per_image.csv | ranking_adjusted_full_hits.csv | "
          "ranking_adjusted_same_frames.csv | ranking_full_hits.csv | "
          "ranking_same_frames.csv | run_config.json")


if __name__ == "__main__":
    run()
