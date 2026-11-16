r"""YOLO11-RGBD L1 detailed ablation around an identified output group.

The two experimental directions are intentionally separate:

(1) restore (necessity in a half-scaled context):
    all L1 channels keep=0.5, then restore selected G2 channels to 1.0.
(2) suppress (sufficiency in the original context):
    all L1 channels keep=1.0, then attenuate one whole group to 0.5.

For each condition, reload the original .pt, hold all other model layers fixed,
and evaluate against the same 100 test frames with the project's own evaluator.

Place alongside ablate_c2psa.py and ablate_layers.py.
Run from project root:
    .venv\Scripts\python.exe ablate_l1_g2_detail.py --groups 4 --target-group 2

Outputs in runs/ablate_l1_g2_detail:
    summary.csv, per_image.csv, paired_per_image.csv,
    same_frames_summary.csv, channel_profile.csv, run_config.json
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

import ablate_c2psa as ac
from ablate_layers import load_items
from ablate_l1_channel_groups import (
    channel_groups,
    check_l1_and_get_channels,
    changes_against,
    hits_map,
    fmt,
)

LAYER_INDEX = 1
KEYPOINTS = ac.KEYPOINT_COUNT


@dataclass(frozen=True)
class Case:
    name: str
    family: str
    channels_to_change: tuple[int, ...]
    base_keep: float
    change_keep: float


class L1ChannelSelectScaler(nn.Module):
    """Wrap an existing L1 Conv; scale its OUTPUT, never its weights.

    Preserve the graph metadata required by Ultralytics module traversal.
    """

    def __init__(self, source: nn.Module, channels: int, spec: Case):
        super().__init__()
        self.source = source
        self.f = source.f
        self.i = source.i
        self.type = "L1ChannelSelectScaler"
        self.channels = channels
        self.spec = spec
        scales = torch.full((channels,), spec.base_keep, dtype=torch.float32)
        for idx in spec.channels_to_change:
            if not 0 <= idx < channels:
                raise ValueError(f"Channel {idx} outside [0,{channels})")
            scales[idx] = spec.change_keep
        self.register_buffer("scales", scales.reshape(1, channels, 1, 1), persistent=False)
        self.ran = False

    def forward(self, x):
        y = self.source(x)
        if not isinstance(y, torch.Tensor) or y.ndim != 4 or y.shape[1] != self.channels:
            raise RuntimeError(f"L1 unexpected tensor: {getattr(y, 'shape', None)}")
        self.ran = True
        return y * self.scales.to(device=y.device, dtype=y.dtype)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO11-RGBD L1 G2 detailed channel sweep")
    parser.add_argument("--dataset-root", default=str(ac._DEFAULT_DATASET))
    parser.add_argument("--weights-dir", default=str(ac._REPO / "train_weight"))
    parser.add_argument("--weights-name", default="yolo11n_rgbd_4ch")
    parser.add_argument("--split", default="test")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--expected-count", type=int, default=100)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--target-group", type=int, default=2,
                        help="1-based group number to split into individual channels")
    parser.add_argument("--base-keep", type=float, default=0.5,
                        help="Attenuated value in all-half baseline")
    parser.add_argument("--restore-keep", type=float, default=1.0)
    parser.add_argument("--mode", choices=("both", "restore", "suppress"), default="both")
    parser.add_argument("--output-dir", default=str(ac._REPO / "runs" / "ablate_l1_g2_detail"))
    return parser.parse_args()


def build_cases(groups: list[tuple[int, int]], target: int,
                half: float, full: float, mode: str) -> list[Case]:
    if target < 0 or target >= len(groups):
        raise ValueError("target group index is out of range")
    lo, hi = groups[target]
    midpoint = lo + (hi - lo) // 2
    if lo == midpoint or midpoint == hi:
        raise ValueError("Target group must contain at least two channels")
    prefix = f"G{target + 1}"
    cases = [Case("original", "control", (), full, full),
             Case("L1_half", "control", (), half, half)]
    if mode in ("both", "restore"):
        cases.append(Case(f"restore_{prefix}", "restore", tuple(range(lo, hi)), half, full))
        cases.append(Case(f"restore_{prefix}_left", "restore", tuple(range(lo, midpoint)), half, full))
        cases.append(Case(f"restore_{prefix}_right", "restore", tuple(range(midpoint, hi)), half, full))
        for ch in range(lo, hi):
            cases.append(Case(f"restore_ch{ch:02d}", "restore", (ch,), half, full))
    if mode in ("both", "suppress"):
        for i, (begin, end) in enumerate(groups, 1):
            cases.append(Case(f"suppress_only_G{i}", "suppress", tuple(range(begin, end)), full, half))
    return cases


def write_csv(filename: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with filename.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_paired_rows(case: str, ref_case: str,
                     ref: list[dict[str, Any]], cur: list[dict[str, Any]]) -> list[dict[str, Any]]:
    before = {r["image"]: r for r in ref}
    after = {r["image"]: r for r in cur}
    if before.keys() != after.keys():
        raise RuntimeError("Image names differ between reference and case")
    output = []
    for img in sorted(before):
        a, b = before[img], after[img]
        paired = a["nme"] is not None and b["nme"] is not None
        row: dict[str, Any] = {
            "case": case, "reference": ref_case, "image": img,
            "paired": paired,
            "reference_matched": a["nme"] is not None,
            "case_matched": b["nme"] is not None,
            "reference_nme": a["nme"], "case_nme": b["nme"],
            "delta_nme": b["nme"] - a["nme"] if paired else None,
            "case_iou": b["iou"], "case_boxes": b["n_boxes"],
        }
        for k in range(KEYPOINTS):
            name = f"k{k}_px"
            row[f"reference_{name}"] = a[name]
            row[f"case_{name}"] = b[name]
            row[f"delta_{name}"] = (b[name] - a[name]
                                      if paired and b[name] is not None and a[name] is not None else None)
        output.append(row)
    return output


def main():
    args = parse_args()
    if args.groups < 2 or not 1 <= args.target_group <= args.groups:
        raise ValueError("Invalid --groups or --target-group")
    if not (0.0 <= args.base_keep < args.restore_keep <= 1.0):
        raise ValueError("Require 0 <= base-keep < restore-keep <= 1")
    root = Path(args.dataset_root)
    weight = Path(args.weights_dir) / (args.weights_name.removesuffix(".pt") + ".pt")
    out_dir = Path(args.output_dir)
    if not weight.is_file():
        raise FileNotFoundError(weight)

    rgb = ac.stem_map(root / "rgb" / "images" / args.split)
    rgbd = ac.stem_map(root / "rgbd" / "images" / args.split)
    label_dir = root / "rgb" / "labels" / args.split
    if not label_dir.is_dir():
        label_dir = root / "rgbd" / "labels" / args.split
    if not label_dir.is_dir():
        raise FileNotFoundError(label_dir)
    stems = sorted(s for s in rgb.keys() & rgbd.keys() if (label_dir / f"{s}.txt").is_file())
    if not stems or (args.expected_count and len(stems) != args.expected_count):
        raise RuntimeError(f"Expected {args.expected_count} aligned images, got {len(stems)}")
    items = load_items(root, args.split, 4, stems)

    probe = ac.YOLO(str(weight))
    channels = check_l1_and_get_channels(probe)
    groups = channel_groups(channels, args.groups)
    del probe
    gc.collect()
    cases = build_cases(groups, args.target_group - 1,
                        args.base_keep, args.restore_keep, args.mode)

    target_lo, target_hi = groups[args.target_group - 1]
    print("=" * 112)
    print("YOLO11-RGBD L1 CHANNEL DETAIL: RESTORE TARGET CHANNELS + SUPPRESS SINGLE GROUPS")
    print(f"weights: {weight}")
    print(f"test images: {len(stems)} | conf={args.conf} imgsz={args.imgsz} match_iou={args.match_iou}")
    print(f"L1 channels={channels}, groups={groups}, target G{args.target_group}=[{target_lo},{target_hi})")
    print(f"half={args.base_keep}, full={args.restore_keep}, mode={args.mode}, cases={len(cases)}")
    print("restore: all channels half, selected channels full; compare against L1_half")
    print("suppress: all channels full, selected group half; compare against original")
    print("=" * 112)

    all_rows: dict[str, list[dict[str, Any]]] = {}
    summary: list[dict[str, Any]] = []
    per_image: list[dict[str, Any]] = []
    for spec in cases:
        model = ac.YOLO(str(weight))
        scaler = None
        if spec.name != "original":
            original_layer = model.model.model[LAYER_INDEX]
            scaler = L1ChannelSelectScaler(original_layer, channels, spec)
            model.model.model[LAYER_INDEX] = scaler
        start = time.perf_counter()
        try:
            rows = ac.run_condition(model, items, args.conf, args.imgsz, args.match_iou)
        finally:
            del model
            gc.collect()
        all_rows[spec.name] = rows
        hits = len(hits_map(rows))
        result = {
            "case": spec.name,
            "family": spec.family,
            "reference": "original" if spec.family == "suppress" else "L1_half",
            "changed_channels": ",".join(map(str, spec.channels_to_change)),
            "base_keep": spec.base_keep,
            "changed_keep": spec.change_keep,
            "frames": len(rows),
            "hits": hits,
            "nme_matched_only": ac.mean_of([r["nme"] for r in rows]),
            "mean_boxes": ac.mean_of([r["n_boxes"] for r in rows]),
            "elapsed_s": round(time.perf_counter() - start, 2),
        }
        for k in range(KEYPOINTS):
            result[f"k{k}_px_matched_only"] = ac.mean_of([r[f"k{k}_px"] for r in rows])
        summary.append(result)
        per_image.extend({"case": spec.name, **r} for r in rows)
        print(f"{spec.name:<23} hits {hits:>3}/{len(rows)} NME={fmt(result['nme_matched_only'])} "
              f"time={result['elapsed_s']:.1f}s")

    paired_rows: list[dict[str, Any]] = []
    for result in summary:
        name = result["case"]
        ref_name = result["reference"]
        comparison = changes_against(all_rows[ref_name], all_rows[name], "paired")
        result.update(comparison)
        if name != ref_name:
            paired_rows.extend(make_paired_rows(name, ref_name, all_rows[ref_name], all_rows[name]))

    print("\nPAIRED COMPARISON (different baselines by intervention type)")
    print(f"{'case':<23} {'reference':<9} {'hits':>4} {'pairs':>5} {'NME':>9} {'delta%':>9} "
          f"{'dK3':>8} {'dK4':>8} {'dK5':>8} {'dK6':>8} {'wins':>5}")
    for r in summary:
        print(f"{r['case']:<23} {r['reference']:<9} {r['hits']:>4} {r['paired_paired']:>5} "
              f"{fmt(r['paired_case_nme']):>9} {fmt(r['paired_delta_pct'],2):>9} "
              f"{fmt(r['paired_k3_delta_px'],2):>8} {fmt(r['paired_k4_delta_px'],2):>8} "
              f"{fmt(r['paired_k5_delta_px'],2):>8} {fmt(r['paired_k6_delta_px'],2):>8} "
              f"{r['paired_win_images']:>5}")

    all_matched = {name: hits_map(rows) for name, rows in all_rows.items()}
    common = sorted(set.intersection(*(set(x) for x in all_matched.values())))
    same_frames_summary = []
    for spec in cases:
        sub = [all_matched[spec.name][s] for s in common]
        entry = {"case": spec.name, "common_frames": len(common),
                 "nme_common": ac.mean_of([r["nme"] for r in sub])}
        for k in range(KEYPOINTS):
            entry[f"k{k}_px_common"] = ac.mean_of([r[f"k{k}_px"] for r in sub])
        same_frames_summary.append(entry)
    print(f"\nAll-{len(cases)}-case matched intersection: {len(common)}/{len(stems)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    channel_profile = []
    for g, (lo, hi) in enumerate(groups, 1):
        for ch in range(lo, hi):
            channel_profile.append({"channel_0based": ch, "channel_1based": ch + 1,
                                    "group": f"G{g}", "target_group": g == args.target_group})
    write_csv(out_dir / "channel_profile.csv", channel_profile, list(channel_profile[0]))
    summary_fields = ["case", "family", "reference", "changed_channels", "base_keep",
                      "changed_keep", "frames", "hits", "nme_matched_only", "mean_boxes",
                      "elapsed_s"] + [f"k{k}_px_matched_only" for k in range(KEYPOINTS)]
    summary_fields += ["paired_paired", "paired_ref_nme", "paired_case_nme",
                       "paired_delta_nme", "paired_delta_pct", "paired_win_images", "paired_lose_images"]
    for k in range(KEYPOINTS):
        summary_fields += [f"paired_k{k}_ref_px", f"paired_k{k}_case_px",
                           f"paired_k{k}_delta_px", f"paired_k{k}_win_images"]
    summary_fields += ["paired_k456_delta_px", "paired_k3456_delta_px"]
    write_csv(out_dir / "summary.csv", summary, summary_fields)
    write_csv(out_dir / "per_image.csv", per_image,
              ["case", "image", "n_boxes", "matched", "iou", "nme"] +
              [f"k{k}_px" for k in range(KEYPOINTS)])
    paired_fields = ["case", "reference", "image", "paired", "reference_matched", "case_matched",
                     "reference_nme", "case_nme", "delta_nme", "case_iou", "case_boxes"]
    for k in range(KEYPOINTS):
        paired_fields += [f"reference_k{k}_px", f"case_k{k}_px", f"delta_k{k}_px"]
    write_csv(out_dir / "paired_per_image.csv", paired_rows, paired_fields)
    write_csv(out_dir / "same_frames_summary.csv", same_frames_summary,
              ["case", "common_frames", "nme_common"] +
              [f"k{k}_px_common" for k in range(KEYPOINTS)])
    config = {
        "dataset_root": str(root), "weights": str(weight), "split": args.split,
        "stems": len(stems), "ultralytics": ac._LOCKED_VERSION,
        "eval_impl": ac._EVAL_SCRIPT.name,
        "layer_index": LAYER_INDEX, "channels": channels,
        "groups": [{"name": f"G{i}", "start_inclusive": lo, "stop_exclusive": hi}
                   for i, (lo, hi) in enumerate(groups, 1)],
        "target_group": args.target_group, "base_keep": args.base_keep,
        "restore_keep": args.restore_keep, "mode": args.mode,
        "conf": args.conf, "imgsz": args.imgsz, "match_iou": args.match_iou,
        "cases": [{"name": c.name, "family": c.family,
                   "channels_to_change": list(c.channels_to_change),
                   "base_keep": c.base_keep, "change_keep": c.change_keep} for c in cases],
        "all_condition_common_frames": len(common),
        "description": "Fresh weights per condition. Restore cases pair vs all-half, suppress cases pair vs original."
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved: {out_dir}")
    print("summary.csv | per_image.csv | paired_per_image.csv | same_frames_summary.csv | channel_profile.csv | run_config.json")


if __name__ == "__main__":
    main()
