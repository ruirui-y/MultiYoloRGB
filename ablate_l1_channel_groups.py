r"""YOLO11-RGBD: restore one L1 output channel group at a time (no retraining).

Design
------
    original     : trained checkpoint, no intervention
    L1_half      : ALL channels of top-level module 1 output multiplied by 0.50
    restore_G1.. : one contiguous group multiplied by 1.00; remaining groups by 0.50

Groups are automatically formed from the *actual configured output-channel count*
of module 1 and validated against its runtime Tensor output. No weights are edited.
Each condition loads a fresh checkpoint. Predictions/GT matching/NME are reused
from the user's existing ablate_c2psa.py, retaining the exact evaluation protocol.

Place this script next to ablate_c2psa.py and ablate_layers.py in MultiYoloRGB.
Run from Windows CMD:
    .venv\Scripts\python.exe ablate_l1_channel_groups.py

Outputs (default runs/ablate_l1_channel_groups/):
    summary.csv                all conditions, paired NME and K0-K6 changes
    per_image.csv              raw per-image evaluation data
    paired_per_image.csv       per-image deltas versus the L1_half baseline
    group_profile.csv          zero- and one-based channel ranges of each group
    same_frames_summary.csv    fair comparison on frames matched by ALL conditions
    run_config.json            exact experiment settings
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

import ablate_c2psa as ac
from ablate_layers import load_items

LAYER_INDEX = 1
KEYPOINTS = ac.KEYPOINT_COUNT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO11-RGBD L1 channel-group restoration sweep")
    parser.add_argument("--dataset-root", default=str(ac._DEFAULT_DATASET))
    parser.add_argument("--weights-dir", default=str(ac._REPO / "train_weight"))
    parser.add_argument("--weights-name", default="yolo11n_rgbd_4ch")
    parser.add_argument("--output-dir", default=str(ac._REPO / "runs" / "ablate_l1_channel_groups"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--base-keep", type=float, default=0.50)
    parser.add_argument("--restore-keep", type=float, default=1.00)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--expected-count", type=int, default=100,
                        help="Expected number of common RGB/RGBD frames (0 disables check)")
    return parser.parse_args()


def channel_groups(channels: int, count: int) -> list[tuple[int, int]]:
    """Contiguous [start, stop) channel ranges, balanced within one channel."""
    if count < 1 or channels < count:
        raise ValueError(f"Cannot split {channels} channels into {count} nonempty groups")
    size, extras = divmod(channels, count)
    result = []
    start = 0
    for group in range(count):
        stop = start + size + (1 if group < extras else 0)
        result.append((start, stop))
        start = stop
    assert start == channels
    return result


def check_l1_and_get_channels(model: Any) -> int:
    layers = model.model.model
    if len(layers) <= LAYER_INDEX:
        raise RuntimeError(f"Model has only {len(layers)} modules; cannot find index {LAYER_INDEX}")
    module = layers[LAYER_INDEX]
    if type(module).__name__ != "Conv":
        raise RuntimeError(f"Layer {LAYER_INDEX} should be Conv, found {type(module).__name__}")
    conv = getattr(module, "conv", None)
    if not isinstance(conv, nn.Conv2d):
        raise RuntimeError(f"Layer {LAYER_INDEX}: expected Ultralytics Conv.conv (Conv2d)")
    return int(conv.out_channels)


class L1ChannelGroupScaler(nn.Module):
    """Execute the original Conv, then rescale specified output channels.

    Preserves Ultralytics graph metadata (.f/.i/.type) exactly as prior
    LayerAttenuator experiments did; intervention happens BEFORE first predict.
    """

    def __init__(self, source: nn.Module, channels: int, ranges: list[tuple[int, int]],
                 base_keep: float, restore_keep: float,
                 restore_index: int | None) -> None:
        super().__init__()
        self.source = source
        self.f = source.f
        self.i = source.i
        self.type = "L1ChannelGroupScaler"
        self.channels = channels
        self.restore_index = restore_index
        self.base_keep = base_keep
        self.restore_keep = restore_keep

        scales = torch.full((channels,), base_keep, dtype=torch.float32)
        if restore_index is not None:
            first, stop = ranges[restore_index]
            scales[first:stop] = restore_keep
        # Buffer tracks the model's device, is excluded from .pt persistence,
        # and is cast to the actual Conv output dtype at inference time.
        self.register_buffer("scales", scales.view(1, channels, 1, 1), persistent=False)
        self._checked = False

    def forward(self, x: Any) -> torch.Tensor:
        y = self.source(x)
        if not isinstance(y, torch.Tensor) or y.ndim != 4:
            raise TypeError("L1 output must be a 4-D NCHW Tensor")
        if y.shape[1] != self.channels:
            raise RuntimeError(f"L1 channels changed: expected {self.channels}, got {y.shape[1]}")
        self._checked = True
        return y * self.scales.to(device=y.device, dtype=y.dtype)


def install_scaler(model: Any, channels: int, ranges: list[tuple[int, int]],
                   base_keep: float, restore_keep: float,
                   restore_index: int | None) -> None:
    current_channels = check_l1_and_get_channels(model)
    if current_channels != channels:
        raise RuntimeError(f"Channel count differs between checkpoints: {current_channels} != {channels}")
    original = model.model.model[LAYER_INDEX]
    model.model.model[LAYER_INDEX] = L1ChannelGroupScaler(
        original, channels, ranges, base_keep, restore_keep, restore_index
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def hits_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["image"]: row for row in rows if row["nme"] is not None}


def fmt(value: Any, decimals: int = 6) -> str:
    return "-" if value is None else f"{value:.{decimals}f}"


def changes_against(reference: list[dict[str, Any]],
                    candidate: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    before = hits_map(reference)
    after = hits_map(candidate)
    frames = sorted(before.keys() & after.keys())
    ref_nme = ac.mean_of([before[s]["nme"] for s in frames])
    cur_nme = ac.mean_of([after[s]["nme"] for s in frames])
    delta_nme = cur_nme - ref_nme if ref_nme is not None and cur_nme is not None else None
    result: dict[str, Any] = {
        f"{prefix}_paired": len(frames),
        f"{prefix}_ref_nme": ref_nme,
        f"{prefix}_case_nme": cur_nme,
        f"{prefix}_delta_nme": delta_nme,
        f"{prefix}_delta_pct": (100.0 * delta_nme / ref_nme
                               if delta_nme is not None and ref_nme else None),
        f"{prefix}_win_images": sum(after[s]["nme"] < before[s]["nme"] for s in frames),
        f"{prefix}_lose_images": sum(after[s]["nme"] > before[s]["nme"] for s in frames),
    }
    for k in range(KEYPOINTS):
        field = f"k{k}_px"
        pairs = [(before[s][field], after[s][field]) for s in frames
                 if before[s][field] is not None and after[s][field] is not None]
        result[f"{prefix}_k{k}_ref_px"] = ac.mean_of([r for r, _ in pairs])
        result[f"{prefix}_k{k}_case_px"] = ac.mean_of([c for _, c in pairs])
        result[f"{prefix}_k{k}_delta_px"] = ac.mean_of([c - r for r, c in pairs])
        result[f"{prefix}_k{k}_win_images"] = sum(c < r for r, c in pairs)
    for name, indices in (("k456", (4, 5, 6)), ("k3456", (3, 4, 5, 6))):
        # Pair per image, then average; equivalent weights per valid image.
        per_image_deltas = []
        for s in frames:
            deltas = [after[s][f"k{k}_px"] - before[s][f"k{k}_px"] for k in indices
                      if after[s][f"k{k}_px"] is not None
                      and before[s][f"k{k}_px"] is not None]
            if deltas:
                per_image_deltas.append(sum(deltas) / len(deltas))
        result[f"{prefix}_{name}_delta_px"] = ac.mean_of(per_image_deltas)
    return result


def make_paired_image_rows(case: str,
                           base_rows: list[dict[str, Any]],
                           case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = {row["image"]: row for row in base_rows}
    altered = {row["image"]: row for row in case_rows}
    if baseline.keys() != altered.keys():
        raise RuntimeError("Image stems differ between baseline and intervention")
    result = []
    for stem in sorted(baseline):
        left, right = baseline[stem], altered[stem]
        paired = left["nme"] is not None and right["nme"] is not None
        nme_delta = right["nme"] - left["nme"] if paired else None
        row: dict[str, Any] = {
            "case": case,
            "image": stem,
            "paired": paired,
            "l1_matched": left["nme"] is not None,
            "case_matched": right["nme"] is not None,
            "case_boxes": right["n_boxes"],
            "case_iou": right["iou"],
            "l1_nme": left["nme"],
            "case_nme": right["nme"],
            "delta_nme": nme_delta,
        }
        for k in range(KEYPOINTS):
            field = f"k{k}_px"
            p, q = left[field], right[field]
            row[f"l1_k{k}_px"] = p
            row[f"case_k{k}_px"] = q
            row[f"delta_k{k}_px"] = q - p if paired and p is not None and q is not None else None
        result.append(row)
    return result


def main() -> None:
    args = parse_args()
    if args.groups < 2:
        raise ValueError("--groups must be >=2 (the requested experiment uses 4)")
    for key, keep in (("base-keep", args.base_keep), ("restore-keep", args.restore_keep)):
        if not 0.0 <= keep <= 1.0:
            raise ValueError(f"--{key} must be between 0 and 1")
    if args.restore_keep < args.base_keep:
        raise ValueError("restore-keep must be >= base-keep for a restoration experiment")

    root = Path(args.dataset_root).expanduser()
    weights = Path(args.weights_dir).expanduser() / (args.weights_name.removesuffix(".pt") + ".pt")
    out_dir = Path(args.output_dir).expanduser()
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")

    rgb = ac.stem_map(root / "rgb" / "images" / args.split)
    rgbd = ac.stem_map(root / "rgbd" / "images" / args.split)
    labels = root / "rgb" / "labels" / args.split
    if not labels.is_dir():
        labels = root / "rgbd" / "labels" / args.split
    if not labels.is_dir():
        raise FileNotFoundError(f"Labels missing: {labels}")
    stems = sorted(s for s in rgb.keys() & rgbd.keys() if (labels / f"{s}.txt").is_file())
    if not stems:
        raise RuntimeError("No aligned RGB/RGBD stems with ground-truth labels")
    if args.expected_count and len(stems) != args.expected_count:
        raise RuntimeError(f"Expected {args.expected_count} shared test frames, found {len(stems)}")

    # The prior benchmark already uses this shared 100-image test set.
    items = load_items(root, args.split, 4, stems)
    probe = ac.YOLO(str(weights))
    channels = check_l1_and_get_channels(probe)
    ranges = channel_groups(channels, args.groups)
    del probe
    gc.collect()

    cases: list[tuple[str, int | None]] = [
        ("original", None), ("L1_half", None),
        *((f"restore_G{i + 1}", i) for i in range(args.groups)),
    ]
    print("=" * 110)
    print("YOLO11-RGBD L1 OUTPUT CHANNEL GROUP RESTORATION (NO TRAINING)")
    print(f"dataset={root} | split={args.split} | frames={len(stems)}")
    print(f"weights={weights} | ultralytics={ac._LOCKED_VERSION}")
    print(f"conf={args.conf} imgsz={args.imgsz} match_iou={args.match_iou}")
    print(f"L1 output channels={channels} | groups={args.groups} | base_keep={args.base_keep} "
          f"| restore_keep={args.restore_keep}")
    for i, (start, stop) in enumerate(ranges, 1):
        print(f"  G{i}: zero-based channels [{start},{stop}) = {start}..{stop - 1} "
              f"({stop - start} channels)")
    print("Each group: only its channels restore to 1.00; all others remain 0.50")
    print("=" * 110)

    all_rows: dict[str, list[dict[str, Any]]] = {}
    summary: list[dict[str, Any]] = []
    per_image: list[dict[str, Any]] = []

    for case, group_index in cases:
        model = ac.YOLO(str(weights))
        if case != "original":
            install_scaler(model, channels, ranges, args.base_keep, args.restore_keep,
                           group_index)
        started = time.perf_counter()
        try:
            rows = ac.run_condition(model, items, args.conf, args.imgsz, args.match_iou)
        finally:
            del model
            gc.collect()
        elapsed = time.perf_counter() - started
        all_rows[case] = rows
        hits = len(hits_map(rows))
        record: dict[str, Any] = {
            "case": case,
            "restore_group": "" if group_index is None else f"G{group_index + 1}",
            "group_start_channel": "" if group_index is None else ranges[group_index][0],
            "group_stop_exclusive": "" if group_index is None else ranges[group_index][1],
            "base_keep": 1.0 if case == "original" else args.base_keep,
            "restored_keep": 1.0 if case == "original" else (
                args.base_keep if group_index is None else args.restore_keep),
            "frames": len(rows),
            "hits": hits,
            "mean_boxes": ac.mean_of([r["n_boxes"] for r in rows]),
            "nme_matched_only": ac.mean_of([r["nme"] for r in rows]),
            "seconds": round(elapsed, 2),
        }
        for k in range(KEYPOINTS):
            record[f"k{k}_px_matched_only"] = ac.mean_of([r[f"k{k}_px"] for r in rows])
        summary.append(record)
        per_image.extend({"case": case, **r} for r in rows)
        print(f"{case:<14} hits={hits:3}/{len(rows)} NME(matched)="
              f"{fmt(record['nme_matched_only'])} elapsed={elapsed:.1f}s")

    paired_rows: list[dict[str, Any]] = []
    for record in summary:
        case = record["case"]
        record.update(changes_against(all_rows["L1_half"], all_rows[case], "vs_l1"))
        record.update(changes_against(all_rows["original"], all_rows[case], "vs_original"))
        if case != "L1_half":
            paired_rows.extend(make_paired_image_rows(case, all_rows["L1_half"], all_rows[case]))

    all_maps = {name: hits_map(rows) for name, rows in all_rows.items()}
    common = sorted(set.intersection(*(set(x) for x in all_maps.values())))
    same_frame_summary: list[dict[str, Any]] = []
    for case, _ in cases:
        paired = [all_maps[case][s] for s in common]
        row = {"case": case, "common_frames": len(common),
               "nme_common": ac.mean_of([r["nme"] for r in paired])}
        for k in range(KEYPOINTS):
            row[f"k{k}_px_common"] = ac.mean_of([r[f"k{k}_px"] for r in paired])
        same_frame_summary.append(row)

    print("\nFair comparison versus L1_half (on each case's same matched frames)")
    print(f"{'case':<14} {'hits':>8} {'paired':>7} {'ref_NME':>10} {'case_NME':>10} "
          f"{'delta%':>9} {'dK4(px)':>9} {'dK5(px)':>9} {'dK6(px)':>9} {'dK456':>9}")
    for record in summary:
        print(f"{record['case']:<14} {record['hits']:>3}/{record['frames']:<4} "
              f"{record['vs_l1_paired']:>7} "
              f"{fmt(record['vs_l1_ref_nme']):>10} "
              f"{fmt(record['vs_l1_case_nme']):>10} "
              f"{fmt(record['vs_l1_delta_pct'], 2):>9} "
              f"{fmt(record['vs_l1_k4_delta_px'], 2):>9} "
              f"{fmt(record['vs_l1_k5_delta_px'], 2):>9} "
              f"{fmt(record['vs_l1_k6_delta_px'], 2):>9} "
              f"{fmt(record['vs_l1_k456_delta_px'], 2):>9}")
    print(f"\nAll-condition matched intersection: {len(common)}/{len(stems)} frames")
    if common:
        for row in sorted(same_frame_summary, key=lambda r: r["nme_common"]):
            print(f"  {row['case']:<14} NME={fmt(row['nme_common'])}")
    print("NOTE: negative delta = lower error; no conclusion about causal root cause yet.")

    out_dir.mkdir(parents=True, exist_ok=True)
    group_profile = [{
        "group": f"G{i + 1}", "output_channels": channels,
        "start_0based_inclusive": start,
        "stop_0based_exclusive": stop,
        "start_1based": start + 1,
        "end_1based": stop,
        "channel_count": stop - start,
        "base_keep": args.base_keep, "restore_keep": args.restore_keep,
    } for i, (start, stop) in enumerate(ranges)]
    write_csv(out_dir / "group_profile.csv", group_profile, list(group_profile[0]))

    summary_fields = [
        "case", "restore_group", "group_start_channel", "group_stop_exclusive",
        "base_keep", "restored_keep", "frames", "hits", "mean_boxes",
        "nme_matched_only", "seconds",
    ] + [f"k{k}_px_matched_only" for k in range(KEYPOINTS)]
    for prefix in ("vs_l1", "vs_original"):
        summary_fields += [
            f"{prefix}_paired", f"{prefix}_ref_nme", f"{prefix}_case_nme",
            f"{prefix}_delta_nme", f"{prefix}_delta_pct",
            f"{prefix}_win_images", f"{prefix}_lose_images",
        ]
        for k in range(KEYPOINTS):
            summary_fields += [f"{prefix}_k{k}_ref_px", f"{prefix}_k{k}_case_px",
                               f"{prefix}_k{k}_delta_px", f"{prefix}_k{k}_win_images"]
        summary_fields += [f"{prefix}_k456_delta_px", f"{prefix}_k3456_delta_px"]
    write_csv(out_dir / "summary.csv", summary, summary_fields)

    per_fields = ["case", "image", "n_boxes", "matched", "iou", "nme"] + [
        f"k{k}_px" for k in range(KEYPOINTS)
    ]
    write_csv(out_dir / "per_image.csv", per_image, per_fields)
    paired_fields = [
        "case", "image", "paired", "l1_matched", "case_matched", "case_boxes", "case_iou",
        "l1_nme", "case_nme", "delta_nme",
    ]
    for k in range(KEYPOINTS):
        paired_fields += [f"l1_k{k}_px", f"case_k{k}_px", f"delta_k{k}_px"]
    write_csv(out_dir / "paired_per_image.csv", paired_rows, paired_fields)
    common_fields = ["case", "common_frames", "nme_common"] + [
        f"k{k}_px_common" for k in range(KEYPOINTS)
    ]
    write_csv(out_dir / "same_frames_summary.csv", same_frame_summary, common_fields)

    (out_dir / "run_config.json").write_text(json.dumps({
        "dataset_root": str(root), "split": args.split, "count": len(stems),
        "weights": str(weights), "model": args.weights_name,
        "layer_index": LAYER_INDEX,
        "intervention": "L1 output channel-wise scaling; four balanced contiguous groups",
        "channels": channels,
        "groups": [{"name": f"G{i + 1}", "start_inclusive": b,
                    "stop_exclusive": e} for i, (b, e) in enumerate(ranges)],
        "base_keep": args.base_keep, "restore_keep": args.restore_keep,
        "cases": [name for name, _ in cases],
        "conf": args.conf, "imgsz": args.imgsz, "match_iou": args.match_iou,
        "ultralytics": ac._LOCKED_VERSION, "eval_impl": ac._EVAL_SCRIPT.name,
        "all_condition_common_frames": len(common),
        "note": "Each case starts with a fresh checkpoint; only one group is restored per run. "
                "NME changes are compared on the same matched image stems.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nResults saved to: {out_dir}")
    print("summary.csv | per_image.csv | paired_per_image.csv | group_profile.csv | "
          "same_frames_summary.csv | run_config.json")


if __name__ == "__main__":
    main()
