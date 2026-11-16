"""Zero-training layer intervention sweep for YOLO11 Pose.

Place this file beside the user's ablate_c2psa.py. It reuses the same
image loading, matching, NME, and keypoint-error functions.

Two DIFFERENT interventions are deliberately kept separate:
  identity  : F(x) -> x, only when actual input and output tensor shapes match.
  attenuate : F(x) -> keep * F(x), for tensor-output layers 0..22.
  blend     : F(x) -> (1-alpha)*F(x) + alpha*x, shape-equal modules only.

Neither intervention estimates retrained architecture performance.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

import ablate_c2psa as ac


class LayerAttenuator(nn.Module):
    """Wrap original module while preserving graph metadata and output shape."""

    def __init__(self, source: nn.Module, keep: float) -> None:
        super().__init__()
        self.source = source
        self.keep = float(keep)
        self.f = source.f
        self.i = source.i
        self.type = "LayerAttenuator"

    def forward(self, x: Any) -> torch.Tensor:
        out = self.source(x)
        if not isinstance(out, torch.Tensor):
            raise TypeError("Attenuation requires a Tensor output")
        return out * self.keep


class LayerBlender(nn.Module):
    """Interpolate between the trained module output and true identity bypass."""

    def __init__(self, source: nn.Module, alpha: float) -> None:
        super().__init__()
        self.source = source
        self.alpha = float(alpha)
        self.f = source.f
        self.i = source.i
        self.type = "LayerBlender"

    def forward(self, x: Any) -> torch.Tensor:
        y = self.source(x)
        if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError("Blend requires Tensor input and output")
        if x.shape != y.shape:
            raise ValueError(f"Blend requires equal shapes: {x.shape} vs {y.shape}")
        return (1.0 - self.alpha) * y + self.alpha * x


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO11 Pose layer-by-layer ablations")
    p.add_argument("--dataset-root", default=str(ac._DEFAULT_DATASET))
    p.add_argument("--weights-dir", default=str(ac._REPO / "train_weight"))
    p.add_argument("--output-dir", default=None,
                   help="Default: runs/ablate_layers/<mode-specific-subfolder>")
    p.add_argument("--models", nargs="+", default=[m[0] for m in ac.DEFAULT_MODELS])
    p.add_argument("--layers", nargs="+", default=["all"],
                   help="Layer indexes, e.g. --layers 2 4 6 8 9 10, or all")
    p.add_argument("--mode", choices=["identity", "attenuate", "blend", "both", "all"],
                   default="identity")
    p.add_argument("--keep", type=float, default=0.5,
                   help="For attenuate: multiplier applied to original layer output, 0..1")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="For blend: 0=original layer, 1=complete identity bypass")
    p.add_argument("--split", default="test")
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--match-iou", type=float, default=0.5)
    p.add_argument("--min-retention", type=float, default=0.9,
                   help="Retention threshold for candidate flag, default 0.9")
    return p.parse_args()


def save_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def extract_model_specs(names: list[str]) -> list[tuple[str, str, int]]:
    specs = []
    for name in names:
        clean = name.removesuffix(".pt")
        match = [entry for entry in ac.DEFAULT_MODELS if entry[0] == clean]
        if not match:
            raise ValueError(f"Unsupported model: {name}; allowed: {ac.DEFAULT_MODELS}")
        specs.append(match[0])
    return specs


def probe_shapes(model: Any, channels: int, size: int) -> list[dict[str, Any]]:
    """Run one RAW forward to discover which top-level layers allow Identity.

    This pass does not use predictor/preprocessing and is for shape checks only.
    """
    modules = model.model.model
    profile = [{"layer": i, "module": type(m).__name__,
                "input_shape": None, "output_shape": None,
                "identity_eligible": False, "attenuate_eligible": False}
               for i, m in enumerate(modules)]
    handles = []

    def pre_hook(index: int):
        def callback(_module: nn.Module, arguments: tuple[Any, ...]) -> None:
            if arguments and isinstance(arguments[0], torch.Tensor):
                profile[index]["input_shape"] = tuple(arguments[0].shape)
        return callback

    def post_hook(index: int):
        def callback(_module: nn.Module, _arguments: tuple[Any, ...], output: Any) -> None:
            if isinstance(output, torch.Tensor):
                profile[index]["output_shape"] = tuple(output.shape)
        return callback

    for i, module in enumerate(modules):
        handles.append(module.register_forward_pre_hook(pre_hook(i)))
        handles.append(module.register_forward_hook(post_hook(i)))
    try:
        raw = model.model
        raw.eval()
        p = next(raw.parameters())
        dummy = torch.zeros((1, channels, size, size),
                            dtype=p.dtype, device=p.device)
        with torch.inference_mode():
            raw(dummy)
    finally:
        for handle in handles:
            handle.remove()

    for entry in profile:
        idx = entry["layer"]
        # Last layer is the Pose prediction head, not a feature tensor.
        eligible_output = entry["output_shape"] is not None and idx < len(profile) - 1
        entry["attenuate_eligible"] = eligible_output
        entry["identity_eligible"] = (
            eligible_output
            and entry["input_shape"] is not None
            and entry["input_shape"] == entry["output_shape"]
        )
    return profile


def choose_layers(layer_args: list[str], profile: list[dict[str, Any]]) -> list[int]:
    max_idx = len(profile) - 2  # exclude output/head layer
    if layer_args == ["all"]:
        return list(range(max_idx + 1))
    try:
        result = sorted(set(int(n) for n in layer_args))
    except ValueError as e:
        raise ValueError("--layers should be all or space-separated integers") from e
    if not result or min(result) < 0 or max(result) > max_idx:
        raise ValueError(f"Valid layer range: 0..{max_idx}")
    return result


def load_items(root: Path, split: str, channels: int,
               common_stems: list[str]) -> list[tuple[str, np.ndarray, dict[str, Any]]]:
    source = root / ("rgbd" if channels == 4 else "rgb") / "images" / split
    label_dir = root / "rgb" / "labels" / split
    if not label_dir.is_dir():
        label_dir = root / "rgbd" / "labels" / split
    img_map = ac.stem_map(source)
    items = []
    for stem in common_stems:
        if stem not in img_map:
            raise FileNotFoundError(f"Image missing: {source} / {stem}")
        image = ac.read_image(img_map[stem], channels)
        h, w = image.shape[:2]
        gt = ac.ev.load_pose_label(label_dir / f"{stem}.txt", w, h)
        items.append((stem, image, gt))
    return items


def analyze_condition(model_name: str, label: str, layer: int, module: str,
                      mode: str, keep: float | None,
                      baseline: list[dict[str, Any]], changed: list[dict[str, Any]],
                      seconds: float) -> dict[str, Any]:
    before = {row["image"]: row for row in baseline if row["nme"] is not None}
    after = {row["image"]: row for row in changed if row["nme"] is not None}
    common = sorted(before.keys() & after.keys())
    base_paired = ac.mean_of([before[s]["nme"] for s in common])
    ablated_paired = ac.mean_of([after[s]["nme"] for s in common])
    delta = (ablated_paired - base_paired) if (base_paired is not None and ablated_paired is not None) else None
    pct = (100 * delta / base_paired) if (delta is not None and base_paired) else None
    count = len(changed)
    base_hits = sum(r["nme"] is not None for r in baseline)
    ablated_hits = len(after)
    row = {
        "model": model_name, "label": label, "layer": layer, "module": module,
        "mode": mode, "keep": keep, "images": count,
        "baseline_hits": base_hits, "ablated_hits": ablated_hits,
        "retention": (ablated_hits / base_hits) if base_hits else None,
        "boxes_per_image": ac.mean_of([r["n_boxes"] for r in changed]),
        "baseline_nme_all_hits": ac.mean_of([r["nme"] for r in baseline]),
        "ablated_nme_own_hits": ac.mean_of([r["nme"] for r in changed]),
        "paired_images": len(common), "paired_baseline_nme": base_paired,
        "paired_ablated_nme": ablated_paired,
        "paired_delta_nme": delta, "paired_delta_pct": pct,
        "seconds": round(seconds, 3),
    }
    for k in range(ac.KEYPOINT_COUNT):
        field = f"k{k}_px"
        paired_k = [(before[s][field], after[s][field]) for s in common
                    if before[s][field] is not None and after[s][field] is not None]
        row[f"k{k}_baseline_px"] = ac.mean_of([pair[0] for pair in paired_k])
        row[f"k{k}_ablated_px"] = ac.mean_of([pair[1] for pair in paired_k])
        row[f"k{k}_delta_px"] = ac.mean_of([pair[1] - pair[0] for pair in paired_k])
    return row


def baseline_row(model_name: str, label: str,
                 original: list[dict[str, Any]], seconds: float) -> dict[str, Any]:
    return analyze_condition(model_name, label, -1, "baseline", "baseline", 1.0,
                             original, original, seconds)


def print_result(row: dict[str, Any]) -> None:
    v = lambda k: "-" if row[k] is None else f"{row[k]:+.2f}"  # noqa: E731
    print(f"  {str(row['layer']).rjust(2)} {row['module']:<16} "
          f"{row['mode']:<10} matched {row['ablated_hits']:>3}/{row['images']} "
          f"paired {row['paired_images']:>3} "
          f"NME(base/abl) {ac.fmt(row['paired_baseline_nme'])}/"
          f"{ac.fmt(row['paired_ablated_nme'])} "
          f"d% {v('paired_delta_pct'):>7} "
          f"K3d {v('k3_delta_px'):>7} px")


def run() -> None:
    args = parse_args()
    if not 0 <= args.keep <= 1:
        raise ValueError("--keep must be between 0 and 1")
    if not 0 <= args.alpha <= 1:
        raise ValueError("--alpha must be between 0 and 1")
    if not 0 <= args.min_retention <= 1:
        raise ValueError("--min-retention must be between 0 and 1")
    root = Path(args.dataset_root)
    weight_dir = Path(args.weights_dir)
    default_subfolder = {
        "identity": "identity",
        "attenuate": f"attenuate_keep_{args.keep:.2f}".replace(".", "p"),
        "blend": f"blend_alpha_{args.alpha:.2f}".replace(".", "p"),
        "both": f"both_keep_{args.keep:.2f}".replace(".", "p"),
        "all": f"all_keep_{args.keep:.2f}_alpha_{args.alpha:.2f}".replace(".", "p"),
    }[args.mode]
    out = Path(args.output_dir) if args.output_dir else ac._REPO / "runs" / "ablate_layers" / default_subfolder
    out.mkdir(parents=True, exist_ok=True)
    specs = extract_model_specs(args.models)
    modes = {"both": ["identity", "attenuate"],
             "all": ["identity", "attenuate", "blend"]}.get(args.mode, [args.mode])

    rgb_map = ac.stem_map(root / "rgb" / "images" / args.split)
    rgbd_map = ac.stem_map(root / "rgbd" / "images" / args.split)
    labels = root / "rgb" / "labels" / args.split
    if not labels.is_dir():
        labels = root / "rgbd" / "labels" / args.split
    if not labels.is_dir():
        raise FileNotFoundError(f"Labels folder missing: {labels}")
    # Same frame set for all selected models; both roots must exist.
    common_stems = sorted(s for s in rgb_map.keys() & rgbd_map.keys()
                          if (labels / f"{s}.txt").is_file())
    if not common_stems:
        raise RuntimeError("No common RGB/RGBD images with labels")

    print(f"Dataset: {root} | common frames: {len(common_stems)}")
    print(f"conf={args.conf} imgsz={args.imgsz} match_iou={args.match_iou}")
    print("identity = pure bypass for shape-equal modules only")
    print("attenuate = scaled output (NOT removal); keep =", args.keep)
    print("blend = gradual bypass for shape-equal layers; alpha =", args.alpha)

    summary: list[dict[str, Any]] = []
    per_image: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    # Needed for cross-model four-way matched comparisons, if both selected.
    all_cases: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    all_bases: dict[str, list[dict[str, Any]]] = {}

    for model_name, label, channels in specs:
        weight_path = weight_dir / f"{model_name}.pt"
        if not weight_path.is_file():
            raise FileNotFoundError(f"Weights missing: {weight_path}")
        print(f"\n=== {label} ({channels}ch) ===")
        items = load_items(root, args.split, channels, common_stems)
        pre = ac.YOLO(str(weight_path))
        profile = probe_shapes(pre, channels, args.imgsz)
        for entry in profile:
            profile_rows.append({"model": model_name, **entry})
        requested = choose_layers(args.layers, profile)
        del pre
        gc.collect()

        baseline_model = ac.YOLO(str(weight_path))
        started = time.perf_counter()
        baseline = ac.run_condition(baseline_model, items, args.conf, args.imgsz, args.match_iou)
        elapsed = time.perf_counter() - started
        all_bases[model_name] = baseline
        base_rec = baseline_row(model_name, label, baseline, elapsed)
        summary.append(base_rec)
        print(f"  BASE matched {base_rec['ablated_hits']}/{base_rec['images']} "
              f"NME {ac.fmt(base_rec['ablated_nme_own_hits'])}")
        del baseline_model
        gc.collect()

        for mode in modes:
            for layer in requested:
                info = profile[layer]
                eligible = (info["attenuate_eligible"] if mode == "attenuate"
                            else info["identity_eligible"])
                if not eligible:
                    skipped.append({"model": model_name, "mode": mode, "layer": layer,
                                    "module": info["module"], "input_shape": info["input_shape"],
                                    "output_shape": info["output_shape"], "reason": "shape/type incompatible"})
                    continue
                m = ac.YOLO(str(weight_path))  # Fresh weights before first prediction.
                orig = m.model.model[layer]
                if mode == "identity":
                    m.model.model[layer] = ac._SkipLayer(orig)
                elif mode == "blend":
                    m.model.model[layer] = LayerBlender(orig, args.alpha)
                else:
                    m.model.model[layer] = LayerAttenuator(orig, args.keep)
                start = time.perf_counter()
                try:
                    changed = ac.run_condition(m, items, args.conf, args.imgsz, args.match_iou)
                except Exception as exc:
                    skipped.append({"model": model_name, "mode": mode, "layer": layer,
                                    "module": info["module"], "input_shape": info["input_shape"],
                                    "output_shape": info["output_shape"], "reason": repr(exc)})
                    print(f"  layer {layer} {mode} FAILED: {exc}")
                    del m
                    gc.collect()
                    continue
                elapsed = time.perf_counter() - start
                rec = analyze_condition(model_name, label, layer, info["module"],
                                        mode, (args.keep if mode == "attenuate" else
                                               args.alpha if mode == "blend" else None),
                                        baseline, changed, elapsed)
                rec["candidate"] = bool(rec["paired_delta_nme"] is not None
                                        and rec["paired_delta_nme"] < 0
                                        and rec["retention"] is not None
                                        and rec["retention"] >= args.min_retention)
                summary.append(rec)
                print_result(rec)
                all_cases[(model_name, mode, layer)] = changed
                for condition, rows in (("baseline", baseline), (mode, changed)):
                    # Do not duplicate baseline in per_image for every layer.
                    if condition == "baseline":
                        continue
                    for row in rows:
                        per_image.append({"model": model_name, "layer": layer,
                                          "module": info["module"], "mode": condition,
                                          **row})
                del m
                gc.collect()

    cross_results: list[dict[str, Any]] = []
    if len(specs) == 2:
        m0, m1 = specs[0][0], specs[1][0]
        modes_and_layers = sorted(set((mode, idx) for model_name, mode, idx in all_cases
                                      if model_name == m0)
                                  & set((mode, idx) for model_name, mode, idx in all_cases
                                        if model_name == m1))
        for mode, layer in modes_and_layers:
            records = {}
            for model_name in (m0, m1):
                base = {r["image"]: r for r in all_bases[model_name]
                        if r["nme"] is not None}
                changed = {r["image"]: r for r in all_cases[(model_name, mode, layer)]
                           if r["nme"] is not None}
                records[model_name] = (base, changed)
            common = sorted(set.intersection(
                *[set(d) for pair in records.values() for d in pair]))
            cross = {"mode": mode, "layer": layer, "common_4way": len(common)}
            for model_name in (m0, m1):
                base, changed = records[model_name]
                b = ac.mean_of([base[s]["nme"] for s in common])
                a = ac.mean_of([changed[s]["nme"] for s in common])
                cross[model_name + "_base"] = b
                cross[model_name + "_ablated"] = a
                cross[model_name + "_delta"] = a - b if a is not None and b is not None else None
            d0, d1 = cross[m0 + "_delta"], cross[m1 + "_delta"]
            cross["delta_difference_model2_minus_model1"] = (d1 - d0) if d0 is not None and d1 is not None else None
            cross_results.append(cross)

    fields = ["model", "label", "layer", "module", "mode", "keep", "images",
              "baseline_hits", "ablated_hits", "retention", "boxes_per_image",
              "baseline_nme_all_hits", "ablated_nme_own_hits", "paired_images",
              "paired_baseline_nme", "paired_ablated_nme", "paired_delta_nme",
              "paired_delta_pct", "candidate", "seconds"]
    for k in range(ac.KEYPOINT_COUNT):
        fields += [f"k{k}_baseline_px", f"k{k}_ablated_px", f"k{k}_delta_px"]
    save_csv(out / "summary.csv", summary, fields)
    per_fields = ["model", "layer", "module", "mode", "image", "n_boxes", "matched",
                  "iou", "nme"] + [f"k{k}_px" for k in range(ac.KEYPOINT_COUNT)]
    save_csv(out / "per_image.csv", per_image, per_fields)
    save_csv(out / "layer_profile.csv", profile_rows,
             ["model", "layer", "module", "input_shape", "output_shape",
              "identity_eligible", "attenuate_eligible"])
    save_csv(out / "skipped.csv", skipped,
             ["model", "mode", "layer", "module", "input_shape", "output_shape", "reason"])
    if cross_results:
        save_csv(out / "cross_model_4way.csv", cross_results,
                 ["mode", "layer", "common_4way"] +
                 [f"{name}_{suffix}" for name in (specs[0][0], specs[1][0])
                  for suffix in ("base", "ablated", "delta")] +
                 ["delta_difference_model2_minus_model1"])
    with (out / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump({"dataset": str(root), "weights_dir": str(weight_dir),
                   "models": args.models, "layers": args.layers, "mode": args.mode,
                   "keep": args.keep, "alpha": args.alpha, "split": args.split, "conf": args.conf,
                   "imgsz": args.imgsz, "match_iou": args.match_iou,
                   "min_retention": args.min_retention,
                   "ultralytics_version": ac._LOCKED_VERSION},
                  f, ensure_ascii=False, indent=2)
    candidates = [r for r in summary if r.get("candidate")]
    candidates.sort(key=lambda r: r["paired_delta_nme"])
    print("\n=== Candidate improvements (high retention, same-frame NME improvement) ===")
    if not candidates:
        print("  No high-retention improvements found with this sweep.")
    for row in candidates:
        print_result(row)
    print(f"\nSaved results to: {out}")
    print("NOTE: An intervention changes an already-trained network; it does not prove a causal root cause or retrained gain.")


if __name__ == "__main__":
    run()
