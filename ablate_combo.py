"""YOLO11 RGB/RGBD multi-layer attenuation experiments (no retraining).

Place beside `ablate_c2psa.py` and `ablate_layers.py`.

Cases (independent, fresh checkpoint per case):
    original     : no attenuation
    L1           : layer 1 *= 0.50
    L1_L9        : layer 1 *= 0.50, layer 9 *= 0.70
    L1_L10       : layer 1 *= 0.50, layer 10 *= 0.80
    L1_L9_L10    : all three layer outputs attenuated in one forward pass

All NME comparisons use matched images from BOTH relevant conditions.
This changes outputs of the trained network; it is not structural removal.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path
from typing import Any

import ablate_c2psa as ac
import ablate_layers as al


CASES: dict[str, tuple[int, ...]] = {
    "L1": (1,),
    "L1_L9": (1, 9),
    "L1_L10": (1, 10),
    "L1_L9_L10": (1, 9, 10),
}
EXPECTED_MODULES: dict[int, str] = {1: "Conv", 9: "SPPF", 10: "C2PSA"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO11 multi-layer attenuation combos")
    p.add_argument("--dataset-root", default=str(ac._DEFAULT_DATASET))
    p.add_argument("--weights-dir", default=str(ac._REPO / "train_weight"))
    p.add_argument("--output-dir", default=str(ac._REPO / "runs" / "ablate_combo"))
    p.add_argument("--models", nargs="+", default=[m[0] for m in ac.DEFAULT_MODELS],
                   help="For RGBD only: --models yolo11n_rgbd_4ch")
    p.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES),
                   help="Which combined conditions to evaluate")
    p.add_argument("--keep-l1", type=float, default=0.50)
    p.add_argument("--keep-l9", type=float, default=0.70)
    p.add_argument("--keep-l10", type=float, default=0.80)
    p.add_argument("--split", default="test")
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--match-iou", type=float, default=0.50)
    return p.parse_args()


def matches(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {r["image"]: r for r in rows if r["nme"] is not None}


def paired_metrics(ref_rows: list[dict[str, Any]],
                   test_rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    ref = matches(ref_rows)
    test = matches(test_rows)
    shared = sorted(ref.keys() & test.keys())
    b = ac.mean_of([ref[s]["nme"] for s in shared])
    a = ac.mean_of([test[s]["nme"] for s in shared])
    delta = a - b if a is not None and b is not None else None
    result: dict[str, Any] = {
        prefix + "_paired": len(shared),
        prefix + "_ref_nme": b,
        prefix + "_case_nme": a,
        prefix + "_delta_nme": delta,
        prefix + "_delta_pct": (100.0 * delta / b if delta is not None and b else None),
    }
    for k in range(ac.KEYPOINT_COUNT):
        field = f"k{k}_px"
        pairs = [(ref[s][field], test[s][field]) for s in shared
                 if ref[s][field] is not None and test[s][field] is not None]
        result[f"{prefix}_k{k}_delta_px"] = ac.mean_of([y - x for x, y in pairs])
    return result


def validate_modules(modules: Any, selected: tuple[int, ...]) -> None:
    for index in selected:
        if index >= len(modules):
            raise RuntimeError(f"Layer {index} not present; model has {len(modules)} layers")
        actual = type(modules[index]).__name__
        expected = EXPECTED_MODULES[index]
        if actual != expected:
            raise RuntimeError(f"Layer {index} expected {expected}, found {actual}; "
                               "refusing to attenuate a different architecture")


def apply_combo(model: Any, selected: tuple[int, ...], keeps: dict[int, float]) -> None:
    """Wrap ALL requested layers before the model's first predict call."""
    modules = model.model.model
    validate_modules(modules, selected)
    for index in selected:
        original = modules[index]
        modules[index] = al.LayerAttenuator(original, keeps[index])
    for index in selected:
        wrapped = modules[index]
        assert isinstance(wrapped, al.LayerAttenuator)
        assert wrapped.keep == keeps[index]


def save_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def nfmt(n: Any) -> str:
    return "-" if n is None else f"{n:.6f}"


def pctfmt(n: Any) -> str:
    return "-" if n is None else f"{n:+.2f}%"


def main() -> None:
    args = parse_args()
    keeps = {1: args.keep_l1, 9: args.keep_l9, 10: args.keep_l10}
    for index, value in keeps.items():
        if not 0 <= value <= 1:
            raise ValueError(f"--keep-l{index} must be in [0, 1], got {value}")
    specs = al.extract_model_specs(args.models)
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.dataset_root).expanduser()
    weights_dir = Path(args.weights_dir).expanduser()

    rgb_map = ac.stem_map(root / "rgb" / "images" / args.split)
    rgbd_map = ac.stem_map(root / "rgbd" / "images" / args.split)
    labels = root / "rgb" / "labels" / args.split
    if not labels.is_dir():
        labels = root / "rgbd" / "labels" / args.split
    if not labels.is_dir():
        raise FileNotFoundError(f"Label directory not found: {labels}")
    stems = sorted(s for s in (rgb_map.keys() & rgbd_map.keys())
                   if (labels / (s + ".txt")).is_file())
    if not stems:
        raise RuntimeError("No RGB/RGBD same-stem images with labels")

    print("=" * 96)
    print("MULTI-LAYER attenuation (simultaneous interventions; fresh weights per condition)")
    print(f"dataset={root}, frames={len(stems)}, conf={args.conf}, "
          f"imgsz={args.imgsz}, match_iou={args.match_iou}")
    print(f"L1={keeps[1]:.2f}, L9={keeps[9]:.2f}, L10={keeps[10]:.2f}")
    print("Cases: original, " + ", ".join(dict.fromkeys(["L1", *args.cases])))
    print("=" * 96)

    summary: list[dict[str, Any]] = []
    per_image: list[dict[str, Any]] = []
    model_weights: dict[str, str] = {}

    for model_name, label, channels in specs:
        weight_file = weights_dir / (model_name + ".pt")
        if not weight_file.is_file():
            raise FileNotFoundError(f"Weights missing: {weight_file}")
        model_weights[model_name] = str(weight_file)
        items = al.load_items(root, args.split, channels, stems)
        print(f"\n=== {label} ({channels}ch) ===")
        tested: dict[str, list[dict[str, Any]]] = {}
        selected_cases = ["original", *dict.fromkeys(["L1", *args.cases])]

        for name in selected_cases:
            model = ac.YOLO(str(weight_file))  # Fresh copy BEFORE predict.
            if name != "original":
                apply_combo(model, CASES[name], keeps)
            start = time.perf_counter()
            try:
                result_rows = ac.run_condition(model, items, args.conf, args.imgsz,
                                               args.match_iou)
            finally:
                del model
                gc.collect()
            elapsed = time.perf_counter() - start
            tested[name] = result_rows
            hit_count = len(matches(result_rows))
            nme = ac.mean_of([r["nme"] for r in result_rows])
            print(f"  {name:<12} hits={hit_count:3}/{len(result_rows)} "
                  f"NME(matched only)={nfmt(nme)}  time={elapsed:.1f}s")

            record: dict[str, Any] = {
                "model": model_name,
                "label": label,
                "case": name,
                "layers": ",".join(str(i) for i in CASES.get(name, ())),
                "keep_l1": keeps[1] if name != "original" else 1.0,
                "keep_l9": keeps[9] if 9 in CASES.get(name, ()) else 1.0,
                "keep_l10": keeps[10] if 10 in CASES.get(name, ()) else 1.0,
                "images": len(result_rows),
                "hits": hit_count,
                "boxes_per_image": ac.mean_of([r["n_boxes"] for r in result_rows]),
                "nme_matched_only": nme,
                "seconds": round(elapsed, 3),
            }
            summary.append(record)
            for r in result_rows:
                per_image.append({"model": model_name, "case": name, **r})

        baseline = tested["original"]
        l1 = tested["L1"]
        for record in summary:
            if record["model"] != model_name:
                continue
            current = tested[record["case"]]
            record.update(paired_metrics(baseline, current, "vs_original"))
            record.update(paired_metrics(l1, current, "vs_l1"))

        print("\n  Fair matched-frame comparison (against L1 only):")
        print("  case          hits  paired    L1 NME    case NME   delta")
        for record in summary:
            if record["model"] != model_name or record["case"] == "original":
                continue
            print(f"  {record['case']:<12} {record['hits']:3}/{record['images']} "
                  f"{record['vs_l1_paired']:6} "
                  f"{nfmt(record['vs_l1_ref_nme']):>10} "
                  f"{nfmt(record['vs_l1_case_nme']):>11} "
                  f"{pctfmt(record['vs_l1_delta_pct']):>9}")

    fields = ["model", "label", "case", "layers", "keep_l1", "keep_l9", "keep_l10",
              "images", "hits", "boxes_per_image", "nme_matched_only", "seconds"]
    for prefix in ("vs_original", "vs_l1"):
        fields.extend([f"{prefix}_paired", f"{prefix}_ref_nme", f"{prefix}_case_nme",
                       f"{prefix}_delta_nme", f"{prefix}_delta_pct"])
        fields.extend([f"{prefix}_k{k}_delta_px" for k in range(ac.KEYPOINT_COUNT)])
    save_csv(out / "summary.csv", summary, fields)
    row_fields = ["model", "case", "image", "n_boxes", "matched", "iou", "nme"]
    row_fields += [f"k{k}_px" for k in range(ac.KEYPOINT_COUNT)]
    save_csv(out / "per_image.csv", per_image, row_fields)
    (out / "run_config.json").write_text(json.dumps({
        "dataset_root": str(root), "split": args.split, "count": len(stems),
        "weights": model_weights, "cases": ["original", *dict.fromkeys(["L1", *args.cases])],
        "keeps": keeps, "conf": args.conf, "imgsz": args.imgsz,
        "match_iou": args.match_iou, "ultralytics": ac._LOCKED_VERSION,
        "note": "All combinations are simultaneous within a case; "
                "NME pair comparisons use the intersection of matched image names."
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nSaved to: {out}")
    print("summary.csv | per_image.csv | run_config.json")


if __name__ == "__main__":
    main()
