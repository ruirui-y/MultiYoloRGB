"""层特征探针：抓取 4 个模型在关键层的输出，看深度对各段特征做了什么。

目的
----
前面已确认：head（框/点分支）的权重在 3ch / 4ch 之间几乎一样，
所以"点分支崩"的差异只能来自**它收到的那份特征**。本脚本就是去抓那份特征。

采样点（按语义对齐，层号不能跨模型直接比）
------------------------------------------
    层 9            SPPF 输出（backbone 末端）  ← v8 / v11 都有，可跨模型比
    层 10           C2PSA 输出                  ← 只有 v11 有
    head 的输入三层  P3 / P4 / P5               ← v8=[15,18,21]  v11=[16,19,22]
                                                  （脚本自动从 head.f 读取，不写死）

对比方式
--------
4 个模型：v8-RGB / v8-RGBD / 11-RGB / 11-RGBD
同一帧的图（3ch 读 rgb/*.jpg，4ch 读 rgbd/*.png），多张取平均。

用法
----
    cd /d H:\\YJJ\\Yolo_RGBD\\MultiYoloRGB
    .venv\\Scripts\\python.exe layer_probe.py
    .venv\\Scripts\\python.exe layer_probe.py --images 20

输出
----
    runs\\layer_probe\\layer_stats.csv   每个 模型 × 层 一行
    runs\\layer_probe\\run_config.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_HERE: Path = Path(__file__).resolve()
_REPO: Path = _HERE.parent
_SIBLING: Path = _REPO.parent / "yolov8_rgbd_detection"

# 项目根剥离（同其它脚本，防 vendored ultralytics 8.3.176 劫持）
_SIBLING_NORM: str = os.path.normcase(os.path.abspath(str(_SIBLING)))
sys.path[:] = [
    e for e in sys.path
    if e and os.path.normcase(os.path.abspath(e)) != _SIBLING_NORM
]

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

IMAGE_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp"})
DEFAULT_DATASET: Path = Path(
    r"H:\YJJ\Yolo_RGBD\Resource\session1_200146\dataset\hand_common"
)

# (权重名, 显示名, 通道数)
TARGET_MODELS: tuple[tuple[str, str, int], ...] = (
    ("yolov8n_rgb_3ch", "v8-RGB", 3),
    ("yolov8n_rgbd_4ch", "v8-RGBD", 4),
    ("yolo11n_rgb_3ch", "11-RGB", 3),
    ("yolo11n_rgbd_4ch", "11-RGBD", 4),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="层特征探针")
    p.add_argument("--dataset-root", default=str(DEFAULT_DATASET))
    p.add_argument("--weights-dir", default=str(_REPO / "train_weight"))
    p.add_argument("--images", type=int, default=10, help="取前 N 张取平均，默认 10")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--split", default="test")
    p.add_argument("--output-dir", default=str(_REPO / "runs" / "layer_probe"))
    return p.parse_args()


def collect_images(root: Path, split: str, channels: int, limit: int) -> list[Path]:
    sub = "rgb" if channels == 3 else "rgbd"
    directory = root / sub / "images" / split
    if not directory.is_dir():
        return []
    files = sorted(p for p in directory.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    return files[:limit] if limit > 0 else files


def tensor_stats(t: torch.Tensor) -> dict[str, float]:
    """一组统计量：整体幅度 + 通道间不均衡程度。"""
    x = t.detach().float()
    ch_mean = x.mean(dim=(0, 2, 3))          # 每个通道的均值
    return {
        "shape_c": float(x.shape[1]),
        "mean": float(x.mean().item()),
        "std": float(x.std().item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "abs_mean": float(x.abs().mean().item()),
        "ch_mean_std": float(ch_mean.std().item()),   # 通道之间是否不均衡
        "ch_mean_max": float(ch_mean.max().item()),
        "ch_mean_min": float(ch_mean.min().item()),
    }


def probe(model_path: Path, channels: int, images: list[Path],
          imgsz: int) -> tuple[list[int], dict[int, dict[str, Any]], dict[int, str]]:
    """跑一次模型，抓取目标层的输出统计。"""
    model = YOLO(str(model_path))
    seq = model.model.model
    has_c2psa = any(type(layer).__name__ == "C2PSA" for layer in seq)
    head_idx = len(seq) - 1
    head_f = [int(i) for i in seq[head_idx].f]

    targets: list[int] = [9]
    if has_c2psa:
        targets.append(10)
    targets += [i for i in head_f if i not in targets]

    captured: dict[int, torch.Tensor] = {}
    hooks = []

    def make_hook(index: int):
        def fn(module, inputs, output) -> None:
            out = output[0] if isinstance(output, (list, tuple)) else output
            if isinstance(out, torch.Tensor):
                captured[index] = out
        return fn

    for idx in targets:
        hooks.append(seq[idx].register_forward_hook(make_hook(idx)))

    per_image: dict[int, list[dict[str, float]]] = {i: [] for i in targets}
    for path in images:
        captured.clear()
        model.predict(str(path), imgsz=imgsz, conf=0.05, verbose=False,
                      device=None)
        for idx in targets:
            t = captured.get(idx)
            if t is not None:
                per_image[idx].append(tensor_stats(t))

    for hook in hooks:
        hook.remove()

    merged: dict[int, dict[str, Any]] = {}
    for idx in targets:
        rows = per_image[idx]
        if not rows:
            continue
        entry: dict[str, Any] = {}
        for key in rows[0]:
            entry[key] = float(np.mean([r[key] for r in rows]))
        entry["shape"] = "x".join(str(v) for v in
                                  (1, int(rows[0]["shape_c"]),
                                   int(captured[idx].shape[2]) if idx in captured else 0,
                                   int(captured[idx].shape[3]) if idx in captured else 0))
        merged[idx] = entry

    types = {i: type(seq[i]).__name__ for i in targets}
    return targets, merged, types


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root).expanduser()
    weights_dir = Path(args.weights_dir).expanduser()
    out_dir = Path(args.output_dir).expanduser()

    print("=" * 108)
    print("层特征探针：深度对各段特征的影响")
    print(f"数据集 : {root}")
    print(f"取图   : 每个模型前 {args.images} 张（同一帧，3ch 读 rgb / 4ch 读 rgbd）")
    print("=" * 108)

    all_results: dict[str, dict[str, Any]] = {}
    all_types: dict[str, dict[int, str]] = {}
    all_targets: dict[str, list[int]] = {}

    for weights_name, label, channels in TARGET_MODELS:
        path = weights_dir / f"{weights_name}.pt"
        if not path.is_file():
            print(f"[跳过] 权重不存在: {path}")
            continue
        images = collect_images(root, args.split, channels, args.images)
        if not images:
            print(f"[跳过] 找不到 {channels} 通道测试图")
            continue
        targets, merged, types = probe(path, channels, images, args.imgsz)
        all_results[label] = merged
        all_types[label] = types
        all_targets[label] = targets
        print(f"\n>>> {label}（{channels}ch，{len(images)} 张）  采样层 = {targets}")
        for idx in targets:
            if idx not in merged:
                continue
            e = merged[idx]
            print(f"    层{idx:<3d} {types[idx]:<8s} shape={e['shape']:<16s} "
                  f"mean={e['mean']:>9.4f}  std={e['std']:>8.4f}  "
                  f"abs_mean={e['abs_mean']:>8.4f}  min={e['min']:>9.3f}  max={e['max']:>9.3f}")

    if not all_results:
        print("[STOP] 没有可用结果")
        return

    # ---- 按语义分组对比 ----
    def find_label(keyword: str) -> list[str]:
        return [lb for lb in all_results if keyword in lb]

    print("\n" + "=" * 108)
    print("【对比 1】backbone 末端 · 层 9（SPPF 输出）")
    print("=" * 108)
    print(f"{'模型':10s}{'shape':>18s}{'mean':>11s}{'std':>10s}{'abs_mean':>11s}"
          f"{'min':>10s}{'max':>10s}{'通道间不均衡':>14s}")
    for label in ("v8-RGB", "v8-RGBD", "11-RGB", "11-RGBD"):
        e = all_results.get(label, {}).get(9)
        if not e:
            continue
        print(f"{label:10s}{e['shape']:>18s}{e['mean']:>11.4f}{e['std']:>10.4f}"
              f"{e['abs_mean']:>11.4f}{e['min']:>10.3f}{e['max']:>10.3f}"
              f"{e['ch_mean_std']:>14.4f}")

    if any(10 in all_results.get(lb, {}) for lb in all_results):
        print("\n" + "=" * 108)
        print("【对比 2】C2PSA（v11 独有）：层 9 → 层 10 的变换")
        print("=" * 108)
        print(f"{'模型':10s}{'层9 mean':>12s}{'层10 mean':>12s}{'Δmean':>12s}"
              f"{'层9 std':>11s}{'层10 std':>11s}{'Δstd':>11s}")
        for label in ("11-RGB", "11-RGBD"):
            a = all_results.get(label, {}).get(9)
            b = all_results.get(label, {}).get(10)
            if not a or not b:
                continue
            print(f"{label:10s}{a['mean']:>12.4f}{b['mean']:>12.4f}"
                  f"{b['mean'] - a['mean']:>+12.4f}"
                  f"{a['std']:>11.4f}{b['std']:>11.4f}"
                  f"{b['std'] - a['std']:>+11.4f}")

    print("\n" + "=" * 108)
    print("【对比 3】head 入口三层（P3 / P4 / P5）—— 这才是直接喂给点分支的特征")
    print("=" * 108)
    print(f"{'模型':10s}{'P3 shape':>16s}{'P3 abs_mean':>13s}"
          f"{'P4 shape':>16s}{'P4 abs_mean':>13s}"
          f"{'P5 shape':>16s}{'P5 abs_mean':>13s}")
    for label in ("v8-RGB", "v8-RGBD", "11-RGB", "11-RGBD"):
        res = all_results.get(label, {})
        targets = all_targets.get(label, [])
        head_layers = [i for i in targets if i >= 15]
        if len(head_layers) < 3:
            continue
        cells: list[str] = []
        for idx in head_layers[:3]:
            e = res.get(idx)
            if not e:
                cells.append(f"{'-':>16s}{'-':>13s}")
                continue
            cells.append(f"{e['shape']:>16s}{e['abs_mean']:>13.4f}")
        print(f"{label:10s}" + "".join(cells))

    # ---- 落盘 ----
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for label, res in all_results.items():
        for idx, e in sorted(res.items()):
            rows.append({
                "model": label,
                "layer": idx,
                "module": all_types[label].get(idx, ""),
                **{k: v for k, v in e.items()},
            })
    fields = ["model", "layer", "module", "shape", "mean", "std", "min", "max",
              "abs_mean", "ch_mean_std", "ch_mean_max", "ch_mean_min"]
    with (out_dir / "layer_stats.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    (out_dir / "run_config.json").write_text(
        json.dumps({
            "dataset_root": str(root),
            "split": args.split,
            "images_per_model": args.images,
            "imgsz": args.imgsz,
            "models": [{"name": n, "label": lb, "channels": c}
                       for n, lb, c in TARGET_MODELS],
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\n结果文件:\n  {out_dir / 'layer_stats.csv'}")


if __name__ == "__main__":
    main()
