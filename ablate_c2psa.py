"""C2PSA 结构消融（零训练）：把 yolo11 的 C2PSA 层"跳过"，直接推理对比 NME。

原理
----
C2PSA 的输入输出通道相同（256 → 256），因此可以把它旁路掉：
在它的 forward hook 里直接返回输入，效果等价于把该层换成 Identity。
通道数不变，后续层（含层 21 的 `Concat [-1, 10]`）不受影响，权重可直接加载。

⚠️ 这是"截断"实验：网络是按"有 C2PSA"训练出来的，跳过它会让后续层收到分布外输入。
   它只能回答「C2PSA 在当前这个网络里贡献多大」，
   **不能**回答「当初训练时就没有它会怎样」（后者需改 yaml 重新训练）。

对比设计
--------
    yolo11n_rgb_3ch   /  yolo11n_rgbd_4ch
    ×  with_c2psa（原样） / without_c2psa（旁路）

用法
----
    cd /d H:\\YJJ\\Yolo_RGBD\\MultiYoloRGB
    .venv\\Scripts\\python.exe ablate_c2psa.py
    .venv\\Scripts\\python.exe ablate_c2psa.py --models yolo11n_rgbd_4ch

输出
----
    runs\\ablate_c2psa\\summary.csv
    runs\\ablate_c2psa\\per_image.csv
    runs\\ablate_c2psa\\run_config.json
"""
from __future__ import annotations
import paths


import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch.nn as nn

_HERE: Path = Path(__file__).resolve()
_REPO: Path = _HERE.parent
_SIBLING: Path = _REPO.parent / "yolov8_rgbd_detection"
_EVAL_SCRIPT: Path = _SIBLING / "YJJ_Pose_Scripts" / "04_eval" / "eval_rgb_vs_rgbd_nme.py"
_DEFAULT_DATASET: Path = Path(
    paths.DEFAULT_DATASET
)

# 剥离项目根（防 vendored ultralytics 8.3.176 劫持）
_SIBLING_NORM: str = os.path.normcase(os.path.abspath(str(_SIBLING)))
sys.path[:] = [
    e for e in sys.path
    if e and os.path.normcase(os.path.abspath(e)) != _SIBLING_NORM
]

import ultralytics  # noqa: E402

_LOCKED_VERSION: str = ultralytics.__version__

if not _EVAL_SCRIPT.is_file():
    raise FileNotFoundError(f"第一阶段评估脚本不存在: {_EVAL_SCRIPT}")

sys.path.insert(0, str(_EVAL_SCRIPT.parent))
import eval_rgb_vs_rgbd_nme as ev  # noqa: E402

if ultralytics.__version__ != _LOCKED_VERSION:
    raise RuntimeError(
        f"ultralytics 被换掉了: {_LOCKED_VERSION} -> {ultralytics.__version__}"
    )

sys.path[:] = [
    e for e in sys.path
    if not (e and os.path.normcase(os.path.abspath(e)) == _SIBLING_NORM)
]

from ultralytics import YOLO  # noqa: E402


IMAGE_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp"})
KEYPOINT_COUNT: int = 7
CONDITIONS: tuple[str, ...] = ("with_c2psa", "without_c2psa")
# (权重名, 显示名, 通道数)
DEFAULT_MODELS: tuple[tuple[str, str, int], ...] = (
    ("yolo11n_rgb_3ch", "11-RGB", 3),
    ("yolo11n_rgbd_4ch", "11-RGBD", 4),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="C2PSA 结构消融（零训练）")
    p.add_argument("--dataset-root", default=str(_DEFAULT_DATASET))
    p.add_argument("--weights-dir", default=str(_REPO / "train_weight"))
    p.add_argument("--models", nargs="+", default=None,
                   help="只跑指定权重名（不含 .pt），默认跑 11-RGB 与 11-RGBD")
    p.add_argument("--split", default="test")
    p.add_argument("--match-iou", type=float, default=0.50)
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--output-dir", default=str(_REPO / "runs" / "ablate_c2psa"))
    return p.parse_args()


def stem_map(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        return {}
    return {
        p.stem: p
        for p in sorted(directory.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    }


def read_image(path: Path, channels: int) -> np.ndarray:
    flag = cv2.IMREAD_UNCHANGED if channels == 4 else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)
    if image is None:
        raise FileNotFoundError(f"图像读取失败: {path}")
    if channels == 4 and (image.ndim != 3 or image.shape[2] != 4):
        raise ValueError(f"图像不是 4 通道: {path} shape={image.shape}")
    return image


def mean_of(values: list[Any]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return float(np.mean(clean)) if clean else None


def find_c2psa_layers(model: Any) -> list[int]:
    return [i for i, layer in enumerate(model.model.model)
            if type(layer).__name__ == "C2PSA"]


class _SkipLayer(nn.Module):
    """占位层：原样返回输入，用于把某一层"旁路"掉。

    必须手动带上 .f / .i —— ultralytics 的 forward 依赖这两个属性做拓扑排序与特征缓存，
    缺了会直接报错。
    """

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.f = source.f
        self.i = source.i
        self.type = "SkipLayer"

    def forward(self, x):  # noqa: ANN001, ANN201
        return x


def bypass_layers(model: Any, layers: list[int]) -> None:
    """把指定层替换成 SkipLayer（原样返回输入）。

    ⚠️ 必须在**该 YOLO 实例首次 predict 之前**完成 —— ultralytics 建 predictor 时会
    复制一份模型（实测 `predictor.model.model is not m.model`），之后再改 `m.model`
    对推理无效。因此每个条件都要重新 `YOLO(path)`。
    """
    for idx in layers:
        model.model.model[idx] = _SkipLayer(model.model.model[idx])


def run_condition(model: Any,
                  items: list[tuple[str, np.ndarray, dict[str, Any]]],
                  conf: float, imgsz: int, match_iou: float
                  ) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stem, image, gt in items:
        result = ev.run_pose_model(model, image, conf, imgsz)
        n_boxes = 0 if result.boxes is None else len(result.boxes)
        pred = ev.find_best_prediction(result, gt["box"], match_iou)
        nme: float | None = None
        kpt_px: list[float | None] = [None] * KEYPOINT_COUNT
        if pred["matched"]:
            try:
                nme_result = ev.calculate_nme(gt["box"], gt["keypoints"],
                                              pred["keypoints"])
                nme = float(nme_result["nme"])
                errors = ev.extract_keypoint_pixel_errors(nme_result, KEYPOINT_COUNT)
                kpt_px = [None if v is None else float(v) for v in errors]
            except ValueError:
                nme = None
        rows.append({
            "image": stem,
            "n_boxes": n_boxes,
            "matched": bool(pred["matched"]),
            "iou": round(float(pred["iou"]), 6),
            "nme": (None if nme is None else round(nme, 6)),
            **{f"k{k}_px": (None if kpt_px[k] is None else round(kpt_px[k], 4))
               for k in range(KEYPOINT_COUNT)},
        })

    return rows


def fmt(value: float | None, digits: int = 6) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root).expanduser()
    weights_dir = Path(args.weights_dir).expanduser()
    out_dir = Path(args.output_dir).expanduser()

    targets = DEFAULT_MODELS
    if args.models:
        wanted = {m if m.endswith(".pt") else f"{m}.pt" for m in args.models}
        targets = tuple(t for t in DEFAULT_MODELS if f"{t[0]}.pt" in wanted)
        if not targets:
            print(f"[STOP] --models 未匹配到任何目标: {args.models}")
            return

    print("=" * 104)
    print("C2PSA 结构消融（零训练）：把 C2PSA 旁路，直接推理对比")
    print(f"数据集 : {root}")
    print(f"口径   : conf={args.conf}   imgsz={args.imgsz}   match_iou={args.match_iou}")
    print(f"实现   : {_EVAL_SCRIPT.name}（ultralytics {_LOCKED_VERSION}）")
    print("=" * 104)

    rgb_map = stem_map(root / "rgb" / "images" / args.split)
    rgbd_map = stem_map(root / "rgbd" / "images" / args.split)
    label_dir = root / "rgb" / "labels" / args.split
    if not label_dir.is_dir():
        label_dir = root / "rgbd" / "labels" / args.split
    if not label_dir.is_dir():
        print("[STOP] 找不到标签目录")
        return

    summary: list[dict[str, Any]] = []
    per_image: list[dict[str, Any]] = []

    for weights_name, label, channels in targets:
        path = weights_dir / f"{weights_name}.pt"
        if not path.is_file():
            print(f"[跳过] 权重不存在: {path}")
            continue
        source_map = rgb_map if channels == 3 else rgbd_map
        stems = [s for s in sorted(source_map) if (label_dir / f"{s}.txt").is_file()]
        if not stems:
            print(f"[跳过] {label}: 找不到可用图片")
            continue
        probe = YOLO(str(path))
        layers = find_c2psa_layers(probe)
        if not layers:
            print(f"[跳过] {label}: 该模型没有 C2PSA 层（可能不是 yolo11）")
            continue
        del probe

        items: list[tuple[str, np.ndarray, dict[str, Any]]] = []
        for stem in stems:
            image = read_image(source_map[stem], channels)
            height, width = image.shape[:2]
            gt = ev.load_pose_label(label_dir / f"{stem}.txt", width, height)
            items.append((stem, image, gt))

        print(f"\n>>> {label}（{channels}ch，{len(items)} 张）  C2PSA 在层 {layers}")
        for condition in CONDITIONS:
            # 每个条件都重新加载权重：ultralytics 建 predictor 时会复制模型，
            # 复用同一个 YOLO 实例改结构对推理无效（实测）
            model = YOLO(str(path))
            bypass = condition == "without_c2psa"
            if bypass:
                bypass_layers(model, layers)
            started = time.perf_counter()
            rows = run_condition(model, items, args.conf, args.imgsz, args.match_iou)
            nme = mean_of([r["nme"] for r in rows])
            matched = sum(1 for r in rows if r["matched"])
            boxes = mean_of([r["n_boxes"] for r in rows])
            summary.append({
                "model": label,
                "channels": channels,
                "condition": condition,
                "n_images": len(rows),
                "matched": matched,
                "nme": nme,
                "boxes_mean": boxes,
                "c2psa_bypassed": bypass,
                **{f"k{k}_px": mean_of([r[f"k{k}_px"] for r in rows])
                   for k in range(KEYPOINT_COUNT)},
                "elapsed_s": round(time.perf_counter() - started, 2),
            })
            for row in rows:
                per_image.append({
                    "model": label, "condition": condition, "image": row["image"],
                    "n_boxes": row["n_boxes"], "matched": row["matched"],
                    "iou": row["iou"], "nme": row["nme"],
                    **{f"k{k}_px": row[f"k{k}_px"] for k in range(KEYPOINT_COUNT)},
                })
            print(f"    {condition:16s} 命中 {matched}/{len(rows)}  "
                  f"框数 {boxes:.2f}  NME {fmt(nme)}  [{time.perf_counter() - started:.1f}s]")
            if bypass:
                actual = type(model.model.model[layers[0]]).__name__
                print(f"      [自检] 本次推理的模型里，层 {layers[0]} 类型 = {actual}")

    if not summary:
        print("[STOP] 没有可用结果")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["model", "channels", "condition", "n_images", "matched", "nme",
              "boxes_mean", "c2psa_bypassed", "elapsed_s"] + \
        [f"k{k}_px" for k in range(KEYPOINT_COUNT)]
    with (out_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)

    with (out_dir / "per_image.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "condition", "image", "n_boxes",
                           "matched", "iou", "nme"]
            + [f"k{k}_px" for k in range(KEYPOINT_COUNT)])
        writer.writeheader()
        writer.writerows(per_image)

    (out_dir / "run_config.json").write_text(
        json.dumps({
            "dataset_root": str(root), "split": args.split,
            "conf": args.conf, "imgsz": args.imgsz, "match_iou": args.match_iou,
            "conditions": list(CONDITIONS),
            "models": [{"name": n, "label": lb, "channels": c}
                       for n, lb, c in targets],
            "ultralytics": _LOCKED_VERSION, "eval_impl": _EVAL_SCRIPT.name,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ---- 终端对比表 ----
    print("\n" + "=" * 104)
    print("NME 对比：原样 vs 旁路 C2PSA")
    print("=" * 104)
    print("（NME 只在该条件下「命中」的图上取平均 —— 务必同时看命中数与框数）")
    print(f"{'模型':10s}{'条件':18s}{'命中':>10s}{'框数':>8s}{'NME':>12s}{'变化':>11s}")
    for weights_name, label, channels in targets:
        rows = [r for r in summary if r["model"] == label]
        if len(rows) < 2:
            continue
        base = rows[0]["nme"]
        for row in rows:
            if row["condition"] == "with_c2psa":
                delta = "-"
            elif base and row["nme"] is not None:
                pct = (row["nme"] - base) / base * 100.0
                delta = f"{pct:+.1f}%"
            else:
                delta = "-"
            print(f"{label:10s}{row['condition']:18s}"
                  f"{str(row['matched']) + '/' + str(row['n_images']):>10s}"
                  f"{row['boxes_mean']:>8.2f}{fmt(row['nme']):>12s}{delta:>11s}")

    # ---- 公平口径：只在「两组都命中」的同一批图上比 ----
    def common_pair(label: str) -> tuple[list[dict[str, Any]],
                                         list[dict[str, Any]], list[str]]:
        own = [r for r in per_image if r["model"] == label]
        base = {r["image"]: r for r in own
                if r["condition"] == "with_c2psa" and r["matched"]}
        abl = {r["image"]: r for r in own
               if r["condition"] == "without_c2psa" and r["matched"]}
        keys = sorted(set(base) & set(abl))
        return [base[k] for k in keys], [abl[k] for k in keys], keys

    print("\n" + "=" * 104)
    print("公平口径：只在「原样与旁路都命中」的同一批图上比较")
    print("=" * 104)
    print("（上面的表两侧分母不同、不可直接比；本表把两侧限制到同一批图）")
    print(f"{'模型':10s}{'共同集':>8s}{'原样NME':>13s}{'旁路NME':>13s}{'公平Δ':>11s}")
    for weights_name, label, channels in targets:
        base_rows, abl_rows, keys = common_pair(label)
        if not keys:
            print(f"{label:10s}{0:>8d}{'-':>13s}{'-':>13s}{'-':>11s}")
            continue
        b = mean_of([r["nme"] for r in base_rows])
        a = mean_of([r["nme"] for r in abl_rows])
        pct = "-" if (not b or a is None) else f"{(a - b) / b * 100:+.1f}%"
        print(f"{label:10s}{len(keys):>8d}{fmt(b):>13s}{fmt(a):>13s}{pct:>11s}")

    print("\nK0~K6 公平口径（px，同一批图）")
    print("=" * 104)
    print(f"{'模型':10s}{'行':16s}"
          + "".join(f"{'K' + str(k):>9s}" for k in range(KEYPOINT_COUNT)))
    for weights_name, label, channels in targets:
        base_rows, abl_rows, keys = common_pair(label)
        if not keys:
            continue
        b_cells = ""
        a_cells = ""
        for k in range(KEYPOINT_COUNT):
            bv = mean_of([r[f"k{k}_px"] for r in base_rows])
            av = mean_of([r[f"k{k}_px"] for r in abl_rows])
            b_cells += f"{'-':>9s}" if bv is None else f"{bv:>9.2f}"
            a_cells += f"{'-':>9s}" if av is None else f"{av:>9.2f}"
        print(f"{label:10s}{'原样':16s}{b_cells}")
        print(f"{label:10s}{'旁路':16s}{a_cells}")

    # ---- K0~K6 逐点 ----
    print("\n" + "=" * 104)
    print("K0~K6 平均像素误差（px；只在该条件下命中的图上取平均）")
    print("=" * 104)
    print(f"{'模型':10s}{'条件':16s}"
          + "".join(f"{'K' + str(k):>9s}" for k in range(KEYPOINT_COUNT)))
    for weights_name, label, channels in targets:
        for row in [r for r in summary if r["model"] == label]:
            cells = ""
            for k in range(KEYPOINT_COUNT):
                value = row[f"k{k}_px"]
                cells += f"{'-':>9s}" if value is None else f"{value:>9.2f}"
            print(f"{label:10s}{row['condition']:16s}{cells}")

    print("\nK0~K6 变化（旁路 − 原样，px；正数 = 误差变大 = 变差）")
    print("=" * 104)
    print(f"{'模型':10s}{'Δ':16s}"
          + "".join(f"{'K' + str(k):>9s}" for k in range(KEYPOINT_COUNT)))
    for weights_name, label, channels in targets:
        rows = [r for r in summary if r["model"] == label]
        if len(rows) < 2:
            continue
        base, now = rows[0], rows[1]
        cells = ""
        for k in range(KEYPOINT_COUNT):
            before, after = base[f"k{k}_px"], now[f"k{k}_px"]
            cells += (f"{'-':>9s}" if (before is None or after is None)
                      else f"{after - before:>+9.2f}")
        print(f"{label:10s}{'Δ':16s}{cells}")

    print("\n[判据]")
    print("（若命中数骤降，先看上方「公平口径」表 —— 那才是同一批图上的可比数）")
    for weights_name, label, channels in targets:
        rows = [r for r in summary if r["model"] == label]
        if len(rows) < 2:
            continue
        base, now = rows[0], rows[1]
        total = base["n_images"]
        drop = base["matched"] - now["matched"]
        print(f"  {label}")
        print(f"      命中 {base['matched']}/{total} -> {now['matched']}/{total}"
              f"（{'减少' if drop > 0 else '未减少'} {abs(drop)} 张）"
              f"   框数 {base['boxes_mean']:.2f} -> {now['boxes_mean']:.2f}")
        collapsed = now["matched"] < base["matched"] * 0.9
        if collapsed:
            print("      → 【检出崩溃】旁路后大量图片找不到目标，"
                  "上方「自己的命中集」NME 不可比")

        base_rows, abl_rows, keys = common_pair(label)
        if not keys:
            print("      → 无共同命中图，无法给出公平结论")
            continue
        b = mean_of([r["nme"] for r in base_rows])
        a = mean_of([r["nme"] for r in abl_rows])
        if not b or a is None:
            print("      → 公平口径 NME 数据不足")
            continue
        pct = (a - b) / b * 100.0
        tag = "（检出已崩溃）" if collapsed else ""
        if pct < -5:
            verdict = "旁路后明显变好 -> C2PSA 对点分支是负担"
        elif pct > 5:
            verdict = "旁路后明显变差 -> C2PSA 对点分支仍有益"
        else:
            verdict = "基本不变 -> C2PSA 对点分支不是关键因素"
        print(f"      公平口径（共同集 {len(keys)} 张）NME {pct:+.1f}%   "
              f"{verdict}{tag}")

    print(f"\n结果文件:\n  {out_dir / 'summary.csv'}\n  {out_dir / 'per_image.csv'}")


if __name__ == "__main__":
    main()
