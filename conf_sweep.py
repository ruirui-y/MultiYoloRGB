"""conf 阈值 × 选框规则 二维扫描：同一批推理结果，两种"选哪个框"规则各算一遍。

背景
----
项目现在的规则是「取 IoU 最大的预测框」（`find_best_prediction`）。
但在框很多的模型上（如 yolo11n_rgbd_4ch 每图 ~176 框），这条规则可能挑中
"恰好与 GT 重叠很高、但关键点是垃圾"的噪声框，从而把 NME 拉高。

本脚本在**每个 conf 阈值**下，对同一批推理结果同时套用两套规则：
    iou  : 取 IoU 最大的框（现用规则，`ev.find_best_prediction`）
    conf : 取 conf 最大的框（实际部署时会信的那个框）
对比两者在同一阈值下的 NME，即可判断"噪声框是否干扰了结果"。

口径
----
与 eval_rgb_vs_rgbd_nme.py 完全同源：
    推理 : ev.run_pose_model（imgsz=640 / rect=False / augment=False，仅 conf 变化）
    框   : ev.calculate_iou
    命中 : IoU >= match-iou
    NME  : ev.calculate_nme（GT 框对角线归一化，只统计 visibility > 0 的点）

NME 只在该阈值下"命中的图"上取平均，并同时报出命中数，避免分母不一致。

用法
----
    cd /d H:\\YJJ\\Yolo_RGBD\\MultiYoloRGB
    .venv\\Scripts\\python.exe conf_sweep.py

    # 只跑 0.001 与 0.05 两档
    .venv\\Scripts\\python.exe conf_sweep.py --confs 0.001 0.05

    # 只跑某几个权重
    .venv\\Scripts\\python.exe conf_sweep.py --weights train_weight\\yolo11n_rgbd_4ch.pt train_weight\\yolov8n_rgbd_4ch.pt

输出
----
    runs\\conf_sweep\\summary.csv     每个 权重 × 阈值 × 规则 一行
    runs\\conf_sweep\\per_image.csv   逐图明细（含两套规则各自的结果）
    runs\\conf_sweep\\run_config.json
"""
from __future__ import annotations

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

_HERE: Path = Path(__file__).resolve()
_REPO: Path = _HERE.parent
_SIBLING: Path = _REPO.parent / "yolov8_rgbd_detection"
_EVAL_SCRIPT: Path = _SIBLING / "YJJ_Pose_Scripts" / "04_eval" / "eval_rgb_vs_rgbd_nme.py"
_DEFAULT_DATASET: Path = Path(
    r"H:\YJJ\Yolo_RGBD\Resource\session1_200146\dataset\hand_common"
)

# 先剥离"项目根"再 import（项目根里有 vendored ultralytics 8.3.176，会顶掉 venv 的 8.4.175）
_SIBLING_NORM: str = os.path.normcase(os.path.abspath(str(_SIBLING)))
sys.path[:] = [
    entry for entry in sys.path
    if entry and os.path.normcase(os.path.abspath(entry)) != _SIBLING_NORM
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
    entry for entry in sys.path
    if not (entry and os.path.normcase(os.path.abspath(entry)) == _SIBLING_NORM)
]

from ultralytics import YOLO  # noqa: E402


IMAGE_EXTS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
)
KEYPOINT_COUNT: int = 7
DEFAULT_CONFS: tuple[float, ...] = (0.001, 0.005, 0.01, 0.05, 0.10, 0.25)
ALL_RULES: tuple[str, ...] = ("iou", "conf")
RULE_LABELS: dict[str, str] = {
    "iou": "取 IoU 最大（现用）",
    "conf": "取 conf 最大",
}
HIGHLIGHT_CONF: float = 0.05


# ============================================================
# 两套选框规则
# ============================================================
def select_by_iou(result: Any, gt_box: np.ndarray, match_iou: float) -> dict[str, Any]:
    """取 IoU 最大的框（= 项目现有规则，直接复用第一阶段实现）。"""
    pred = ev.find_best_prediction(result, gt_box, match_iou)
    return {
        "matched": bool(pred["matched"]),
        "index": pred["index"],
        "iou": float(pred["iou"]),
        "confidence": pred["confidence"],
        "keypoints": pred["keypoints"],
        "n_boxes": 0 if result.boxes is None else len(result.boxes),
    }


def select_by_conf(result: Any, gt_box: np.ndarray, match_iou: float) -> dict[str, Any]:
    """取 conf 最大的框（实际部署时模型自己最信的那个）。

    与 select_by_iou 的唯一区别是"挑哪一个框"，命中判定与 NME 口径完全一致。
    """
    output: dict[str, Any] = {
        "matched": False, "index": None, "iou": 0.0,
        "confidence": None, "keypoints": None, "n_boxes": 0,
    }
    if result.boxes is None or len(result.boxes) == 0:
        return output
    if result.keypoints is None or len(result.keypoints) == 0:
        return output

    pred_boxes = result.boxes.xyxy.cpu().numpy().astype(np.float64)
    output["n_boxes"] = len(pred_boxes)
    if result.boxes.conf is not None:
        confs = result.boxes.conf.cpu().numpy().astype(np.float64)
    else:
        confs = np.zeros(len(pred_boxes), dtype=np.float64)

    index = int(np.argmax(confs))
    output["index"] = index
    output["confidence"] = float(confs[index])
    output["iou"] = float(ev.calculate_iou(gt_box, pred_boxes[index]))

    if output["iou"] < match_iou:
        return output

    output["matched"] = True
    output["keypoints"] = (
        result.keypoints.data[index].cpu().numpy().astype(np.float64)
    )
    return output


SELECTORS: dict[str, Any] = {
    "iou": select_by_iou,
    "conf": select_by_conf,
}


# ============================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="conf 阈值 × 选框规则 二维扫描")
    p.add_argument("--dataset-root", default=str(_DEFAULT_DATASET))
    p.add_argument("--weights-dir", default=str(_REPO / "train_weight"))
    p.add_argument("--weights", nargs="+", default=None)
    p.add_argument("--confs", nargs="+", type=float, default=list(DEFAULT_CONFS))
    p.add_argument("--rules", nargs="+", default=list(ALL_RULES),
                   choices=list(ALL_RULES))
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--match-iou", type=float, default=0.50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--output-dir", default=str(_REPO / "runs" / "conf_sweep"))
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


def median_of(values: list[Any]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return float(np.median(clean)) if clean else None


def fmt_conf(value: float) -> str:
    return f"{value:g}"


def fmt_nme(value: float | None, digits: int = 6) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root).expanduser()
    out_dir = Path(args.output_dir).expanduser()
    confs: list[float] = list(dict.fromkeys(args.confs))
    rules: list[str] = list(dict.fromkeys(args.rules))

    print("=" * 110)
    print("conf 阈值 × 选框规则 二维扫描")
    print(f"测试集 : {root}")
    print(f"阈值   : {confs}")
    print(f"规则   : " + "   ".join(f"{r}（{RULE_LABELS[r]}）" for r in rules))
    print(f"口径   : match_iou={args.match_iou}   imgsz={args.imgsz}")
    print(f"实现   : {_EVAL_SCRIPT.name}（ultralytics {_LOCKED_VERSION}）")
    print("=" * 110)

    rgb_map = stem_map(root / "rgb" / "images" / args.split)
    rgbd_map = stem_map(root / "rgbd" / "images" / args.split)
    label_dir = root / "rgb" / "labels" / args.split
    if not label_dir.is_dir():
        label_dir = root / "rgbd" / "labels" / args.split
    if not label_dir.is_dir():
        print("[STOP] 找不到标签目录")
        return
    print(f"源图片 : rgb={len(rgb_map)} 张，rgbd={len(rgbd_map)} 张")

    if args.weights:
        candidates = [Path(w).expanduser() for w in args.weights]
    else:
        weights_dir = Path(args.weights_dir).expanduser()
        candidates = (
            sorted(p for p in weights_dir.iterdir()
                   if p.is_file() and p.suffix.lower() == ".pt")
            if weights_dir.is_dir() else []
        )
    if not candidates:
        print("[STOP] 没有找到任何 .pt 权重")
        return

    models: list[dict[str, Any]] = []
    failed: list[str] = []
    for weights in candidates:
        if not weights.is_file():
            failed.append(f"{weights.name}(不存在)")
            continue
        try:
            model = YOLO(str(weights))
        except Exception as exc:  # noqa: BLE001
            print(f"[错误] {weights.stem} 加载失败: {type(exc).__name__}: {exc}")
            failed.append(f"{weights.stem}({type(exc).__name__})")
            continue
        channels = int(model.model.model[0].conv.weight.shape[1])
        if channels not in (3, 4):
            failed.append(f"{weights.stem}(ch={channels})")
            continue
        source_map = rgb_map if channels == 3 else rgbd_map
        if not source_map:
            failed.append(f"{weights.stem}(缺{'rgb' if channels == 3 else 'rgbd'})")
            continue
        models.append({"name": weights.stem, "path": weights, "model": model,
                       "channels": channels, "source_map": source_map})
        print(f"已加载: {weights.stem:24s} {channels}ch")

    if failed:
        print()
        print("!" * 110)
        print("[STOP] 有权重未能参与，结果不完整：")
        for name in failed:
            print(f"         - {name}")
        print("        修复后重跑。")
        print("!" * 110)
        return
    if not models:
        print("[STOP] 没有可用的权重")
        return

    stems = sorted(set(rgb_map) | set(rgbd_map))
    summary_rows: list[dict[str, Any]] = []
    per_image_rows: list[dict[str, Any]] = []

    for item in models:
        name: str = item["name"]
        channels: int = item["channels"]
        source_map = item["source_map"]
        usable = [s for s in stems
                  if s in source_map and (label_dir / f"{s}.txt").is_file()]
        print(f"\n>>> {name}（{channels}ch，{len(usable)} 张）")

        # 图像与标签只读一次，所有 conf / 规则复用
        cache: list[tuple[str, np.ndarray, dict[str, Any]]] = []
        for stem in usable:
            image = read_image(source_map[stem], channels)
            height, width = image.shape[:2]
            gt = ev.load_pose_label(label_dir / f"{stem}.txt", width, height)
            cache.append((stem, image, gt))

        for conf in confs:
            started = time.perf_counter()
            box_counts: list[int] = []
            # 每套规则各攒一份
            bucket: dict[str, dict[str, list[Any]]] = {
                r: {"iou": [], "nme": []} for r in rules
            }
            matched_count: dict[str, int] = {r: 0 for r in rules}
            image_detail: list[dict[str, Any]] = []

            for stem, image, gt in cache:
                result = ev.run_pose_model(item["model"], image, conf, args.imgsz)
                n_boxes = 0 if result.boxes is None else len(result.boxes)
                box_counts.append(n_boxes)
                detail: dict[str, Any] = {"image": stem, "n_boxes": n_boxes}
                for rule in rules:
                    picked = SELECTORS[rule](result, gt["box"], args.match_iou)
                    nme_value: float | None = None
                    if picked["matched"]:
                        matched_count[rule] += 1
                        bucket[rule]["iou"].append(picked["iou"])
                        try:
                            nme_value = float(
                                ev.calculate_nme(gt["box"], gt["keypoints"],
                                                 picked["keypoints"])["nme"]
                            )
                            bucket[rule]["nme"].append(nme_value)
                        except ValueError:
                            nme_value = None
                    detail[f"{rule}_matched"] = bool(picked["matched"])
                    detail[f"{rule}_iou"] = round(float(picked["iou"]), 6)
                    detail[f"{rule}_idx"] = picked["index"]
                    detail[f"{rule}_nme"] = (None if nme_value is None
                                             else round(nme_value, 6))
                image_detail.append(detail)

            for rule in rules:
                nme_mean = mean_of(bucket[rule]["nme"])
                summary_rows.append({
                    "model": name,
                    "channels": channels,
                    "conf": conf,
                    "rule": rule,
                    "n_images": len(cache),
                    "boxes_total": int(sum(box_counts)),
                    "boxes_mean": mean_of(box_counts),
                    "boxes_median": median_of(box_counts),
                    "boxes_max": (max(box_counts) if box_counts else None),
                    "matched": matched_count[rule],
                    "matched_pct": (matched_count[rule] / len(cache) * 100.0
                                    if cache else None),
                    "selected_iou_mean": mean_of(bucket[rule]["iou"]),
                    "nme": nme_mean,
                    "n_nme": len(bucket[rule]["nme"]),
                    "elapsed_s": round(time.perf_counter() - started, 2),
                })

            for detail in image_detail:
                per_image_rows.append({
                    "model": name, "channels": channels, "conf": conf,
                    "image": detail["image"], "n_boxes": detail["n_boxes"],
                    **{k: v for k, v in detail.items()
                       if k not in ("image", "n_boxes")},
                })

            cells: list[str] = []
            for rule in rules:
                row = next(r for r in summary_rows
                           if r["model"] == name and r["conf"] == conf
                           and r["rule"] == rule)
                cells.append(f"{rule}:{fmt_nme(row['nme'])}({row['matched']})")
            print(f"    conf={fmt_conf(conf):<6s} 框数均值 "
                  f"{mean_of(box_counts):>7.2f} | " + "  ".join(cells))

    out_dir.mkdir(parents=True, exist_ok=True)

    summary_fields = ["model", "channels", "conf", "rule", "n_images",
                      "boxes_total", "boxes_mean", "boxes_median", "boxes_max",
                      "matched", "matched_pct", "selected_iou_mean",
                      "nme", "n_nme", "elapsed_s"]
    with (out_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)

    per_fields = ["model", "channels", "conf", "image", "n_boxes"]
    for rule in rules:
        per_fields += [f"{rule}_matched", f"{rule}_iou", f"{rule}_idx", f"{rule}_nme"]
    with (out_dir / "per_image.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(per_image_rows)

    (out_dir / "run_config.json").write_text(
        json.dumps({
            "dataset_root": str(root), "split": args.split, "confs": confs,
            "rules": rules, "match_iou": args.match_iou, "imgsz": args.imgsz,
            "weights": [{"name": m["name"], "channels": m["channels"],
                         "path": str(m["path"])} for m in models],
            "ultralytics": _LOCKED_VERSION, "eval_impl": _EVAL_SCRIPT.name,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def lookup(model: str, conf: float, rule: str) -> dict[str, Any]:
        return next(r for r in summary_rows
                    if r["model"] == model and r["conf"] == conf and r["rule"] == rule)

    # ---- 终端表 1：每图平均框数 ----
    print("\n" + "=" * 110)
    print("每图平均框数（与选框规则无关）")
    print("=" * 110)
    print(f"{'模型':24s}" + "".join(
        f"{(fmt_conf(c) + (' ←' if abs(c - HIGHLIGHT_CONF) < 1e-12 else '')):>16s}"
        for c in confs))
    for item in models:
        cells = [f"{lookup(item['name'], c, rules[0])['boxes_mean']:.2f}" for c in confs]
        print(f"{item['name']:24s}" + "".join(f"{c:>16s}" for c in cells))

    # ---- 终端表 2/3：每个规则一张 NME 表 ----
    for rule in rules:
        print("\n" + "=" * 110)
        print(f"NME —— 规则【{rule}】（{RULE_LABELS[rule]}）；"
              f"括号内为命中数，只在该阈值下命中的图上平均")
        print("=" * 110)
        print(f"{'模型':24s}" + "".join(
            f"{(fmt_conf(c) + (' ←' if abs(c - HIGHLIGHT_CONF) < 1e-12 else '')):>16s}"
            for c in confs))
        for item in models:
            cells: list[str] = []
            for conf in confs:
                row = lookup(item["name"], conf, rule)
                cells.append("-" if row["nme"] is None
                             else f"{row['nme']:.4f}({row['matched']})")
            print(f"{item['name']:24s}" + "".join(f"{c:>16s}" for c in cells))

    # ---- 终端表 4：conf=0.05 处两规则对照 ----
    if len(rules) == 2 and any(abs(c - HIGHLIGHT_CONF) < 1e-12 for c in confs):
        print("\n" + "=" * 110)
        print(f"conf={fmt_conf(HIGHLIGHT_CONF)} 处，两套规则对照")
        print("=" * 110)
        print(f"{'模型':24s}{'框数均值':>10s}{'iou规则NME':>13s}{'conf规则NME':>13s}"
              f"{'差':>10s}{'iou命中':>9s}{'conf命中':>10s}")
        for item in models:
            a = lookup(item["name"], HIGHLIGHT_CONF, "iou")
            b = lookup(item["name"], HIGHLIGHT_CONF, "conf")
            delta = ("-" if (a["nme"] is None or b["nme"] is None)
                     else f"{a['nme'] - b['nme']:+.6f}")
            print(f"{item['name']:24s}{a['boxes_mean']:>10.2f}"
                  f"{fmt_nme(a['nme']):>13s}{fmt_nme(b['nme']):>13s}"
                  f"{delta:>10s}{a['matched']:>9d}{b['matched']:>10d}")

    # ---- 判据 ----
    print("\n[判据]")
    for item in models:
        print(f"  {item['name']}")
        for rule in rules:
            base = lookup(item["name"], confs[0], rule)
            final = lookup(item["name"], confs[-1], rule)
            if base["nme"] is None or final["nme"] is None:
                print(f"      {rule:5s} 数据不足")
                continue
            pct = (base["nme"] - final["nme"]) / base["nme"] * 100.0
            print(f"      {rule:5s} conf {fmt_conf(confs[0])} -> {fmt_conf(confs[-1])}："
                  f"NME {base['nme']:.6f} -> {final['nme']:.6f} "
                  f"({'-' if pct >= 0 else '+'}{abs(pct):.1f}%)   "
                  f"框数 {base['boxes_mean']:.2f} -> {final['boxes_mean']:.2f}")
        if len(rules) == 2:
            a = lookup(item["name"], confs[-1], rules[0])
            b = lookup(item["name"], confs[-1], rules[1])
            if a["nme"] is not None and b["nme"] is not None:
                gap = (a["nme"] - b["nme"]) / a["nme"] * 100.0
                verdict = ("两规则一致" if abs(gap) < 1.0
                           else "iou 规则明显更差 -> 噪声框在干扰")
                print(f"      最高阈值处两规则差 {gap:+.1f}%   {verdict}")

    print(f"\n结果文件:\n  {out_dir / 'summary.csv'}\n  {out_dir / 'per_image.csv'}")


if __name__ == "__main__":
    main()
