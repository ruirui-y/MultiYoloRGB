"""逐框关系分析：模型一共吐多少框，以及"框的质量"与"点的质量"是什么关系。

要回答的两个问题
----------------
1. 每个模型在测试集（默认公共集 100 张）上，每张图会检索出多少个框？
2. 每一个框对应的 7 个点质量如何？「框有多准(IoU)」与「点有多准(NME)」是否同步？

为什么做这个
------------
当前评估规则是「取 IoU 最大的那个框，用它的点算 NME」。
若「框最准」不等于「点最准」，这条规则就会选错框，从而低估模型的真实定位能力。

口径（与第一阶段评估脚本完全同源，不重写算法）
----------------------------------------------
    推理   : run_pose_model      conf=0.001 / imgsz=640 / rect=False / augment=False
    框 IoU : calculate_iou       [x1,y1,x2,y2] 格式，交集/并集
    NME    : calculate_nme       除以【GT 框对角线】归一化，只统计 visibility > 0 的点
    逐点   : extract_keypoint_pixel_errors

用法
----
    cd /d H:\\YJJ\\Yolo_RGBD\\MultiYoloRGB
    .venv\\Scripts\\python.exe box_kpt_relation.py

    只跑部分权重：
    .venv\\Scripts\\python.exe box_kpt_relation.py --weights train_weight\\yolov8n_rgbd_4ch.pt

输出
----
    runs\\box_kpt_relation\\summary.csv      每个模型一行（框数统计 + 框点关系指标）
    runs\\box_kpt_relation\\box_count.csv    每个模型的框数分布直方
    runs\\box_kpt_relation\\per_image.csv    逐图（框数 / 规则选中框 / oracle）
    runs\\box_kpt_relation\\per_box.csv      逐框明细（conf / IoU / NME / 图内排名）
    runs\\box_kpt_relation\\run_config.json
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

# 先剥离"项目根"再 import。
# GUI 启动的子进程会把项目根塞进 PYTHONPATH（core/process_runner.py），
# 而项目根里有 vendored ultralytics 8.3.176，会把 venv 的 8.4.175 顶掉，
# 导致 YOLO26 权重直接加载失败。必须在任何 ultralytics import 之前摘掉。
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

# 框数分布分桶（左闭右闭）
BOX_BUCKETS: tuple[tuple[int, int, str], ...] = (
    (0, 0, "0"),
    (1, 2, "1-2"),
    (3, 5, "3-5"),
    (6, 10, "6-10"),
    (11, 50, "11-50"),
    (51, 200, "51-200"),
    (201, 10 ** 9, "201+"),
)


# ============================================================
# 小工具
# ============================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="逐框关系分析（框数 + 框点一致性）")
    p.add_argument("--dataset-root", default=str(_DEFAULT_DATASET),
                   help="数据集根目录（需含 rgb/ 与 rgbd/）")
    p.add_argument("--weights-dir", default=str(_REPO / "train_weight"))
    p.add_argument("--weights", nargs="+", default=None,
                   help="显式指定权重；不传则扫描 weights-dir 下的全部 .pt")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--match-iou", type=float, default=0.50,
                   help="命中判定阈值（只影响 matched 标签，不影响逐框统计）")
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--output-dir", default=str(_REPO / "runs" / "box_kpt_relation"))
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
    """3 通道用 IMREAD_COLOR，4 通道必须用 IMREAD_UNCHANGED（否则第 4 通道被静默丢弃）。"""
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


def percentile_of(values: list[Any], q: float) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return float(np.percentile(clean, q)) if clean else None


def pct_true(flags: list[bool]) -> float | None:
    return (sum(1 for f in flags if f) / len(flags) * 100.0) if flags else None


def safe_corr(xs: list[float], ys: list[float]) -> float | None:
    """皮尔逊相关；点数不足或常量输入时返回 None。"""
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    a = np.asarray(xs, dtype=np.float64)
    b = np.asarray(ys, dtype=np.float64)
    if a.std() <= 0 or b.std() <= 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


# ============================================================
# 单图逐框采集
# ============================================================
def collect_boxes(
    result: Any,
    gt_box: np.ndarray,
    gt_keypoints: np.ndarray,
) -> list[dict[str, Any]]:
    """把一个模型对一张图的全部输出框，逐个整理成记录。

    每个框都单独算：与 GT 框的 IoU、该框自己 7 个点的误差与 NME。
    """
    records: list[dict[str, Any]] = []
    if result.boxes is None or len(result.boxes) == 0:
        return records
    if result.keypoints is None or len(result.keypoints) == 0:
        return records

    pred_boxes = result.boxes.xyxy.cpu().numpy().astype(np.float64)
    if result.boxes.conf is not None:
        pred_confs = result.boxes.conf.cpu().numpy().astype(np.float64)
    else:
        pred_confs = np.zeros(len(pred_boxes), dtype=np.float64)
    pred_kpts = result.keypoints.data.cpu().numpy().astype(np.float64)

    for index in range(len(pred_boxes)):
        iou = float(ev.calculate_iou(gt_box, pred_boxes[index]))
        nme: float | None = None
        mean_px: float | None = None
        kpt_px: list[float | None] = [None] * KEYPOINT_COUNT
        valid_points = 0
        try:
            nme_result = ev.calculate_nme(gt_box, gt_keypoints, pred_kpts[index])
            nme = float(nme_result["nme"])
            mean_px = float(nme_result["mean_pixel_error"])
            valid_points = len(nme_result["valid_indices"])
            errors = ev.extract_keypoint_pixel_errors(nme_result, KEYPOINT_COUNT)
            kpt_px = [None if v is None else float(v) for v in errors]
        except (ValueError, IndexError):
            # 该框的关键点在 GT 有效点上取不到值，或对角线非法
            pass

        records.append({
            "box_index": index,
            "conf": float(pred_confs[index]),
            "iou": iou,
            "nme": nme,
            "mean_px": mean_px,
            "valid_points": valid_points,
            "kpt_px": kpt_px,
        })

    return records


def build_image_record(
    boxes: list[dict[str, Any]],
    match_iou: float,
) -> dict[str, Any]:
    """在一张图内做排名：哪一个是"框最准"、哪一个是"点最准"。"""
    record: dict[str, Any] = {
        "n_boxes": len(boxes),
        "best_iou_index": None,
        "best_iou": None,
        "best_iou_matched": False,
        "best_iou_nme": None,
        "best_iou_nme_rank": None,
        "oracle_index": None,
        "oracle_nme": None,
        "within_corr": None,
        "n_valid_boxes": 0,
        "n_valid_ranks": 0,
    }
    if not boxes:
        return record

    # ---- 规则选中：IoU 最大的框 ----
    best = max(boxes, key=lambda b: b["iou"])
    record["best_iou_index"] = best["box_index"]
    record["best_iou"] = best["iou"]
    record["best_iou_matched"] = bool(best["iou"] >= match_iou)
    record["best_iou_nme"] = best["nme"]

    # ---- 图内排名：按 NME 升序（越小越好），只对 NME 有效的框排名 ----
    valid = [b for b in boxes if b["nme"] is not None]
    record["n_valid_boxes"] = len(valid)
    record["n_valid_ranks"] = len(valid)
    if valid:
        ordered = sorted(valid, key=lambda b: b["nme"])
        for position, item in enumerate(ordered, start=1):
            item["nme_rank"] = position
        record["oracle_index"] = ordered[0]["box_index"]
        record["oracle_nme"] = ordered[0]["nme"]
        if best["nme"] is not None:
            record["best_iou_nme_rank"] = best["nme_rank"]

        # ---- 图内相关：框 IoU vs 该框 NME ----
        record["within_corr"] = safe_corr(
            [b["iou"] for b in valid],
            [b["nme"] for b in valid],
        )

    for item in boxes:
        item.setdefault("nme_rank", None)
    return record


# ============================================================
def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root).expanduser()
    out_dir = Path(args.output_dir).expanduser()

    print("=" * 108)
    print("逐框关系分析：模型输出多少个框 + 框的质量与点的质量的关系")
    print(f"测试集 : {root}")
    print(f"split  : {args.split}")
    print(f"口径   : conf={args.conf}   imgsz={args.imgsz}   match_iou={args.match_iou}")
    print(f"实现   : {_EVAL_SCRIPT.name}（ultralytics {_LOCKED_VERSION}）")
    print("=" * 108)

    rgb_map = stem_map(root / "rgb" / "images" / args.split)
    rgbd_map = stem_map(root / "rgbd" / "images" / args.split)
    label_dir = root / "rgb" / "labels" / args.split
    if not label_dir.is_dir():
        label_dir = root / "rgbd" / "labels" / args.split
    if not label_dir.is_dir():
        print(f"[STOP] 找不到标签目录: {root / 'rgb' / 'labels' / args.split}")
        return
    print(f"源图片 : rgb={len(rgb_map)} 张，rgbd={len(rgbd_map)} 张，标签: {label_dir}")

    # ---- 加载权重（自动判 3ch / 4ch）----
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
            print(f"[跳过] 不存在: {weights}")
            failed.append(f"{weights.name}(不存在)")
            continue
        try:
            model = YOLO(str(weights))
        except Exception as exc:  # noqa: BLE001
            print(f"[错误] {weights.stem} 加载失败: {type(exc).__name__}: {exc}")
            if "Pose26" in str(exc):
                print("        该权重是 YOLO26，需要 ultralytics >= 8.4；"
                      "当前解释器加载的是项目内 vendored 8.3.176。")
                print("        请在 GUI 中把 Python 解释器指向 "
                      "MultiYoloRGB\\.venv\\Scripts\\python.exe")
            failed.append(f"{weights.stem}({type(exc).__name__})")
            continue
        channels = int(model.model.model[0].conv.weight.shape[1])
        if channels not in (3, 4):
            print(f"[跳过] {weights.stem} 第一层通道={channels}（非 3/4）")
            failed.append(f"{weights.stem}(ch={channels})")
            continue
        if channels == 3 and not rgb_map:
            print(f"[跳过] {weights.stem} 是 3 通道，但源缺少 rgb/{args.split}")
            failed.append(f"{weights.stem}(缺rgb)")
            continue
        if channels == 4 and not rgbd_map:
            print(f"[跳过] {weights.stem} 是 4 通道，但源缺少 rgbd/{args.split}")
            failed.append(f"{weights.stem}(缺rgbd)")
            continue
        head = type(model.model.model[-1]).__name__
        models.append({"name": weights.stem, "path": weights,
                       "model": model, "channels": channels, "head": head})
        print(f"已加载: {weights.stem:24s} {channels}ch   head={head}")

    if failed:
        print()
        print("!" * 108)
        print(f"[STOP] 有 {len(failed)} 个权重未能参与统计，结果不完整：")
        for name in failed:
            print(f"         - {name}")
        print("        修复后重跑；确需忽略请显式删掉 --weights 里的对应项。")
        print("!" * 108)
        return
    if not models:
        print("[STOP] 没有可用的权重")
        return

    # ---- 逐模型逐图推理 ----
    stems = sorted(set(rgb_map) | set(rgbd_map))
    per_box_rows: list[dict[str, Any]] = []
    per_image_rows: list[dict[str, Any]] = []
    model_stats: dict[str, dict[str, Any]] = {}

    for item in models:
        name: str = item["name"]
        channels: int = item["channels"]
        source_map = rgb_map if channels == 3 else rgbd_map
        usable = [s for s in stems if s in source_map and (label_dir / f"{s}.txt").is_file()]
        started = time.perf_counter()
        frame_boxes: list[int] = []
        frame_corrs: list[float] = []
        frame_neg: list[bool] = []
        sel_iou: list[float] = []
        sel_nme: list[float] = []
        sel_rank: list[float] = []
        sel_is_oracle: list[bool] = []
        oracle_nme: list[float] = []
        all_box_nme: list[float] = []
        all_box_iou: list[float] = []
        zero_frames = 0

        for position, stem in enumerate(usable, start=1):
            label_path = label_dir / f"{stem}.txt"
            image = read_image(source_map[stem], channels)
            height, width = image.shape[:2]
            gt = ev.load_pose_label(label_path, width, height)
            result = ev.run_pose_model(item["model"], image, args.conf, args.imgsz)

            boxes = collect_boxes(result, gt["box"], gt["keypoints"])
            record = build_image_record(boxes, args.match_iou)

            frame_boxes.append(record["n_boxes"])
            if record["n_boxes"] == 0:
                zero_frames += 1
            if record["within_corr"] is not None:
                frame_corrs.append(record["within_corr"])
                frame_neg.append(record["within_corr"] < 0)
            for b in boxes:
                all_box_iou.append(b["iou"])
                if b["nme"] is not None:
                    all_box_nme.append(b["nme"])
            if record["best_iou_nme"] is not None:
                sel_nme.append(record["best_iou_nme"])
            if record["best_iou"] is not None:
                sel_iou.append(record["best_iou"])
            if record["best_iou_nme_rank"] is not None:
                sel_rank.append(float(record["best_iou_nme_rank"]))
                sel_is_oracle.append(record["best_iou_nme_rank"] == 1)
            if record["oracle_nme"] is not None:
                oracle_nme.append(record["oracle_nme"])

            per_image_rows.append({
                "model": name,
                "channels": channels,
                "image": stem,
                "n_boxes": record["n_boxes"],
                "n_valid_boxes": record["n_valid_boxes"],
                "best_iou_index": record["best_iou_index"],
                "best_iou": (None if record["best_iou"] is None
                             else round(record["best_iou"], 6)),
                "best_iou_matched": record["best_iou_matched"],
                "best_iou_nme": (None if record["best_iou_nme"] is None
                                 else round(record["best_iou_nme"], 6)),
                "best_iou_nme_rank": record["best_iou_nme_rank"],
                "oracle_index": record["oracle_index"],
                "oracle_nme": (None if record["oracle_nme"] is None
                               else round(record["oracle_nme"], 6)),
                "within_corr": (None if record["within_corr"] is None
                                else round(record["within_corr"], 6)),
            })

            for b in boxes:
                per_box_rows.append({
                    "model": name,
                    "channels": channels,
                    "image": stem,
                    "box_index": b["box_index"],
                    "conf": round(b["conf"], 6),
                    "iou": round(b["iou"], 6),
                    "nme": (None if b["nme"] is None else round(b["nme"], 6)),
                    "mean_px": (None if b["mean_px"] is None else round(b["mean_px"], 3)),
                    "valid_points": b["valid_points"],
                    "nme_rank": b["nme_rank"],
                    "is_rule_selected": b["box_index"] == record["best_iou_index"],
                    **{f"k{k}_px": (None if b["kpt_px"][k] is None
                                    else round(b["kpt_px"][k], 3))
                       for k in range(KEYPOINT_COUNT)},
                })

            if position % 25 == 0 or position == len(usable):
                print(f"  {name:22s} 进度 {position}/{len(usable)}", flush=True)

        elapsed = time.perf_counter() - started
        stats: dict[str, Any] = {
            "model": name,
            "channels": channels,
            "head": item["head"],
            "n_images": len(usable),
            "boxes_total": int(sum(frame_boxes)),
            "boxes_mean": mean_of(frame_boxes),
            "boxes_median": median_of(frame_boxes),
            "boxes_min": (min(frame_boxes) if frame_boxes else None),
            "boxes_max": (max(frame_boxes) if frame_boxes else None),
            "boxes_p90": percentile_of(frame_boxes, 90),
            "frames_zero_box": zero_frames,
            "pct_le2": pct_true([n <= 2 for n in frame_boxes]),
            "pct_ge50": pct_true([n >= 50 for n in frame_boxes]),
            "corr_global": safe_corr(all_box_iou, all_box_nme),
            "corr_within_mean": mean_of(frame_corrs),
            "pct_frames_neg_corr": pct_true(frame_neg),
            "n_frames_with_corr": len(frame_corrs),
            "selected_iou_mean": mean_of(sel_iou),
            "selected_nme_mean": mean_of(sel_nme),
            "oracle_nme_mean": mean_of(oracle_nme),
            "selected_rank_mean": mean_of(sel_rank),
            "selected_rank_median": median_of(sel_rank),
            "pct_selected_is_best_nme": pct_true(sel_is_oracle),
            "all_box_nme_mean": mean_of(all_box_nme),
            "boxes_counted": len(all_box_iou),
            "elapsed_s": round(elapsed, 2),
        }
        if stats["selected_nme_mean"] and stats["oracle_nme_mean"]:
            stats["oracle_gain_pct"] = (
                (stats["selected_nme_mean"] - stats["oracle_nme_mean"])
                / stats["selected_nme_mean"] * 100.0
            )
        else:
            stats["oracle_gain_pct"] = None
        model_stats[name] = stats

        # ---- 框数分布分桶 ----
        stats["buckets"] = [
            {"label": label,
             "count": sum(1 for n in frame_boxes if low <= n <= high)}
            for low, high, label in BOX_BUCKETS
        ]

        print(f"  -> {name}: 共 {stats['boxes_total']} 框 / {len(usable)} 张，"
              f"均值 {stats['boxes_mean']:.2f}，用时 {elapsed:.1f}s")

    # ---- 落盘 ----
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_fields = [
        "model", "channels", "head", "n_images",
        "boxes_total", "boxes_mean", "boxes_median", "boxes_min", "boxes_max",
        "boxes_p90", "frames_zero_box", "pct_le2", "pct_ge50",
        "corr_global", "corr_within_mean", "pct_frames_neg_corr",
        "n_frames_with_corr", "boxes_counted",
        "selected_iou_mean", "selected_nme_mean", "oracle_nme_mean",
        "oracle_gain_pct", "selected_rank_mean", "selected_rank_median",
        "pct_selected_is_best_nme", "all_box_nme_mean", "elapsed_s",
    ]
    with (out_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(model_stats.values())

    bucket_rows: list[dict[str, Any]] = []
    for name, stats in model_stats.items():
        for bucket in stats["buckets"]:
            bucket_rows.append({
                "model": name,
                "range": bucket["label"],
                "frames": bucket["count"],
                "pct": (bucket["count"] / stats["n_images"] * 100.0
                        if stats["n_images"] else None),
            })
    with (out_dir / "box_count.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "range", "frames", "pct"])
        writer.writeheader()
        writer.writerows(bucket_rows)

    image_fields = [
        "model", "channels", "image", "n_boxes", "n_valid_boxes",
        "best_iou_index", "best_iou", "best_iou_matched",
        "best_iou_nme", "best_iou_nme_rank",
        "oracle_index", "oracle_nme", "within_corr",
    ]
    with (out_dir / "per_image.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=image_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(per_image_rows)

    box_fields = [
        "model", "channels", "image", "box_index", "conf", "iou", "nme", "mean_px",
        "valid_points", "nme_rank", "is_rule_selected",
    ] + [f"k{k}_px" for k in range(KEYPOINT_COUNT)]
    with (out_dir / "per_box.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=box_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(per_box_rows)

    (out_dir / "run_config.json").write_text(
        json.dumps({
            "dataset_root": str(root),
            "split": args.split,
            "stems": len(stems),
            "match_iou": args.match_iou,
            "conf": args.conf,
            "imgsz": args.imgsz,
            "weights": [{"name": m["name"], "channels": m["channels"],
                         "path": str(m["path"])} for m in models],
            "ultralytics": _LOCKED_VERSION,
            "eval_impl": _EVAL_SCRIPT.name,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ---- 终端表 1：框数 ----
    print("\n" + "=" * 108)
    print(f"框数统计（{len(stems)} 张）")
    print("=" * 108)
    print(f"{'模型':24s}{'通道':>5s}{'总框数':>9s}{'均值':>8s}{'中位':>7s}"
          f"{'最大':>7s}{'P90':>7s}{'0框图':>7s}{'≤2框占比':>10s}{'≥50框占比':>10s}")
    for stats in model_stats.values():
        print(f"{stats['model']:24s}{stats['channels']:>5d}{stats['boxes_total']:>9d}"
              f"{stats['boxes_mean']:>8.2f}{stats['boxes_median']:>7.0f}"
              f"{stats['boxes_max']:>7d}{stats['boxes_p90']:>7.0f}"
              f"{stats['frames_zero_box']:>7d}"
              f"{stats['pct_le2']:>9.1f}%{stats['pct_ge50']:>9.1f}%")

    # ---- 终端表 2：框点关系 ----
    print("\n" + "=" * 108)
    print("框的质量 vs 点的质量（规则 = 取 IoU 最大的框）")
    print("=" * 108)
    print(f"{'模型':24s}{'全局相关':>9s}{'图内相关':>9s}{'负相关图':>9s}"
          f"{'选中框NME':>10s}{'oracle':>9s}{'上限收益':>9s}"
          f"{'选中框排名中位':>14s}{'选中=最优':>10s}")
    for stats in model_stats.values():
        def fmt(key: str, digits: int = 4) -> str:
            value = stats.get(key)
            return "-" if value is None else f"{value:.{digits}f}"

        print(f"{stats['model']:24s}{fmt('corr_global'):>9s}"
              f"{fmt('corr_within_mean'):>9s}"
              f"{(fmt('pct_frames_neg_corr', 1) + '%'):>9s}"
              f"{fmt('selected_nme_mean', 6):>10s}"
              f"{fmt('oracle_nme_mean', 6):>9s}"
              f"{(fmt('oracle_gain_pct', 1) + '%'):>9s}"
              f"{fmt('selected_rank_median', 0):>14s}"
              f"{(fmt('pct_selected_is_best_nme', 1) + '%'):>10s}")

    # ---- 终端判据 ----
    print("\n[判据]")
    for stats in model_stats.values():
        global_corr = stats.get("corr_global")
        rank_median = stats.get("selected_rank_median")
        gain = stats.get("oracle_gain_pct")
        print(f"  {stats['model']}（{stats['channels']}ch，"
              f"平均 {stats['boxes_mean']:.2f} 框/图）")
        if global_corr is None:
            print("      点数不足，无法判断相关性")
            continue
        if global_corr < -0.5:
            verdict = "框好则点也好 → 取 IoU 最大的规则基本不会选错"
        elif global_corr < 0:
            verdict = "弱同步 → 规则大多数情况下可用"
        else:
            verdict = "不同步（框越好点未必越好）→ 规则可能选错框"
        print(f"      全局相关 {global_corr:+.3f}   {verdict}")
        if rank_median is not None:
            print(f"      规则选中的框，其点质量在图内排名中位数 = {rank_median:.0f}"
                  f"（1 = 该图点最准的框）")
        if gain is not None:
            print(f"      若允许直接挑点最准的框，NME 还能再降 {gain:.1f}%"
                  f"（这只是量损失的上限，不是可用成绩）")

    print(f"\n结果文件:")
    print(f"  {out_dir / 'summary.csv'}")
    print(f"  {out_dir / 'box_count.csv'}")
    print(f"  {out_dir / 'per_image.csv'}")
    print(f"  {out_dir / 'per_box.csv'}")
    print(f"  {out_dir / 'run_config.json'}")


if __name__ == "__main__":
    main()
