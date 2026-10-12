"""三版本 cross-subject 评估：三个模型过同一批 101 张图，出对比表。

评估口径与第一阶段完全一致 —— 直接 import 第一阶段的评估脚本并复用它的函数，
不重写 IoU / NME / 匹配算法：

    matched : 预测框与 GT 框 IoU >= --match-iou（默认 0.50）
    NME     : 每个 visibility>0 的关键点，像素误差 / GT框对角线，再取平均
    conf    : --conf（默认 0.001，与第一阶段一致）

用法：
    全部评估：
        .venv\\Scripts\\python.exe eval_versions.py

    换口径（改了就与第一阶段不可比，需记录）：
        .venv\\Scripts\\python.exe eval_versions.py --match-iou 0.50 --conf 0.001

输出：
    runs\\eval_versions\\per_image.csv    每张图每个模型的结果
    runs\\eval_versions\\summary.csv      汇总对比表
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO: Path = Path(__file__).resolve().parent
WEIGHTS_DIR: Path = REPO / "weights"
TRAIN_DIR: Path = REPO / "runs" / "train"
OUT_DIR: Path = REPO / "runs" / "eval_versions"

CROSS_ROOT: Path = Path(
    r"H:\YJJ\Yolo_RGBD\Resource\session1_200146\dataset\hand_cross_subject\rgb"
)
IMG_DIR: Path = CROSS_ROOT / "images" / "test"
LBL_DIR: Path = CROSS_ROOT / "labels" / "test"

MODELS: tuple[str, ...] = ("yolov8n-pose", "yolo11n-pose", "yolo26n-pose")
KEYPOINT_COUNT: int = 7

# 第一阶段的评估脚本：算法唯一来源
EVAL_SCRIPT: Path = (
    Path(r"H:\YJJ\Yolo_RGBD\yolov8_rgbd_detection")
    / "YJJ_Pose_Scripts" / "04_eval" / "eval_rgb_vs_rgbd_nme.py"
)

# ============================================================
# 关键：先锁定 venv 里的 ultralytics，再加载第一阶段脚本。
# 那个脚本顶部会把自己的项目根插进 sys.path（里面有 vendored 8.3.176），
# 顺序反了就会把 8.4.175 换掉，yolo26 会直接加载失败。
# ============================================================
import ultralytics  # noqa: E402

_LOCKED_VERSION: str = ultralytics.__version__

if not EVAL_SCRIPT.is_file():
    raise FileNotFoundError(f"第一阶段评估脚本不存在: {EVAL_SCRIPT}")

sys.path.insert(0, str(EVAL_SCRIPT.parent))
import eval_rgb_vs_rgbd_nme as ev  # noqa: E402

if ultralytics.__version__ != _LOCKED_VERSION:
    raise RuntimeError(
        f"ultralytics 被换掉了: {_LOCKED_VERSION} -> {ultralytics.__version__}，"
        "检查 import 顺序"
    )

_LEAKED_PATH = str(EVAL_SCRIPT.parents[2])
while _LEAKED_PATH in sys.path:
    sys.path.remove(_LEAKED_PATH)

from ultralytics import YOLO  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="三版本 cross-subject 评估")
    p.add_argument("--match-iou", type=float, default=0.50, help="bbox 匹配阈值，默认 0.50")
    p.add_argument("--conf", type=float, default=0.001, help="推理置信度阈值，默认 0.001")
    p.add_argument("--imgsz", type=int, default=640, help="推理尺寸，默认 640")
    return p.parse_args()


def list_test_images() -> list[Path]:
    if not IMG_DIR.is_dir():
        raise FileNotFoundError(f"测试图目录不存在: {IMG_DIR}")
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(p for p in IMG_DIR.iterdir() if p.is_file() and p.suffix.lower() in exts)


def eval_one_image(
    image_path: Path,
    models: dict[str, YOLO],
    match_iou: float,
    conf: float,
    imgsz: int,
) -> dict[str, Any] | None:
    """一张图跑完三个模型，返回一行结果；GT 缺失或无标签返回 None。"""
    label_path = LBL_DIR / f"{image_path.stem}.txt"
    if not label_path.is_file():
        return None

    image = cv2.imread(str(image_path))
    if image is None:
        return None
    height, width = image.shape[:2]

    gt = ev.load_pose_label(label_path, width, height)
    gt_box = gt["box"]
    gt_keypoints = gt["keypoints"]

    row: dict[str, Any] = {
        "image": image_path.name,
        "bbox_diagonal": float(ev.calculate_bbox_diagonal(gt_box)),
    }

    for name, model in models.items():
        started = time.perf_counter()
        result = ev.run_pose_model(model, image, conf, imgsz)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        pred = ev.find_best_prediction(result, gt_box, match_iou)
        row[f"{name}_matched"] = bool(pred["matched"])
        row[f"{name}_iou"] = round(float(pred["iou"]), 6)
        row[f"{name}_conf"] = (
            round(float(pred["confidence"]), 6) if pred["confidence"] is not None else None
        )
        row[f"{name}_ms"] = round(elapsed_ms, 2)

        if not pred["matched"]:
            row[f"{name}_nme"] = None
            for k in range(KEYPOINT_COUNT):
                row[f"{name}_k{k}_px"] = None
            continue

        nme_result = ev.calculate_nme(gt_box, gt_keypoints, pred["keypoints"])
        row[f"{name}_nme"] = float(nme_result["nme"])

        pixel_errors = ev.extract_keypoint_pixel_errors(nme_result, KEYPOINT_COUNT)
        for k, value in enumerate(pixel_errors):
            row[f"{name}_k{k}_px"] = None if value is None else round(float(value), 4)

    return row


def mean_of(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None]
    return float(np.mean(clean)) if clean else None


def summarize(rows: list[dict[str, Any]], total: int) -> list[dict[str, Any]]:
    """在「三模型都命中」的共同集上算 NME —— 这才是可比口径。"""
    common_rows = [
        r for r in rows
        if all(bool(r[f"{name}_matched"]) for name in MODELS)
    ]

    summary: list[dict[str, Any]] = []
    for name in MODELS:
        self_rows = [r for r in rows if bool(r[f"{name}_matched"])]
        entry: dict[str, Any] = {
            "model": name,
            "matched": len(self_rows),
            "total": total,
            "match_rate": round(len(self_rows) / total, 4) if total else 0.0,
            "nme_self": mean_of([r[f"{name}_nme"] for r in self_rows]),
            "nme_common": mean_of([r[f"{name}_nme"] for r in common_rows]),
            "avg_ms": mean_of([r[f"{name}_ms"] for r in rows]),
            "mean_px_common": mean_of([
                mean_of([r[f"{name}_k{k}_px"] for r in common_rows])
                for k in range(KEYPOINT_COUNT)
            ]),
        }
        for k in range(KEYPOINT_COUNT):
            entry[f"k{k}_px_common"] = mean_of([r[f"{name}_k{k}_px"] for r in common_rows])
        summary.append(entry)

    return summary


def fmt(value: float | None, digits: int = 6) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def main() -> None:
    args = parse_args()
    images = list_test_images()

    print("=" * 88)
    print("三版本 cross-subject 评估")
    print(f"测试集    : {IMG_DIR}")
    print(f"图片数量  : {len(images)}")
    print(f"匹配阈值  : IoU >= {args.match_iou}   conf={args.conf}   imgsz={args.imgsz}")
    print(f"评估口径  : 复用 {EVAL_SCRIPT.name}（ultralytics {_LOCKED_VERSION}）")
    print("=" * 88)

    models: dict[str, YOLO] = {}
    for name in MODELS:
        weights = TRAIN_DIR / name / "weights" / "best.pt"
        if not weights.is_file():
            print(f"[STOP] 训练产物不存在: {weights}")
            return
        models[name] = YOLO(str(weights))
        print(f"已加载: {name}  <-  {weights}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    skipped = 0
    for index, image_path in enumerate(images, start=1):
        row = eval_one_image(image_path, models, args.match_iou, args.conf, args.imgsz)
        if row is None:
            skipped += 1
            continue
        rows.append(row)
        if index % 20 == 0 or index == len(images):
            print(f"  进度 {index}/{len(images)}")

    if not rows:
        print("[STOP] 没有可评估的样本")
        return

    fieldnames = ["image", "bbox_diagonal"]
    for name in MODELS:
        fieldnames += [f"{name}_matched", f"{name}_iou", f"{name}_conf", f"{name}_ms", f"{name}_nme"]
        fieldnames += [f"{name}_k{k}_px" for k in range(KEYPOINT_COUNT)]

    per_image = OUT_DIR / "per_image.csv"
    with per_image.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows, total=len(rows))
    summary_fields = [
        "model", "matched", "total", "match_rate", "nme_self", "nme_common",
        "avg_ms", "mean_px_common",
    ] + [f"k{k}_px_common" for k in range(KEYPOINT_COUNT)]

    summary_csv = OUT_DIR / "summary.csv"
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)

    common_count = sum(
        1 for r in rows if all(bool(r[f"{name}_matched"]) for name in MODELS)
    )

    print("\n" + "=" * 88)
    print("汇总（评估集 = cross-subject test）")
    print("=" * 88)
    print(f"{'模型':18s} {'命中':>10s} {'命中率':>8s} {'自身NME':>10s} {'共同集NME':>11s} {'平均耗时':>10s}")
    for entry in summary:
        print(
            f"{entry['model']:18s} "
            f"{entry['matched']:>4d}/{entry['total']:<5d} "
            f"{entry['match_rate'] * 100:>7.1f}% "
            f"{fmt(entry['nme_self']):>10s} "
            f"{fmt(entry['nme_common']):>11s} "
            f"{fmt(entry['avg_ms'], 2):>9s}ms"
        )

    print(f"\n三模型共同命中的图片: {common_count} / {len(rows)}"
          f"（只有在这批图上比 NME 才公平）")
    if skipped:
        print(f"跳过（缺图或缺标签）: {skipped}")

    print("\n共同集上的 K0~K6 平均像素误差：")
    header = f"{'模型':18s} " + " ".join(f"{'K' + str(k):>8s}" for k in range(KEYPOINT_COUNT))
    print(header)
    for entry in summary:
        cells = " ".join(
            f"{fmt(entry[f'k{k}_px_common'], 2):>8s}" for k in range(KEYPOINT_COUNT)
        )
        print(f"{entry['model']:18s} {cells}")

    print(f"\n结果文件:\n  {per_image}\n  {summary_csv}")


if __name__ == "__main__":
    main()
