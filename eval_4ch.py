"""四通道（RGBD）cross-subject 评估：新训 yolo11-4ch vs 第一阶段 v8-4ch 基线。

评估口径复用第一阶段脚本（import 复用，不重写算法），与 RGB 版评估保持一致：

    matched : bbox IoU >= --match-iou（默认 0.50）
    NME     : 每个 visibility>0 的关键点，像素误差 / GT框对角线，再取平均
    conf    : --conf（默认 0.001）

与 RGB 版评估的唯一区别：**RGBD 图像必须用 cv2.IMREAD_UNCHANGED 读取**，
否则 cv2 默认只给 3 通道，第 4 通道（Depth）会被静默丢弃 —— 那样测的就不是 RGBD。

用法：
    .venv\\Scripts\\python.exe eval_4ch.py

输出：
    runs\\eval_4ch\\per_image.csv
    runs\\eval_4ch\\summary.csv
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
OUT_DIR: Path = REPO / "runs" / "eval_4ch"

CROSS_ROOT: Path = Path(
    r"H:\YJJ\Yolo_RGBD\Resource\session1_200146\dataset\hand_cross_subject\rgbd"
)
IMG_DIR: Path = CROSS_ROOT / "images" / "test"
LBL_DIR: Path = CROSS_ROOT / "labels" / "test"

# 被评估的 4 通道模型（name -> 权重路径）
MODELS: dict[str, Path] = {
    "yolov8n_4ch_baseline": Path(
        r"H:\YJJ\Yolo_RGBD\yolov8_rgbd_detection\runs\pose\train_pose_rgbd_4ch5\weights\best.pt"
    ),
    "yolo11n_4ch_new": Path(
        r"H:\YJJ\Yolo_RGBD\yolov8_rgbd_detection\runs\pose\runs\pose\train_yolo11_rgbd_4ch\weights\best.pt"
    ),
}

KEYPOINT_COUNT: int = 7

# 第一阶段的评估脚本：算法唯一来源
EVAL_SCRIPT: Path = (
    Path(r"H:\YJJ\Yolo_RGBD\yolov8_rgbd_detection")
    / "YJJ_Pose_Scripts" / "04_eval" / "eval_rgb_vs_rgbd_nme.py"
)

# ============================================================
# 先锁定 venv 的 ultralytics，再加载第一阶段脚本（它会把原项目根插进 sys.path，
# 里面有 vendored 8.3.176，顺序反了会把 8.4.175 顶掉）。
# ============================================================
import ultralytics  # noqa: E402

_LOCKED_VERSION: str = ultralytics.__version__

if not EVAL_SCRIPT.is_file():
    raise FileNotFoundError(f"第一阶段评估脚本不存在: {EVAL_SCRIPT}")

sys.path.insert(0, str(EVAL_SCRIPT.parent))
import eval_rgb_vs_rgbd_nme as ev  # noqa: E402

if ultralytics.__version__ != _LOCKED_VERSION:
    raise RuntimeError(
        f"ultralytics 被换掉了: {_LOCKED_VERSION} -> {ultralytics.__version__}"
    )

_LEAKED_PATH = str(EVAL_SCRIPT.parents[2])
while _LEAKED_PATH in sys.path:
    sys.path.remove(_LEAKED_PATH)

from ultralytics import YOLO  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="四通道 RGBD 模型 cross-subject 评估")
    p.add_argument("--match-iou", type=float, default=0.50, help="bbox 匹配阈值，默认 0.50")
    p.add_argument("--conf", type=float, default=0.001, help="推理置信度阈值，默认 0.001")
    p.add_argument("--imgsz", type=int, default=640, help="推理尺寸，默认 640")
    return p.parse_args()


def list_test_images() -> list[Path]:
    if not IMG_DIR.is_dir():
        raise FileNotFoundError(f"4 通道测试图目录不存在: {IMG_DIR}")
    exts = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}
    return sorted(p for p in IMG_DIR.iterdir() if p.is_file() and p.suffix.lower() in exts)


def read_rgbd(image_path: Path) -> np.ndarray:
    """4 通道图必须 IMREAD_UNCHANGED，否则第 4 通道被丢弃。"""
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"图像读取失败: {image_path}")
    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError(f"图像不是 4 通道: {image_path}  shape={image.shape}")
    return image


def eval_one_image(
    image_path: Path,
    models: dict[str, YOLO],
    match_iou: float,
    conf: float,
    imgsz: int,
) -> dict[str, Any] | None:
    label_path = LBL_DIR / f"{image_path.stem}.txt"
    if not label_path.is_file():
        return None

    image = read_rgbd(image_path)
    height, width = image.shape[:2]

    gt = ev.load_pose_label(label_path, width, height)
    gt_box = gt["box"]
    gt_keypoints = gt["keypoints"]

    row: dict[str, Any] = {
        "image": image_path.name,
        "bbox_diagonal": float(ev.calculate_bbox_diagonal(gt_box)),
        "ch4_mean": float(np.asarray(image[:, :, 3], dtype=np.float64).mean()),
    }

    for name, model in models.items():
        started = time.perf_counter()
        result = ev.run_pose_model(model, image, conf, imgsz)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        pred = ev.find_best_prediction(result, gt_box, match_iou)
        row[f"{name}_matched"] = bool(pred["matched"])
        row[f"{name}_iou"] = round(float(pred["iou"]), 6)
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


def mean_of(values: list[Any]) -> float | None:
    clean = [v for v in values if v is not None]
    return float(np.mean(clean)) if clean else None


def main() -> None:
    args = parse_args()
    images = list_test_images()

    print("=" * 92)
    print("四通道（RGBD）cross-subject 评估")
    print(f"测试集    : {IMG_DIR}")
    print(f"图片数量  : {len(images)}")
    print(f"匹配阈值  : IoU >= {args.match_iou}   conf={args.conf}   imgsz={args.imgsz}")
    print(f"评估口径  : 复用 {EVAL_SCRIPT.name}（ultralytics {_LOCKED_VERSION}）")
    print("=" * 92)

    models: dict[str, YOLO] = {}
    for name, weights in MODELS.items():
        if not weights.is_file():
            print(f"[警告] 权重不存在，跳过: {name} -> {weights}")
            continue
        model = YOLO(str(weights))
        head = model.model.model[-1]
        first_ch = model.model.model[0].conv.weight.shape[1]
        print(f"已加载 {name:22s} 第一层通道={first_ch}  head={type(head).__name__}")
        models[name] = model

    if not models:
        print("[STOP] 没有可评估的模型")
        return

    # 第 4 通道自检：确认读进来的确实是 4 通道数据，不是被丢掉的 3 通道
    probe = read_rgbd(images[0])
    ch4 = np.asarray(probe[:, :, 3], dtype=np.float64)
    print(f"\n[自检] {images[0].name}  shape={probe.shape}  "
          f"第4通道 min={ch4.min():.0f} max={ch4.max():.0f} mean={ch4.mean():.2f}")
    if ch4.max() == 0:
        print("[警告] 第 4 通道全为 0，Depth 可能没读进来")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    skipped = 0
    for index, image_path in enumerate(images, start=1):
        try:
            row = eval_one_image(image_path, models, args.match_iou, args.conf, args.imgsz)
        except Exception as exc:  # noqa: BLE001
            print(f"  [跳过] {image_path.name}: {type(exc).__name__}: {exc}")
            skipped += 1
            continue
        if row is None:
            skipped += 1
            continue
        rows.append(row)
        if index % 20 == 0 or index == len(images):
            print(f"  进度 {index}/{len(images)}")

    if not rows:
        print("[STOP] 没有可评估的样本")
        return

    fieldnames = ["image", "bbox_diagonal", "ch4_mean"]
    for name in models:
        fieldnames += [f"{name}_matched", f"{name}_iou", f"{name}_ms", f"{name}_nme"]
        fieldnames += [f"{name}_k{k}_px" for k in range(KEYPOINT_COUNT)]

    per_image = OUT_DIR / "per_image.csv"
    with per_image.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    common = [r for r in rows if all(bool(r[f"{n}_matched"]) for n in models)]
    names = list(models.keys())

    summary: list[dict[str, Any]] = []
    for name in names:
        self_rows = [r for r in rows if bool(r[f"{name}_matched"])]
        entry: dict[str, Any] = {
            "model": name,
            "matched": len(self_rows),
            "total": len(rows),
            "match_rate": round(len(self_rows) / len(rows), 4),
            "nme_self": mean_of([r[f"{name}_nme"] for r in self_rows]),
            "nme_common": mean_of([r[f"{name}_nme"] for r in common]),
            "avg_ms": mean_of([r[f"{name}_ms"] for r in rows]),
        }
        for k in range(KEYPOINT_COUNT):
            entry[f"k{k}_px_common"] = mean_of([r[f"{name}_k{k}_px"] for r in common])
        summary.append(entry)

    summary_fields = ["model", "matched", "total", "match_rate", "nme_self",
                      "nme_common", "avg_ms"] + [f"k{k}_px_common" for k in range(KEYPOINT_COUNT)]
    summary_csv = OUT_DIR / "summary.csv"
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)

    print("\n" + "=" * 92)
    print("汇总（评估集 = hand_cross_subject/rgbd test）")
    print("=" * 92)
    print(f"{'模型':24s} {'命中':>10s} {'命中率':>8s} {'自身NME':>10s} {'共同集NME':>11s} {'平均耗时':>10s}")
    for e in summary:
        fmt = lambda v, d=6: "-" if v is None else f"{v:.{d}f}"
        print(f"{e['model']:24s} {e['matched']:>4d}/{e['total']:<5d} "
              f"{e['match_rate'] * 100:>7.1f}% {fmt(e['nme_self']):>10s} "
              f"{fmt(e['nme_common']):>11s} {fmt(e['avg_ms'], 2):>9s}ms")

    print(f"\n共同命中的图片: {len(common)} / {len(rows)}")
    if skipped:
        print(f"跳过（缺图或缺标签）: {skipped}")

    print("\n共同集上的 K0~K6 平均像素误差：")
    print(f"{'模型':24s} " + " ".join(f"{'K' + str(k):>8s}" for k in range(KEYPOINT_COUNT)))
    for e in summary:
        cells = " ".join(
            ("-" if e[f"k{k}_px_common"] is None else f"{e[f'k{k}_px_common']:.2f}").rjust(8)
            for k in range(KEYPOINT_COUNT)
        )
        print(f"{e['model']:24s} {cells}")

    print(f"\n结果文件:\n  {per_image}\n  {summary_csv}")


if __name__ == "__main__":
    main()
