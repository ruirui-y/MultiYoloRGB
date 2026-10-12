"""统一测试：把所有训练权重放在同一个公共测试集上评估。

为什么独立于 GUI
----------------
GUI 适合"划分数据集"（选目录、点按钮），但批量跑多个权重做对比很笨重。
本脚本只做一件事：**给定一组权重 + 一个测试集 → 出统一口径的对比表**。

口径（与 eval_rgb_vs_rgbd_nme.py 同源，不重写算法）
--------------------------------------------------
    matched : bbox IoU >= 0.50（取 IoU 最大的预测实例）
    NME     : 每个 visibility>0 的关键点 → |pred-gt| / GT框对角线，再取平均
    分母    : 测试集全部图片（公共集全量口径）
    conf=0.05   imgsz=640
    （0.001 为历史值：会让输出框数异常的模型吐出大量 conf<0.005 的噪音框，
      而"取 IoU 最大"会被这些框带偏 → 见 runs/conf_sweep 的扫描结果）

跨模态
------
3 通道权重读 <root>/rgb/images/<split>，4 通道权重读 <root>/rgbd/images/<split>；
两模态按 **stem** 对齐（同一批帧，扩展名不同：rgb 多为 .jpg，rgbd 为 .png）。

用法
----
    cd /d H:\\YJJ\\Yolo_RGBD\\MultiYoloRGB
    .venv\\Scripts\\python.exe test_all.py

    # 换权重目录 / 测试集
    .venv\\Scripts\\python.exe test_all.py --weights-dir train_weight ^
        --dataset-root H:\\...\\dataset\\hand_common

输出
----
    runs\\test_all\\per_image.csv      每张图 × 每个模型
    runs\\test_all\\summary.csv        对比表
    runs\\test_all\\run_config.json    本次参数（可复现）
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
_REPO: Path = _HERE.parent                                   # MultiYoloRGB
_SIBLING: Path = _REPO.parent / "yolov8_rgbd_detection"      # 同级仓库（第一阶段）
_EVAL_SCRIPT: Path = (
    _SIBLING / "YJJ_Pose_Scripts" / "04_eval" / "eval_rgb_vs_rgbd_nme.py"
)
_DEFAULT_DATASET: Path = Path(
    r"H:\YJJ\Yolo_RGBD\Resource\session1_200146\dataset\hand_common"
)

# ============================================================
# 先剥离"项目根"再 import ultralytics。
# 坑：第一阶段仓库根目录下有一份 vendored ultralytics 8.3.176，
# 一旦它进入 sys.path（例如在项目根 cwd 下运行，或 GUI 的
# ProcessRunner 把项目根前置到 PYTHONPATH），import ultralytics 就会
# 命中它，导致 YOLO26 权重无法加载。本脚本必须用 8.4.x。
# ============================================================
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

# 第一阶段脚本会 sys.path.insert(0, 项目根)，这里再清一次
while _SIBLING_NORM in [os.path.normcase(os.path.abspath(p)) for p in sys.path if p]:
    for index, entry in enumerate(sys.path):
        if entry and os.path.normcase(os.path.abspath(entry)) == _SIBLING_NORM:
            del sys.path[index]
            break

from ultralytics import YOLO  # noqa: E402


IMAGE_EXTS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
)
KEYPOINT_COUNT: int = 7


# ============================================================
# 参数
# ============================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="统一测试：所有训练权重在同一公共测试集上评估")
    p.add_argument("--dataset-root", default=str(_DEFAULT_DATASET),
                   help=f"测试集根目录（含 rgb/ 或 rgbd/），默认 {_DEFAULT_DATASET}")
    p.add_argument("--weights-dir", default=str(_REPO / "train_weight"),
                   help="权重目录（自动收集其中所有 .pt），默认 train_weight")
    p.add_argument("--weights", nargs="+", default=None,
                   help="显式指定权重（覆盖 --weights-dir）")
    p.add_argument("--split", default="test", choices=["test"],
                   help="评估哪个 split，目前仅 test")
    p.add_argument("--match-iou", type=float, default=0.50, help="命中判定 IoU，默认 0.50")
    p.add_argument("--conf", type=float, default=0.05,
                   help="推理置信度，默认 0.05（0.001 会引入大量噪音框，"
                        "使『取 IoU 最大』挑错框，见 runs/conf_sweep）")
    p.add_argument("--imgsz", type=int, default=640, help="推理尺寸，默认 640")
    p.add_argument("--output-dir", default=str(_REPO / "runs" / "test_all"),
                   help="结果输出目录，默认 runs/test_all")
    return p.parse_args()


# ============================================================
# 工具
# ============================================================
def stem_map(directory: Path) -> dict[str, Path]:
    """图片 stem -> 路径。"""
    if not directory.is_dir():
        return {}
    return {
        p.stem: p
        for p in sorted(directory.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    }


def read_image(path: Path, channels: int) -> np.ndarray:
    """3 通道用 IMREAD_COLOR；4 通道必须 IMREAD_UNCHANGED，否则 Depth 被静默丢弃。"""
    flag = cv2.IMREAD_UNCHANGED if channels == 4 else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)
    if image is None:
        raise FileNotFoundError(f"图像读取失败: {path}")
    if image.ndim != 3:
        raise ValueError(f"图像维度异常: {path} shape={image.shape}")
    return image


def load_model(weights: Path) -> tuple[YOLO, int, str]:
    """返回 (model, 第一层输入通道数, head 类名)。"""
    model = YOLO(str(weights))
    channels = int(model.model.model[0].conv.weight.shape[1])
    head = type(model.model.model[-1]).__name__
    return model, channels, head


def mean_of(values: list[Any]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return float(np.mean(clean)) if clean else None


# ============================================================
# 主流程
# ============================================================
def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root).expanduser()
    weights_dir = Path(args.weights_dir).expanduser()
    out_dir = Path(args.output_dir).expanduser()
    split = args.split

    print("=" * 100)
    print("统一测试（所有训练权重 × 同一公共测试集）")
    print(f"测试集   : {root}")
    print(f"权重目录 : {weights_dir}")
    print(f"输出目录 : {out_dir}")
    print(f"口径     : IoU >= {args.match_iou}   conf={args.conf}   imgsz={args.imgsz}   split={split}")
    print(f"实现     : {_EVAL_SCRIPT.name}（ultralytics {_LOCKED_VERSION}）")
    print("=" * 100)

    # ---- 1. 收集权重 ----
    if args.weights:
        weight_paths = [Path(w).expanduser() for w in args.weights]
    elif weights_dir.is_dir():
        weight_paths = sorted(
            p for p in weights_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".pt"
        )
    else:
        weight_paths = []
    if not weight_paths:
        print(f"[STOP] 没有找到权重：{weights_dir}")
        return

    # ---- 2. 收集测试图片 ----
    rgb_map = stem_map(root / "rgb" / "images" / split)
    rgbd_map = stem_map(root / "rgbd" / "images" / split)
    if not rgb_map and not rgbd_map:
        print(f"[STOP] 测试集里没有图片：{root}")
        return
    stems = sorted(set(rgb_map) & set(rgbd_map)) if (rgb_map and rgbd_map) \
        else sorted(rgb_map or rgbd_map)
    if not stems:
        print("[STOP] rgb/ 与 rgbd/ 没有共同的 stem")
        return
    label_dir = (root / "rgb" / "labels" / split)
    if not label_dir.is_dir():
        label_dir = root / "rgbd" / "labels" / split
    if not label_dir.is_dir():
        print(f"[STOP] 找不到标签目录：{root}\\rgb\\labels\\{split}")
        return

    print(f"测试图片 : rgb={len(rgb_map)}，rgbd={len(rgbd_map)}，可用 stem={len(stems)}")
    print(f"标签目录 : {label_dir}")

    # ---- 3. 加载模型（按通道选图源）----
    models: list[dict[str, Any]] = []
    for weights in weight_paths:
        try:
            model, channels, head = load_model(weights)
        except Exception as exc:  # noqa: BLE001
            print(f"[跳过] {weights.name} 加载失败: {type(exc).__name__}: {exc}")
            continue
        if channels == 3 and not rgb_map:
            print(f"[跳过] {weights.name} 是 3 通道，但测试集缺少 rgb/")
            continue
        if channels == 4 and not rgbd_map:
            print(f"[跳过] {weights.name} 是 4 通道，但测试集缺少 rgbd/")
            continue
        models.append({"name": weights.stem, "path": weights, "model": model,
                       "channels": channels, "head": head})
        print(f"已加载: {weights.stem:22s} {channels}ch   head={head}")

    if not models:
        print("[STOP] 没有可用的权重")
        return

    # ---- 4. 逐图推理 ----
    rows: list[dict[str, Any]] = []
    for index, stem in enumerate(stems, start=1):
        label_path = label_dir / f"{stem}.txt"
        if not label_path.is_file():
            continue
        row: dict[str, Any] = {"image": stem}
        for item in models:
            name = item["name"]
            image_path = (rgb_map if item["channels"] == 3 else rgbd_map).get(stem)
            if image_path is None:
                row[f"{name}_matched"] = False
                row[f"{name}_iou"] = 0.0
                row[f"{name}_ms"] = None
                row[f"{name}_nme"] = None
                for k in range(KEYPOINT_COUNT):
                    row[f"{name}_k{k}_px"] = None
                continue
            try:
                image = read_image(image_path, item["channels"])
                height, width = image.shape[:2]
                gt = ev.load_pose_label(label_path, width, height)
                started = time.perf_counter()
                result = ev.run_pose_model(item["model"], image, args.conf, args.imgsz)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                pred = ev.find_best_prediction(result, gt["box"], args.match_iou)
                row[f"{name}_matched"] = bool(pred["matched"])
                row[f"{name}_iou"] = round(float(pred["iou"]), 6)
                row[f"{name}_ms"] = round(elapsed_ms, 2)
                if not pred["matched"]:
                    row[f"{name}_nme"] = None
                    for k in range(KEYPOINT_COUNT):
                        row[f"{name}_k{k}_px"] = None
                    continue
                nme_result = ev.calculate_nme(gt["box"], gt["keypoints"], pred["keypoints"])
                row[f"{name}_nme"] = float(nme_result["nme"])
                pixel_errors = ev.extract_keypoint_pixel_errors(nme_result, KEYPOINT_COUNT)
                for k, value in enumerate(pixel_errors):
                    row[f"{name}_k{k}_px"] = None if value is None else round(float(value), 4)
            except Exception as exc:  # noqa: BLE001
                print(f"  [跳过] {stem} / {name}: {type(exc).__name__}: {exc}")
                row[f"{name}_matched"] = False
                row[f"{name}_iou"] = 0.0
                row[f"{name}_ms"] = None
                row[f"{name}_nme"] = None
                for k in range(KEYPOINT_COUNT):
                    row[f"{name}_k{k}_px"] = None
        rows.append(row)
        if index % 20 == 0 or index == len(stems):
            print(f"  进度 {index}/{len(stems)}", flush=True)

    if not rows:
        print("[STOP] 没有可评估的样本")
        return

    # ---- 5. 写 per_image ----
    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = ["image"]
    for item in models:
        name = item["name"]
        fieldnames += [f"{name}_matched", f"{name}_iou", f"{name}_ms", f"{name}_nme"]
        fieldnames += [f"{name}_k{k}_px" for k in range(KEYPOINT_COUNT)]

    per_image = out_dir / "per_image.csv"
    with per_image.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # ---- 6. 汇总（分母 = 测试集全部图片，公共集全量口径）----
    summary: list[dict[str, Any]] = []
    for item in models:
        name = item["name"]
        matched_rows = [r for r in rows if bool(r[f"{name}_matched"])]
        entry: dict[str, Any] = {
            "model": name,
            "channels": item["channels"],
            "head": item["head"],
            "weights": str(item["path"]),
            "matched": len(matched_rows),
            "total": len(rows),
            "match_rate": round(len(matched_rows) / len(rows), 4),
            "nme": mean_of([r[f"{name}_nme"] for r in rows]),
            "avg_ms": mean_of([r[f"{name}_ms"] for r in rows]),
        }
        for k in range(KEYPOINT_COUNT):
            entry[f"k{k}_px"] = mean_of([r[f"{name}_k{k}_px"] for r in rows])
        summary.append(entry)

    summary_fields = ["model", "channels", "head", "matched", "total", "match_rate",
                      "nme", "avg_ms"] + [f"k{k}_px" for k in range(KEYPOINT_COUNT)]
    summary_csv = out_dir / "summary.csv"
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)

    (out_dir / "run_config.json").write_text(
        json.dumps({
            "dataset_root": str(root), "split": split,
            "match_iou": args.match_iou, "conf": args.conf, "imgsz": args.imgsz,
            "total_images": len(rows),
            "weights": [{"name": m["name"], "path": str(m["path"]),
                         "channels": m["channels"], "head": m["head"]} for m in models],
            "ultralytics": _LOCKED_VERSION,
            "eval_impl": _EVAL_SCRIPT.name,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    # ---- 7. 终端对比表 ----
    print("\n" + "=" * 100)
    print(f"对比表（测试集 = {root.name}/{split}，共 {len(rows)} 张，NME 分母 = 全部图片）")
    print("=" * 100)
    print(f"{'模型':24s} {'通道':>4s} {'命中':>10s} {'NME':>10s} {'平均耗时':>10s}")
    for e in summary:
        nme_text = "-" if e["nme"] is None else f"{e['nme']:.6f}"
        ms_text = "-" if e["avg_ms"] is None else f"{e['avg_ms']:.2f}ms"
        print(f"{e['model']:24s} {str(e['channels']) + 'ch':>4s} "
              f"{e['matched']:>4d}/{e['total']:<5d} {nme_text:>10s} {ms_text:>10s}")

    print(f"\nK0~K6 平均像素误差（分母 = 全部图片）：")
    print(f"{'模型':24s} " + " ".join(f"{'K' + str(k):>8s}" for k in range(KEYPOINT_COUNT)))
    for e in summary:
        cells = " ".join(
            ("-" if e[f"k{k}_px"] is None else f"{e[f'k{k}_px']:.2f}").rjust(8)
            for k in range(KEYPOINT_COUNT)
        )
        print(f"{e['model']:24s} {cells}")

    # ---- 8. 一致性提醒 ----
    incomplete = [e for e in summary if e["matched"] < e["total"]]
    if incomplete:
        print("\n[警告] 以下模型在测试集上有未命中的图片 —— 它们的 NME 不是在"
              "「全部图片」上算的，与其他模型不可直接比较：")
        for e in incomplete:
            print(f"        - {e['model']}: 命中 {e['matched']}/{e['total']}"
                  f"（若要严格对比，请把该权重纳入公共测试集筛选后重跑）")

    print(f"\n结果文件:\n  {per_image}\n  {summary_csv}\n  {out_dir / 'run_config.json'}")


if __name__ == "__main__":
    main()
