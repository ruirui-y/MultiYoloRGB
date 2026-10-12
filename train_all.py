"""正式训练：三个 YOLO 版本各训 120 轮，产出三套可对比的权重。

超参一字不改地取自第一阶段 RGB 基线：
    H:\\YJJ\\Yolo_RGBD\\yolov8_rgbd_detection\\runs\\pose\\train_pose_rgb_3ch5\\args.yaml
三个版本唯一的不同，只有"用哪个权重开头"。

用法：
    全部跑（约 16 小时，建议分次）：
        .venv\\Scripts\\python.exe train_all.py

    只跑某一个（推荐分次跑）：
        .venv\\Scripts\\python.exe train_all.py --only yolov8n-pose.pt
        .venv\\Scripts\\python.exe train_all.py --only yolo11n-pose.pt
        .venv\\Scripts\\python.exe train_all.py --only yolo26n-pose.pt

输出：
    runs\\train\\<版本名>\\weights\\best.pt
"""
from __future__ import annotations

import argparse
import csv
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO

REPO: Path = Path(__file__).resolve().parent
WEIGHTS_DIR: Path = REPO / "weights"
DATA_YAML: Path = Path(
    r"H:\YJJ\Yolo_RGBD\Resource\session1_200146\dataset\hand\rgb\data_rgb.yaml"
)
RUNS_DIR: Path = REPO / "runs" / "train"
ALL_MODELS: tuple[str, ...] = ("yolov8n-pose.pt", "yolo11n-pose.pt", "yolo26n-pose.pt")

TRAIN_ARGS: dict[str, Any] = dict(
    epochs=120,
    patience=20,
    batch=4,
    imgsz=640,
    device="0",
    workers=0,
    cache=False,
    amp=False,
    fraction=1.0,
    pretrained=True,
    optimizer="auto",
    seed=0,
    deterministic=True,
    cos_lr=False,
    close_mosaic=10,
    lr0=0.0005,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=10,
    warmup_momentum=0.8,
    mosaic=0.0,
    mixup=0.0,
    copy_paste=0.0,
    fliplr=0.0,
    flipud=0.0,
    degrees=10.0,
    translate=0.1,
    scale=0.2,
    shear=0.0,
    perspective=0.0,
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    save=True,
    val=True,
    plots=True,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="三版本 RGB Pose 正式训练")
    p.add_argument(
        "--only",
        action="append",
        default=None,
        help=f"只训指定权重，可重复；不传则训全部。可选值: {', '.join(ALL_MODELS)}",
    )
    return p.parse_args()


def run_one(name: str) -> dict[str, Any]:
    """训一个版本：加载 -> 120 轮 -> 回报产物与耗时。异常记录不中断后续版本。"""
    row: dict[str, Any] = {"model": name, "ok": False, "seconds": 0.0, "best": None, "error": None}
    weights = WEIGHTS_DIR / name
    if not weights.is_file():
        row["error"] = f"权重不存在: {weights}"
        return row

    started = time.time()
    try:
        model = YOLO(str(weights))
        model.train(
            task="pose",
            data=str(DATA_YAML),
            project=str(RUNS_DIR),
            name=name.replace(".pt", ""),
            **TRAIN_ARGS,
        )
        save_dir = Path(model.trainer.save_dir)
        best = save_dir / "weights" / "best.pt"
        row["best"] = str(best) if best.is_file() else None
        row["ok"] = best.is_file()
        row["save_dir"] = str(save_dir)
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    row["seconds"] = time.time() - started
    return row


def read_val_map50(save_dir: str) -> dict[str, str]:
    """读 results.csv 最后一行里的两个 mAP，用于快速确认训练确实在收敛。"""
    csv_path = Path(save_dir) / "results.csv"
    if not csv_path.is_file():
        return {}
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    last = {k.strip(): v for k, v in rows[-1].items()}
    return {
        "轮数": last.get("epoch", "?"),
        "Box mAP50": last.get("metrics/mAP50(B)", "?"),
        "Pose mAP50": last.get("metrics/mAP50(P)", "?"),
        "Pose mAP50-95": last.get("metrics/mAP50-95(P)", "?"),
    }


def main() -> None:
    args = parse_args()
    targets = tuple(args.only) if args.only else ALL_MODELS
    unknown = [t for t in targets if t not in ALL_MODELS]
    if unknown:
        print(f"[STOP] 未知权重名: {unknown}；可选: {', '.join(ALL_MODELS)}")
        return
    if not DATA_YAML.is_file():
        print(f"[STOP] 数据配置不存在: {DATA_YAML}")
        return

    print("=" * 78)
    print("正式训练：120 轮")
    print(f"数据配置: {DATA_YAML}")
    print(f"输出目录: {RUNS_DIR}")
    print(f"本次要跑: {', '.join(targets)}")
    print(f"设备    : {'GPU ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 78)

    results: list[dict[str, Any]] = []
    for name in targets:
        print(f"\n{'=' * 78}\n>>> 开始训练: {name}\n{'=' * 78}")
        row = run_one(name)
        results.append(row)
        print(f"\n--- {name} 结束，用时 {row['seconds'] / 3600:.2f} 小时 ---")
        if row["error"]:
            print(f"  报错: {row['error']}")
        if row.get("save_dir"):
            info = read_val_map50(row["save_dir"])
            if info:
                print(f"  最后一行记录: {info}")

    print(f"\n{'=' * 78}\n汇总\n{'=' * 78}")
    for row in results:
        status = "完成" if row["ok"] else "失败"
        print(f"{row['model']:22s} {status:4s}  {row['seconds'] / 3600:5.2f} 小时  {row['best'] or row['error']}")

    if all(row["ok"] for row in results):
        print("\n[READY] 三个版本都训练完成，可以进入评估环节")
    else:
        print("\n[注意] 有版本没跑完，看上面的报错；已完成的产物不受影响")


if __name__ == "__main__":
    main()
