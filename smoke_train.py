"""冒烟训练：三个 YOLO 版本各训 1 轮，验证训练链路能否跑通。

只回答一件事：三个版本能不能真正开始训练、7 点关键点能不能正确替换。
不评估精度、不比分数。

用法：
    .venv\\Scripts\\python.exe smoke_train.py

与正式训练的差异（必须有记录，否则结果不可比）：
    epochs      120 -> 1      （冒烟专用）
    patience    沿用值 -> 0    （只训 1 轮，早停无意义）
    plots       True -> False （仅少生成几张图，不影响训练数值）
    其余超参与第一阶段 train_rgb_pose.py 完全一致，一字未改。

注意：epochs=1 时 warmup_epochs=10 会占满全程，loss 不会正常下降，
      这是正常的，不要当成训练失败。
"""
from __future__ import annotations

import csv
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
RUNS_DIR: Path = REPO / "runs" / "smoke"
MODELS: tuple[str, ...] = ("yolov8n-pose.pt", "yolo11n-pose.pt", "yolo26n-pose.pt")
EXPECTED_KPT: tuple[int, ...] = (7, 3)

TRAIN_ARGS: dict[str, Any] = dict(
    epochs=1,
    imgsz=640,
    batch=4,
    patience=0,
    save=True,
    plots=False,
    verbose=True,
    workers=0,
    cache=False,
    amp=False,
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
    lr0=0.0005,
    lrf=0.01,
    warmup_epochs=10,
)


def probe_stem_input_channels(model: YOLO) -> int | None:
    """读第一层卷积吃几个通道。第一阶段靠 assert==3 检查，这里只探测不拦截。"""
    try:
        conv = model.model.model[0].conv
        return int(conv.weight.shape[1])
    except Exception:  # noqa: BLE001
        return None


def read_last_loss_row(save_dir: Path) -> dict[str, str]:
    """从 results.csv 读最后一行里的 train/* 列（动态取列名，兼容没有 DFL 的版本）。"""
    csv_path = save_dir / "results.csv"
    if not csv_path.is_file():
        return {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}
    last = rows[-1]
    return {k.strip(): v for k, v in last.items() if k.strip().startswith("train/")}


def reload_kpt_shape(weights_path: Path) -> tuple[int, ...] | None:
    """重新加载训练产物，读它的关键点配置，验证'存下来的能读回来'。"""
    if not weights_path.is_file():
        return None
    try:
        reloaded = YOLO(str(weights_path))
        shape = reloaded.model.yaml.get("kpt_shape")
        return tuple(shape) if shape is not None else None
    except Exception:  # noqa: BLE001
        return None


def smoke_one(name: str) -> dict[str, Any]:
    """跑一个版本：加载 -> 训 1 轮 -> 检查产物。任何异常都记录，不中断其他版本。"""
    row: dict[str, Any] = {
        "model": name,
        "started": False,
        "stem_ch": None,
        "losses": {},
        "saved_kpt": None,
        "error": None,
    }
    weights = WEIGHTS_DIR / name
    if not weights.is_file():
        row["error"] = f"权重不存在: {weights}"
        return row

    try:
        model = YOLO(str(weights))
        row["stem_ch"] = probe_stem_input_channels(model)
        row["started"] = True

        model.train(
            task="pose",
            data=str(DATA_YAML),
            project=str(RUNS_DIR),
            name=name.replace(".pt", ""),
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            **TRAIN_ARGS,
        )

        save_dir = Path(model.trainer.save_dir)
        row["losses"] = read_last_loss_row(save_dir)
        row["saved_kpt"] = reload_kpt_shape(save_dir / "weights" / "last.pt")
        row["save_dir"] = str(save_dir)
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    return row


def judge(row: dict[str, Any]) -> dict[str, bool]:
    """四条判定，缺一不可。"""
    return {
        "训练无报错": row["error"] is None and row["started"],
        "权重点数为 7": row["saved_kpt"] == EXPECTED_KPT,
        "loss 有数值": bool(row["losses"]) and all(
            v.strip() not in ("", "nan", "NaN") for v in row["losses"].values()
        ),
        "权重可重新读回": row["saved_kpt"] is not None,
    }


def main() -> None:
    print("=" * 78)
    print("冒烟训练：三个版本各训 1 轮")
    print(f"数据配置: {DATA_YAML}")
    print(f"输出目录: {RUNS_DIR}")
    print("=" * 78)

    if not DATA_YAML.is_file():
        print(f"[STOP] 数据配置不存在: {DATA_YAML}")
        return

    results: list[dict[str, Any]] = []
    for name in MODELS:
        print(f"\n{'=' * 78}\n>>> 开始: {name}\n{'=' * 78}")
        row = smoke_one(name)
        row["checks"] = judge(row)
        results.append(row)

        print(f"\n--- {name} 结果 ---")
        print(f"  第一层输入通道 : {row['stem_ch']}")
        print(f"  训练 loss      : {row['losses'] or '（未取到）'}")
        print(f"  产物关键点配置 : {row['saved_kpt']}")
        if row["error"]:
            print(f"  报错           : {row['error']}")
        for label, ok in row["checks"].items():
            print(f"  [{'通过' if ok else '不通过'}] {label}")

    print(f"\n{'=' * 78}\n汇总\n{'=' * 78}")
    print(f"{'模型':22s} {'无报错':>7s} {'7点':>6s} {'loss':>6s} {'可读回':>8s}")
    for row in results:
        c = row["checks"]
        print(
            f"{row['model']:22s} "
            f"{('是' if c['训练无报错'] else '否'):>7s} "
            f"{('是' if c['权重点数为 7'] else '否'):>6s} "
            f"{('是' if c['loss 有数值'] else '否'):>6s} "
            f"{('是' if c['权重可重新读回'] else '否'):>8s}"
        )

    all_ok = all(all(row["checks"].values()) for row in results)
    print()
    if all_ok:
        print("[READY] 三个版本都能训通，7 点关键点正确，可以开正式训练")
    else:
        print("[BLOCKED] 有版本没过，先看上面的不通过项，不要直接跑 120 轮")


if __name__ == "__main__":
    main()
