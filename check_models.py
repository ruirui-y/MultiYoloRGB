"""三代 YOLO-pose 权重推理结构一致性验证。

只回答一个问题：v8 / 11 / 26 三个版本的推理输出，能否被同一套后处理逻辑
（bbox 匹配 + keypoints 取点）消费。不训练、不评估精度。

注意：权重自带 COCO 17 点结构，此处推理输出即为 17 点；7 点（K0~K6）
是训练时由 data yaml 的 kpt_shape 重建 pose head 后才生效。

用法：
    .venv\\Scripts\\python.exe check_models.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics import YOLO

REPO: Path = Path(__file__).resolve().parent
WEIGHTS_DIR: Path = REPO / "weights"
CROSS_TEST_DIR: Path = Path(
    r"H:\YJJ\Yolo_RGBD\Resource\session1_200146\dataset\hand_cross_subject\rgb\images\test"
)
MODELS: tuple[str, ...] = (
    "yolov8n-pose.pt",
    "yolo11n-pose.pt",
    "yolo26n-pose.pt",
)
IMAGE_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp"})


def pick_test_image() -> Path:
    """取 cross-subject test 中的第一张图，保证三代输入完全一致。"""
    if not CROSS_TEST_DIR.is_dir():
        raise FileNotFoundError(f"测试图目录不存在: {CROSS_TEST_DIR}")
    images = sorted(
        p for p in CROSS_TEST_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"测试图目录为空: {CROSS_TEST_DIR}")
    return images[0]


def inspect(weights: Path, image: Path) -> dict[str, Any]:
    """加载单个权重并对同一张图推理，返回结构描述。"""
    model = YOLO(str(weights))
    results = model.predict(str(image), imgsz=640, verbose=False)
    result = results[0]

    boxes = result.boxes
    keypoints = result.keypoints
    end2end: bool = bool(getattr(model.model, "end2end", False))

    return {
        "model": weights.name,
        "task": model.task,
        "end2end": end2end,
        "n_boxes": 0 if boxes is None else int(boxes.xyxy.shape[0]),
        "box_shape": None if boxes is None else tuple(boxes.xyxy.shape),
        "has_conf": False if boxes is None else boxes.conf is not None,
        "kpt_shape": None if keypoints is None else tuple(keypoints.xy.shape),
        "kpt_conf_shape": None if keypoints is None else tuple(keypoints.conf.shape),
        "n_points": None if keypoints is None else int(keypoints.xy.shape[1]),
    }


def main() -> None:
    image = pick_test_image()
    print(f"测试图: {image.name}")
    print(f"输入尺寸: 640 | 权重目录: {WEIGHTS_DIR}")
    print("-" * 96)

    rows: list[dict[str, Any]] = []
    for name in MODELS:
        weights = WEIGHTS_DIR / name
        if not weights.is_file():
            print(f"{name:22s} 缺失 -> {weights}")
            continue
        try:
            rows.append(inspect(weights, image))
        except Exception as exc:  # noqa: BLE001
            print(f"{name:22s} FAIL {type(exc).__name__}: {exc}")

    header = (
        f"{'模型':22s} {'end2end':>8s} {'检出':>5s} "
        f"{'boxes.xyxy':>16s} {'kpts.xy':>18s} {'kpts.conf':>16s}"
    )
    print(header)
    for row in rows:
        print(
            f"{row['model']:22s} {str(row['end2end']):>8s} {row['n_boxes']:>5d} "
            f"{str(row['box_shape']):>16s} {str(row['kpt_shape']):>18s} "
            f"{str(row['kpt_conf_shape']):>16s}"
        )

    print("-" * 96)
    if len(rows) < len(MODELS):
        print(f"[FAIL] 只成功推理 {len(rows)}/{len(MODELS)} 个模型，先解决加载/推理失败")
        return

    point_counts = {row["n_points"] for row in rows}
    has_kpts = all(row["kpt_shape"] is not None for row in rows)
    has_boxes = all(row["box_shape"] is not None for row in rows)
    rank_ok = all(len(row["kpt_shape"]) == 3 for row in rows)

    print(f"关键点数量一致: {point_counts}  ->  {'通过' if len(point_counts) == 1 else '不通过'}")
    print(f"均能取到 boxes     : {'通过' if has_boxes else '不通过'}")
    print(f"均能取到 keypoints : {'通过' if has_kpts else '不通过'}")
    print(f"kpts 均为 3 维张量 : {'通过' if rank_ok else '不通过'}")
    verdict = len(point_counts) == 1 and has_kpts and has_boxes and rank_ok
    print(f"\n[{'READY' if verdict else 'BLOCKED'}] "
          f"{'三代输出结构一致，可用同一套后处理' if verdict else '结构不一致，后处理无法统一，需先定位'}")


if __name__ == "__main__":
    main()
