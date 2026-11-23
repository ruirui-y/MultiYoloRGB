r"""生成"深度中心化"数据集：把每张图的电平固定住，只留下起伏。

规则（对第 4 通道，逐张图）：
    取该图有效像素的中位数 m
    新值 = 128 + (原值 − m)，再夹到 1~255；无效像素保持 0

这样：
    · 每张图的平均电平都被拉到 128 附近（不再携带"整体亮度"信息）
    · 手和背景的高低起伏完整保留（几何信息还在）

产出：
    dataset/hand_center/rgbd           训练 + 验证（75 张）
    dataset/hand_common_center/rgbd    跨人测试（100 张）

用法：
    .venv\Scripts\python.exe make_centered_depth_dataset.py
    .venv\Scripts\python.exe make_centered_depth_dataset.py --force
"""
from __future__ import annotations
import paths


import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

DATASET_ROOT = paths.DATASET_PARENT
TRAIN_SRC = DATASET_ROOT / "hand" / "rgbd"
TEST_SRC = DATASET_ROOT / "hand_common" / "rgbd"
TRAIN_DST = DATASET_ROOT / "hand_center" / "rgbd"
TEST_DST = DATASET_ROOT / "hand_common_center" / "rgbd"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
TARGET_LEVEL = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成深度中心化数据集")
    parser.add_argument("--train-src", default=str(TRAIN_SRC))
    parser.add_argument("--train-dst", default=str(TRAIN_DST))
    parser.add_argument("--test-src", default=str(TEST_SRC))
    parser.add_argument("--test-dst", default=str(TEST_DST))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def center_channel(image: np.ndarray) -> tuple[np.ndarray, float]:
    """把第 4 通道按该图中位数平移到 128；返回 (新图, 原中位数)。"""
    out = image.copy()
    channel = out[:, :, 3]
    valid = channel > 0
    if not valid.any():
        return out, 0.0
    median = float(np.median(channel[valid]))
    centered = TARGET_LEVEL + (channel[valid].astype(np.float32) - median)
    channel[valid] = np.clip(np.round(centered), 1, 255).astype(np.uint8)
    return out, median


def mirror_tree(src: Path, dst: Path, force: bool) -> dict[str, int]:
    """整棵树复制；images/ 下的图做中心化，labels/ 与其它文件原样复制。"""
    if dst.exists():
        if not force:
            raise FileExistsError(f"{dst} 已存在；要重建请加 --force")
        shutil.rmtree(dst)
    counts = {"images": 0, "labels": 0, "other": 0}
    medians: list[float] = []
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(src)
        target = dst / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative.parts[0] == "images" and path.suffix.lower() in IMAGE_SUFFIXES:
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None or image.ndim != 3 or image.shape[2] != 4:
                raise ValueError(f"不是 4 通道图: {path}")
            transformed, median = center_channel(image)
            cv2.imwrite(str(target), transformed)
            medians.append(median)
            counts["images"] += 1
        else:
            shutil.copy2(path, target)
            if path.suffix.lower() == ".txt":
                counts["labels"] += 1
            else:
                counts["other"] += 1
    counts["median_depth_mean"] = round(float(np.mean(medians)), 1) if medians else 0
    return counts


def write_yaml(dst: Path) -> Path:
    yaml_path = dst / "data_rgbd_center.yaml"
    yaml_path.write_text(
        "# 深度中心化版 RGBD 数据集（由 make_centered_depth_dataset.py 生成）\n"
        f"path: {dst.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        "names:\n"
        "  0: hand\n"
        "channels: 4\n"
        "kpt_shape: [7, 3]\n",
        encoding="utf-8",
    )
    return yaml_path


def main() -> None:
    args = parse_args()
    print("=" * 96)
    print("生成深度中心化数据集（逐张减去中位数、平移到 128，起伏保留）")
    print("=" * 96)
    report = {}
    for label, src, dst in (("train", Path(args.train_src), Path(args.train_dst)),
                            ("test", Path(args.test_src), Path(args.test_dst))):
        counts = mirror_tree(src, dst, args.force)
        yaml_path = write_yaml(dst)
        report[label] = {"src": str(src), "dst": str(dst), **counts, "yaml": str(yaml_path)}
        print(f"{label:<6} 图片 {counts['images']} 张，标签 {counts['labels']} 个，"
              f"原深度中位数平均 {counts['median_depth_mean']}")

    sample = next((Path(args.train_dst) / "images" / "train").iterdir())
    original = cv2.imread(str(Path(args.train_src) / "images" / "train" / sample.name), cv2.IMREAD_UNCHANGED)
    centered = cv2.imread(str(sample), cv2.IMREAD_UNCHANGED)
    valid = original[:, :, 3] > 0
    print("\n抽查其中一张：")
    print(f"  前三通道未变: {bool(np.array_equal(original[:, :, :3], centered[:, :, :3]))}")
    print(f"  原深度均值 {original[:, :, 3][valid].mean():.1f} → 中心化后 "
          f"{centered[:, :, 3][valid].mean():.1f}")
    print(f"  无效像素仍为 0: {bool(np.all(centered[:, :, 3][~valid] == 0))}")
    (Path(args.train_dst).parent / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n训练配置：{Path(args.train_dst) / 'data_rgbd_center.yaml'}")


if __name__ == "__main__":
    main()
