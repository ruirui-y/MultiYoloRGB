r"""生成"深度反转"版本的 RGBD 数据集（不修改原始数据）。

规则：第 4 通道（深度）每个有效像素取 255 − 原值；无效像素（0）保持 0。
其余三个通道、标签文件、目录结构原样复制。

产出两套：
    Resource/session1_200146/dataset/hand_inv/rgbd        训练/验证用（75 张）
    Resource/session1_200146/dataset/hand_common_inv/rgbd  测试用（100 张）

用法：
    .venv\Scripts\python.exe make_inverted_depth_dataset.py
    .venv\Scripts\python.exe make_inverted_depth_dataset.py --force
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
TRAIN_DST = DATASET_ROOT / "hand_inv" / "rgbd"
TEST_DST = DATASET_ROOT / "hand_common_inv" / "rgbd"

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成深度反转版数据集")
    parser.add_argument("--train-src", default=str(TRAIN_SRC))
    parser.add_argument("--train-dst", default=str(TRAIN_DST))
    parser.add_argument("--test-src", default=str(TEST_SRC))
    parser.add_argument("--test-dst", default=str(TEST_DST))
    parser.add_argument("--force", action="store_true", help="目标已存在时先删掉")
    return parser.parse_args()


def invert_channel(image: np.ndarray) -> np.ndarray:
    """第 4 通道取 255−原值；0（无效）保持 0。"""
    out = image.copy()
    channel = out[:, :, 3]
    valid = channel > 0
    channel[valid] = 255 - channel[valid]
    return out


def mirror_tree(src: Path, dst: Path, force: bool) -> dict[str, int]:
    """把 src 整棵树复制成 dst；images/ 下的图反转第 4 通道，labels/ 与其它文件原样复制。

    目录结构是 images/<split>/*.png 与 labels/<split>/*.txt，所以按相对路径递归复制，
    不能把 images/ 当成 split 目录直接遍历。
    """
    if dst.exists():
        if not force:
            raise FileExistsError(f"{dst} 已存在；确认要重建请加 --force")
        shutil.rmtree(dst)
    counts = {"images": 0, "labels": 0, "other": 0}
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(src)
        target = dst / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        in_images = relative.parts[0] == "images"
        if in_images and path.suffix.lower() in IMAGE_SUFFIXES:
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None or image.ndim != 3 or image.shape[2] != 4:
                raise ValueError(f"不是 4 通道图: {path} shape={None if image is None else image.shape}")
            cv2.imwrite(str(target), invert_channel(image))
            counts["images"] += 1
        else:
            shutil.copy2(path, target)
            if path.suffix.lower() == ".txt":
                counts["labels"] += 1
            else:
                counts["other"] += 1
    return counts


def write_yaml(dst: Path) -> Path:
    """写训练用的 data yaml（指向反转后的目录，channels=4）。"""
    yaml_path = dst / "data_rgbd_inv.yaml"
    yaml_path.write_text(
        "# 深度反转版 RGBD 数据集配置（由 make_inverted_depth_dataset.py 生成）\n"
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
    print("生成深度反转版数据集（第 4 通道 255−原值，无效保持 0，其余原样）")
    print("=" * 96)
    report = {}
    for label, src, dst in (("train", Path(args.train_src), Path(args.train_dst)),
                            ("test", Path(args.test_src), Path(args.test_dst))):
        if not src.is_dir():
            raise FileNotFoundError(src)
        counts = mirror_tree(src, dst, args.force)
        yaml_path = write_yaml(dst)
        report[label] = {"src": str(src), "dst": str(dst), **counts, "yaml": str(yaml_path)}
        print(f"{label:<6} {src} -> {dst}")
        print(f"       图片 {counts['images']} 张，标签 {counts['labels']} 个，配置 {yaml_path.name}")

    # 抽查：确认反转关系成立
    sample = next((Path(args.train_dst) / "images" / "train").iterdir())
    original = cv2.imread(str(Path(args.train_src) / "images" / "train" / sample.name), cv2.IMREAD_UNCHANGED)
    inverted = cv2.imread(str(sample), cv2.IMREAD_UNCHANGED)
    valid = original[:, :, 3] > 0
    same_rgb = bool(np.array_equal(original[:, :, :3], inverted[:, :, :3]))
    inverted_ok = bool(np.array_equal(inverted[:, :, 3][valid],
                                      255 - original[:, :, 3][valid]))
    zero_kept = bool(np.all(inverted[:, :, 3][~valid] == 0))
    print("\n抽查其中一张：")
    print(f"  前三通道未变: {same_rgb}")
    print(f"  第 4 通道 = 255−原值: {inverted_ok}")
    print(f"  无效像素仍为 0: {zero_kept}")
    (Path(args.train_dst).parent / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n训练配置（供训练脚本使用）：")
    print(f"  {Path(args.train_dst) / 'data_rgbd_inv.yaml'}")


if __name__ == "__main__":
    main()
