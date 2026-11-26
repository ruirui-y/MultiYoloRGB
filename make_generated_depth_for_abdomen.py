r"""给腹部数据集生成"26 生成的深度"，做成 4 通道数据集（与真实深度同一套目录结构）。

输入：``paths.ABDOMEN_DATASET/rgb/{images,labels}/{train,val,test}``
处理：每张 RGB → ``yolo26n-depth.pt`` → 相对深度 → 1%/99% 分位裁剪后归一化到 1~255
输出：``paths.ABDOMEN_DATASET_GEN/{rgb,rgbd}``（rgb 原样复制，rgbd 的第 4 层是生成深度）

方向与传感器编码一致：1 = 近，255 = 远。
生成深度没有"毫米"这个单位，所以只能按分位数归一化，encoder 与真实深度不同构。

用法：
    .venv\Scripts\python.exe make_generated_depth_for_abdomen.py
    .venv\Scripts\python.exe make_generated_depth_for_abdomen.py --force
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

import paths

SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="腹部生成深度数据集")
    parser.add_argument("--src", default=str(paths.ABDOMEN_DATASET), help="源数据集（含 rgb/）")
    parser.add_argument("--dst", default=str(paths.ABDOMEN_DATASET_GEN), help="输出目录")
    parser.add_argument("--depth-weights", default=str(paths.REPO / "yolo26n-depth.pt"))
    parser.add_argument("--low-percentile", type=float, default=1.0)
    parser.add_argument("--high-percentile", type=float, default=99.0)
    parser.add_argument("--class-name", default="abdomen", help="类别名（写进 yaml 的 names）")
    parser.add_argument("--kpt", type=int, default=3, help="关键点数量（写进 yaml 的 kpt_shape）")
    parser.add_argument("--force", action="store_true", help="目标已存在时先删掉")
    return parser.parse_args()


def to_channel(depth: np.ndarray, low: float, high: float) -> np.ndarray:
    """相对深度 → 1~255；分位数裁剪避免极值把中间段压扁。"""
    low_value = float(np.percentile(depth, low))
    high_value = float(np.percentile(depth, high))
    if high_value - low_value < 1e-6:
        return np.full(depth.shape, 128, dtype=np.uint8)
    normalized = (depth - low_value) / (high_value - low_value)
    return np.clip(np.round(1 + normalized * 254), 1, 255).astype(np.uint8)


def write_yaml(root: Path, channels: int, class_name: str = "abdomen", kpt: int = 3) -> None:
    (root / f"data_rgb{'' if channels == 3 else 'd'}_gen.yaml").write_text(
        "# 生成深度数据集（yolo26n-depth 生成，1%/99% 分位归一化）\n"
        f"path: {root.as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        f"nc: 1\nnames:\n  0: {class_name}\n"
        f"channels: {channels}\n"
        f"rgbd: {'true' if channels == 4 else 'false'}\n"
        f"kpt_shape: [{kpt}, 3]\n",
        encoding="utf-8")


def main() -> None:
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    if not (src / "rgb" / "images").is_dir():
        raise FileNotFoundError(f"源数据集缺 rgb/images: {src}")
    if dst.exists():
        if not args.force:
            raise FileExistsError(f"{dst} 已存在；要重建请加 --force")
        shutil.rmtree(dst)

    model = YOLO(args.depth_weights)
    total = 0
    for split in SPLITS:
        source_images = src / "rgb" / "images" / split
        if not source_images.is_dir():
            continue
        shutil.copytree(source_images, dst / "rgb" / "images" / split, dirs_exist_ok=True)
        shutil.copytree(src / "rgb" / "labels" / split, dst / "rgb" / "labels" / split, dirs_exist_ok=True)
        shutil.copytree(src / "rgb" / "labels" / split, dst / "rgbd" / "labels" / split, dirs_exist_ok=True)
        out_images = dst / "rgbd" / "images" / split
        out_images.mkdir(parents=True, exist_ok=True)
        count = 0
        # 腹部原图是 png、手心是 jpg，两种都收
        source_list = sorted(list(source_images.glob("*.png")) + list(source_images.glob("*.jpg")))
        for path in source_list:
            rgb = cv2.imread(str(path), cv2.IMREAD_COLOR)
            result = model.predict(str(path), verbose=False)[0]
            generated = np.asarray(result.depth.data.cpu()).squeeze().astype(np.float32)
            if generated.shape != rgb.shape[:2]:
                generated = cv2.resize(generated, (rgb.shape[1], rgb.shape[0]),
                                       interpolation=cv2.INTER_LINEAR)
            channel = to_channel(generated, args.low_percentile, args.high_percentile)
            # 输出一律写 .png：JPEG 不支持 4 通道，保留原扩展名会把第 4 层丢掉
            cv2.imwrite(str(out_images / f"{path.stem}.png"), np.dstack([rgb, channel]))
            count += 1
        print(f"{split}: {count} 张")
        total += count

    write_yaml(dst / "rgb", 3, args.class_name, args.kpt)
    write_yaml(dst / "rgbd", 4, args.class_name, args.kpt)
    print(f"合计 {total} 张 → {dst}")
    print(f"训练用 yaml: {dst / 'rgbd' / 'data_rgbd_gen.yaml'}")


if __name__ == "__main__":
    main()
