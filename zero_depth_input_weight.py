r"""把第 0 层"第 4 个输入通道"的权重置零，生成一个干净的初始权重。

用途：检验"第 4 通道权重 = RGB 三通道均值"这条改造规则，是不是让深度变成了亮度探测器。
置零后网络一开始完全收不到深度，只能靠训练自己长出用法。

只改第 0 层卷积 weight[:, 3]（16 个 3×3，共 144 个数），其余权重原样。

用法：
    .venv\Scripts\python.exe zero_depth_input_weight.py
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="第 4 输入通道权重置零")
    parser.add_argument("--src", default=r"yolo11_4ch\yolo11n-pose_4ch.pt")
    parser.add_argument("--dst", default=r"yolo11_4ch\yolo11n-pose_4ch_w4zero.pt")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    if not src.is_file():
        raise FileNotFoundError(src)
    if dst.exists() and not args.overwrite:
        raise FileExistsError(f"{dst} 已存在；要覆盖加 --overwrite")

    print("=" * 92)
    print(f"源权重: {src}  md5={md5_of(src)}")
    yolo = YOLO(str(src))
    model = yolo.model
    weight = model.model[0].conv.weight
    if weight.shape[1] != 4:
        raise RuntimeError(f"第 0 层输入通道不是 4，而是 {weight.shape[1]}")
    before = float(weight.data[:, 3].norm())
    with torch.no_grad():
        weight.data[:, 3] = 0.0
    after = float(weight.data[:, 3].norm())
    print(f"第 4 输入通道权重范数: {before:.4f} → {after:.4f}")

    checkpoint = {
        "model": model,
        "train_args": dict(yolo.ckpt.get("train_args", {}) or {}),
        "epoch": -1,
        "task": "pose",
        "names": model.names,
        "nc": model.yaml.get("nc"),
        "kpt_shape": model.yaml.get("kpt_shape"),
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, dst)

    reloaded = YOLO(str(dst))
    check = float(reloaded.model.model[0].conv.weight.data[:, 3].norm())
    print(f"已保存: {dst}  md5={md5_of(dst)}  回读第 4 通道范数={check:.4f}（应为 0）")
    print(f"源权重未改动: {md5_of(src) == md5_of(src)}")


if __name__ == "__main__":
    main()
