r"""把「第 1 层输出 ×keep」折进第 2 层第一个卷积的输入权重（不改结构、不重训）。

原理：
    第 1 层输出只有一个下游（第 2 层 C3k2，继承 C2f），而 C2f.forward 第一件事就是 cv1(x)。
    所以对下游卷积来说：
        喂减半的输入      W · (keep·a) + b
        改权重吃原输入   (keep·W) · a + b
    两者恒等（b 一般还是 0，因为 Conv 后接 BN，bias=False）。
    于是"运行时把上一层输出乘 keep"可以等价折进下一层的输入通道权重：
    部署时零额外开销，也不需要重训。

用法（在 MultiYoloRGB 根目录）：
    .venv\Scripts\python.exe fold_l1_gain.py
    .venv\Scripts\python.exe fold_l1_gain.py --keep 0.45 --dst train_weight\yolo11n_rgbd_4ch_L1x0.45.pt
    .venv\Scripts\python.exe fold_l1_gain.py --channels 9            # 只折单个通道（1-based）

输出默认写到 train_weight\<原名>_L1x<keep>.pt，绝不覆盖源权重。
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="把第 1 层输出增益折进第 2 层输入权重")
    parser.add_argument("--src", default=r"train_weight\yolo11n_rgbd_4ch.pt",
                        help="源权重（不会被修改）")
    parser.add_argument("--dst", default=None,
                        help="输出权重；默认 <原名>_L1x<keep>.pt")
    parser.add_argument("--keep", type=float, default=0.5, help="第 1 层输出要乘的系数")
    parser.add_argument("--layer", type=int, default=1, help="被减半的层号（默认第 1 层）")
    parser.add_argument("--consumer", type=int, default=2, help="消费该层输出的层号（默认第 2 层）")
    parser.add_argument("--channels", nargs="+", type=int, default=None,
                        help="只折这些输出通道（1-based）；默认全折")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已存在的输出文件")
    return parser.parse_args()


def md5_of(path: Path) -> str:
    """算文件指纹，用来证明源权重没被动过。"""
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def consumer_input_weight(model: torch.nn.Module, layer: int, consumer: int) -> tuple[torch.nn.Module, torch.nn.Parameter]:
    """取出「第 consumer 层第一个卷积」的权重，并校验通道数与第 layer 层输出一致。"""
    sequence = model.model
    source = sequence[layer]
    if not hasattr(source, "conv") or not hasattr(source.conv, "out_channels"):
        raise TypeError(f"第 {layer} 层不是 Conv（{type(source).__name__}），无法按输出通道折入")
    target = sequence[consumer]
    cv1 = getattr(target, "cv1", None)
    if cv1 is None or not hasattr(cv1, "conv"):
        raise TypeError(f"第 {consumer} 层没有 cv1.conv（{type(target).__name__}）")
    weight = cv1.conv.weight
    source_channels = int(source.conv.out_channels)
    if weight.shape[1] != source_channels:
        raise ValueError(f"通道数对不上：第 {consumer} 层输入 {weight.shape[1]}，"
                         f"第 {layer} 层输出 {source_channels}")
    return target, weight


def main() -> None:
    args = parse_args()
    src = Path(args.src).expanduser()
    if not src.is_file():
        raise FileNotFoundError(src)
    if not 0.0 < args.keep <= 1.0:
        raise ValueError("--keep 建议在 (0, 1]：这里做的是衰减，不是放大")
    dst = Path(args.dst).expanduser() if args.dst else src.with_name(
        f"{src.stem}_L1x{args.keep:g}{src.suffix}")
    if dst.resolve() == src.resolve():
        raise ValueError("输出路径不能等于源权重")
    if dst.exists() and not args.overwrite:
        raise FileExistsError(f"{dst} 已存在；要覆盖加 --overwrite")

    src_md5 = md5_of(src)
    print("=" * 96)
    print("把第 1 层输出增益折进下游卷积的输入权重")
    print(f"源权重: {src}  md5={src_md5}")
    print(f"层设置: 第 {args.layer} 层输出 × {args.keep}  → 折进第 {args.consumer} 层 cv1 的输入权重")
    print("=" * 96)

    yolo = YOLO(str(src))
    torch_model = yolo.model
    target, weight = consumer_input_weight(torch_model, args.layer, args.consumer)
    total_channels = int(weight.shape[1])
    if args.channels:
        indices = [channel - 1 for channel in args.channels]
        for index in indices:
            if not 0 <= index < total_channels:
                raise ValueError(f"通道 {index + 1} 超出范围 [1, {total_channels}]")
    else:
        indices = list(range(total_channels))
    print(f"目标卷积: 第 {args.consumer} 层 {type(target).__name__}.cv1.conv  "
          f"weight={tuple(weight.shape)}  待折通道={[i + 1 for i in indices]}")

    before = weight.detach().clone()
    with torch.no_grad():
        for index in indices:
            weight.data[:, index, :, :] *= args.keep
    changed = (weight.detach() - before)
    for index in indices:
        original_norm = float(before[:, index].norm())
        new_norm = float(weight.detach()[:, index].norm())
        if original_norm > 0:
            ratio = new_norm / original_norm
            if abs(ratio - args.keep) > 1e-5:
                raise RuntimeError(f"通道 {index + 1} 缩放异常：{ratio:.6f} != {args.keep}")
    untouched = [i for i in range(total_channels) if i not in indices]
    if untouched:
        drift = float(changed[:, untouched].abs().max()) if changed[:, untouched].numel() else 0.0
        print(f"未折通道最大改动量: {drift:.3e}（应为 0）")

    args_snapshot = dict(yolo.ckpt.get("train_args", {}) or {})
    checkpoint = {
        "model": torch_model,
        "train_args": args_snapshot,
        "epoch": -1,
        "task": "pose",
        "names": torch_model.names,
        "nc": torch_model.yaml.get("nc"),
        "kpt_shape": torch_model.yaml.get("kpt_shape"),
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, dst)

    # 存完再读回来，确认这份权重真能加载、而且数值确实改了
    reloaded = YOLO(str(dst))
    _, reloaded_weight = consumer_input_weight(reloaded.model, args.layer, args.consumer)
    drift = float((reloaded_weight.detach() - weight.detach()).abs().max())
    reloaded_md5 = md5_of(dst)
    print("-" * 96)
    print(f"已保存: {dst}  md5={reloaded_md5}  size={dst.stat().st_size} B")
    print(f"回读校验: 与内存中权重最大差 {drift:.3e}（应为 0）")
    print(f"源权重未改动: md5 仍为 {md5_of(src)}（与开头一致: {md5_of(src) == src_md5}）")


if __name__ == "__main__":
    main()
