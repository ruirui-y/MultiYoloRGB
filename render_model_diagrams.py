r"""画三张说明图：网络层与改动点、折入示意、32 个输出通道。

产出（都写进 docs/）：
    docs/fig-05-network-map.png   23 层结构，标出我们动过的位置
    docs/fig-06-fold.png          "第 1 层输出 ×0.5" 与 "第 2 层输入权重 ×0.5" 为什么等价
    docs/fig-07-channels.png      32 个输出通道，标出 ch09 / ch13 / ch15

用法：
    .venv\Scripts\python.exe render_model_diagrams.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

from ultralytics import YOLO

DOCS = Path(r"H:\YJJ\Yolo_RGBD\MultiYoloRGB\docs")
FIGURES = DOCS / "figures"
WEIGHTS = Path(r"H:\YJJ\Yolo_RGBD\MultiYoloRGB\train_weight\yolo11n_rgbd_4ch.pt")
CHANNEL_CSV = Path(r"H:\YJJ\Yolo_RGBD\MultiYoloRGB\runs\l1_attribution_allch\channel_reference.csv")


def setup_font() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="画三张说明图")
    parser.add_argument("--out-dir", default=str(DOCS))
    return parser.parse_args()


def draw_network_map(out: Path) -> None:
    """23 层结构 + 我们动过的位置。"""
    model = YOLO(str(WEIGHTS)).model
    layers = [(i, type(layer).__name__) for i, layer in enumerate(model.model)]
    highlight = {
        0: ("深度从这里进来\n（第 4 个输入通道，144 个数）", "#ffd166"),
        1: ("这里的输出乘 0.5\n（最有效的一处，−14.9%）", "#90e0ef"),
        9: ("SPPF：乘 0.5 有 −6.7%\n但会掉命中", "#cdb4db"),
        10: ("C2PSA：乘 0.5 有 −8.5%\n但会掉命中；删掉会崩", "#cdb4db"),
        23: ("输出：框 + 7 个关键点", "#b7e4c7"),
    }
    figure, axis = plt.subplots(figsize=(9.6, 8.2))
    axis.set_xlim(0, 10)
    axis.set_ylim(-len(layers) - 0.6, 1.2)
    axis.axis("off")
    for index, (layer_index, name) in enumerate(layers):
        y = -index
        colour = highlight.get(layer_index, ("", "#e9ecef"))[1]
        axis.add_patch(Rectangle((0.4, y), 2.2, 0.72, facecolor=colour, edgecolor="#495057", lw=1))
        axis.text(1.5, y + 0.36, f"{layer_index}  {name}", ha="center", va="center", fontsize=9)
        if layer_index in highlight:
            text = highlight[layer_index][0]
            axis.add_patch(Rectangle((2.9, y), 4.4, 0.72, facecolor="white",
                                     edgecolor="#adb5bd", lw=0.8, ls="--"))
            axis.text(3.0, y + 0.36, text, ha="left", va="center", fontsize=8.5)
            axis.add_patch(FancyArrowPatch((2.65, y + 0.36), (2.88, y + 0.36),
                                           arrowstyle="->", mutation_scale=10, color="#495057"))
        if index < len(layers) - 1:
            axis.add_patch(FancyArrowPatch((1.5, y), (1.5, y - 0.28), arrowstyle="->",
                                           mutation_scale=9, color="#868e96"))
    axis.text(5.1, 0.8, "YOLO11n-Pose 的 23 层，以及我们动过的位置", fontsize=12, ha="left")
    axis.text(5.1, 0.28, "黄色=深度入口　蓝色=最有效的一处　紫色=次要　绿色=输出",
              fontsize=9, ha="left", color="#495057")
    figure.tight_layout()
    figure.savefig(out / "figures" / "fig-05-network-map.png", dpi=150)
    plt.close(figure)


def draw_fold(out: Path) -> None:
    """折入示意：把"第 1 层输出乘 0.5"搬到"第 2 层输入权重乘 0.5"。"""
    figure, axis = plt.subplots(figsize=(10.2, 4.4))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 5)
    axis.axis("off")

    def box(x, y, w, h, text, colour, fontsize=9.5):
        axis.add_patch(Rectangle((x, y), w, h, facecolor=colour, edgecolor="#495057", lw=1))
        axis.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)

    def arrow(x1, y1, x2, y2, text="", colour="#495057"):
        axis.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->",
                                       mutation_scale=12, color=colour))
        if text:
            axis.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.22, text, ha="center", fontsize=8.5, color=colour)

    # 上面一排：现在的做法（推理时改输出）
    box(0.3, 3.4, 1.5, 1.0, "输入\n4 层数字", "#e9ecef")
    box(2.3, 3.4, 1.6, 1.0, "第 1 层\n4→16", "#ffd166")
    box(4.4, 3.4, 1.8, 1.0, "输出乘 0.5\n（临时动作）", "#90e0ef")
    box(6.7, 3.4, 1.6, 1.0, "第 2 层\n16→32", "#e9ecef")
    box(8.8, 3.4, 1.4, 1.0, "继续往后", "#e9ecef")
    for x1, x2 in ((1.8, 2.3), (3.9, 4.4), (6.2, 6.7), (8.3, 8.8)):
        arrow(x1, 3.9, x2, 3.9)
    axis.text(0.3, 4.75, "做法 A（推理时临时改）：", fontsize=11, color="#c92a2a")

    # 下面一排：折进权重
    box(0.3, 1.3, 1.5, 1.0, "输入\n4 层数字", "#e9ecef")
    box(2.3, 1.3, 1.6, 1.0, "第 1 层\n4→16", "#ffd166")
    box(4.4, 1.3, 1.8, 1.0, "原样输出", "#e9ecef")
    box(6.7, 1.3, 1.6, 1.0, "第 2 层\n16→32\n权重×0.5", "#f4a261")
    box(8.8, 1.3, 1.4, 1.0, "继续往后", "#e9ecef")
    for x1, x2 in ((1.8, 2.3), (3.9, 4.4), (6.2, 6.7), (8.3, 8.8)):
        arrow(x1, 1.8, x2, 1.8)
    axis.text(0.3, 2.65, "做法 B（改权重，等价）：", fontsize=11, color="#2b8a3e")

    axis.text(10.4, 3.9, "结果 0.059823", fontsize=9, va="center")
    axis.text(10.4, 1.8, "结果 0.059823\n（逐位相同）", fontsize=9, va="center")

    axis.text(0.3, 0.45,
              "为什么等价：第 1 层输出只有一个下游（第 2 层），而第 2 层进门第一件事是乘加运算。\n"
              "把输入整体减半 ≡ 把那份乘法里的系数减半。所以改权重之后，推理时不需要任何额外动作。",
              fontsize=9, color="#495057")
    figure.tight_layout()
    figure.savefig(out / "figures" / "fig-06-fold.png", dpi=150)
    plt.close(figure)


def draw_channels(out: Path) -> None:
    """32 个输出通道与"和关键点热区的相似度"，标出 ch09/13/15。"""
    import csv
    rows = list(csv.DictReader(open(CHANNEL_CSV, encoding="utf-8-sig")))
    channels = [int(r["channel_1based"]) for r in rows]
    values = [float(r["gt_heat_spearman_mean"]) for r in rows]
    targets = {9, 13, 15}
    colours = ["#e5383b" if c in targets else "#adb5bd" for c in channels]
    figure, axis = plt.subplots(figsize=(10.4, 4.2))
    axis.bar(channels, values, color=colours)
    for channel, value in zip(channels, values):
        if channel in targets:
            axis.text(channel, value + 0.015, f"ch{channel:02d}", ha="center", fontsize=9, color="#e5383b")
    axis.axhline(0, color="#495057", lw=0.8)
    axis.set_xticks(channels)
    axis.set_xticklabels([f"{c}" for c in channels], fontsize=7)
    axis.set_xlabel("第 1 层的 32 个输出通道（编号）")
    axis.set_ylabel("和「关键点位置参考图」的相似度\n（1=完全一样，0=无关）")
    axis.set_title("哪些通道在「看关键点」：红色是我们重点查的三个", fontsize=11)
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(out / "figures" / "fig-07-channels.png", dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    setup_font()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    draw_network_map(out)
    draw_fold(out)
    draw_channels(out)
    print(f"已生成：{out / 'figures' / 'fig-05-network-map.png'}")
    print(f"已生成：{out / 'figures' / 'fig-06-fold.png'}")
    print(f"已生成：{out / 'figures' / 'fig-07-channels.png'}")


if __name__ == "__main__":
    main()
