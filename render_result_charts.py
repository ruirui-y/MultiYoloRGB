r"""补充两张图：所有训练产物的成绩对比、反转模型的案例对比。

产出：
    docs/fig-08-weight-ranking.png   各权重的定位误差对比（按从好到差排序）
    docs/fig-09-inv-cases.png        反转模型：改善最多 / 变差最多的案例

用法：
    .venv\Scripts\python.exe render_result_charts.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DOCS = Path(r"H:\YJJ\Yolo_RGBD\MultiYoloRGB\docs")
CASES = Path(r"H:\YJJ\Yolo_RGBD\MultiYoloRGB\runs\inv_visualizations")

# (显示名, 定位误差, 命中, 分组)
SCORES = [
    ("26-RGBD 原样", 0.059238, 100, "原始模型"),
    ("11-RGBD 反转训练", 0.056028, 97, "深度处理改进"),
    ("11-RGBD 反转深度\n(只在推理时反转)", 0.059484, 98, "深度处理改进"),
    ("11-RGBD 第1层×0.5", 0.059823, 100, "模型内部改动"),
    ("11-RGBD 置零重训", 0.060118, 92, "深度处理改进"),
    ("v8-RGBD 原样", 0.067529, 100, "原始模型"),
    ("11-RGB 原样", 0.069164, 100, "原始模型"),
    ("11-RGBD 原样", 0.070292, 100, "原始模型"),
    ("11-RGBD 折入后再训", 0.073084, 100, "模型内部改动"),
    ("26-RGB 原样", 0.079497, 100, "原始模型"),
    ("11-RGBD 中心化重训", 0.081842, 100, "深度处理改进"),
    ("v8-RGB 原样", 0.081902, 100, "原始模型"),
    ("v8-RGBD 反转训练", 0.086354, 100, "深度处理改进"),
]
COLOURS = {"原始模型": "#adb5bd", "模型内部改动": "#4dabf7", "深度处理改进": "#f76707"}


def setup_font() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False


def draw_ranking(out: Path) -> None:
    items = sorted(SCORES, key=lambda item: item[1])
    labels = [item[0] for item in items]
    values = [item[1] for item in items]
    colours = [COLOURS[item[3]] for item in items]
    figure, axis = plt.subplots(figsize=(11.2, 5.4))
    positions = np.arange(len(items))
    axis.barh(positions, values, color=colours)
    for position, value, item in zip(positions, values, items):
        axis.text(value + 0.001, position, f"{value:.4f}  命中 {item[2]}", va="center", fontsize=8.5)
    axis.set_yticks(positions)
    axis.set_yticklabels(labels, fontsize=8.5)
    axis.invert_yaxis()
    axis.set_xlabel("定位误差（越小越好）")
    axis.set_title("各模型在跨人测试集上的成绩（各自命中集，命中数标在右侧；越小越好）", fontsize=11)
    axis.set_xlim(0, max(values) * 1.22)
    axis.grid(axis="x", alpha=0.3)
    figure.tight_layout()
    figure.savefig(out / "fig-08-weight-ranking.png", dpi=150)
    plt.close(figure)


def draw_cases(out: Path) -> None:
    best = sorted(CASES.glob("best_*.png"))[:2]
    worst = sorted(CASES.glob("worst_*.png"))[:1]
    images = []
    for path in best + worst:
        image = cv2.imread(str(path))
        if image is not None:
            images.append(image)
    if not images:
        print("没有找到案例图，跳过 fig-09")
        return
    width = min(image.shape[1] for image in images)
    resized = [cv2.resize(image, (width, int(image.shape[0] * width / image.shape[1]))) for image in images]
    combined = np.vstack(resized)
    cv2.imwrite(str(out / "fig-09-inv-cases.png"), combined)
    print(f"案例图: {out / 'fig-09-inv-cases.png'}（{len(resized)} 组）")


def main() -> None:
    parser = argparse.ArgumentParser(description="成绩对比与案例图")
    parser.add_argument("--out-dir", default=str(DOCS / "figures"))
    args = parser.parse_args()
    setup_font()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    draw_ranking(out)
    draw_cases(out)
    print(f"已生成：{out / 'fig-08-weight-ranking.png'}")


if __name__ == "__main__":
    main()
