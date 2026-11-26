r"""手心实验的图：九模型成绩、三部位深度敏感度对照。

产出（docs/figures/）：
    fig-18-palm-scores.png        手心九组模型成绩
    fig-19-palm-vs-others.png     手背 / 腹部 / 手心 的深度敏感度对照

数据来源：
    runs/test_palm_3x3、runs/test_palm_gen、runs/test_palm_zero
    runs/palm_battery/*/summary.csv、runs/palm_battery_gen/*/summary.csv
    runs/depth_battery/*/summary.csv（手背）、runs/abdomen_battery/*/summary.csv（腹部）

用法：
    .venv\Scripts\python.exe render_palm_figures.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(r"H:\YJJ\Yolo_RGBD\MultiYoloRGB")
FIGURES = REPO / "docs" / "figures"

# (显示名, NME, 分组)
SCORES = [
    ("26 + 真实深度（A 通用起点）", 0.131841, "真实深度"),
    ("11 + 真实深度（A 通用起点）", 0.132386, "真实深度"),
    ("11 + 真实深度（B 手心权重）", 0.132734, "真实深度"),
    ("v8 + 真实深度（B 手心权重）", 0.135161, "真实深度"),
    ("26 + 深度全零重训", 0.135317, "对照"),
    ("26 + 真实深度（B 手心权重）", 0.136620, "真实深度"),
    ("v8 + 生成深度", 0.138979, "生成深度"),
    ("11 + 生成深度", 0.144862, "生成深度"),
    ("v8 + 真实深度（A 通用起点）", 0.155224, "真实深度"),
    ("26 + 生成深度", 0.154748, "生成深度"),
    ("v8 只有 RGB", 0.129479, "只有 RGB"),
    ("11 只有 RGB", 0.141361, "只有 RGB"),
    ("26 只有 RGB", 0.159087, "只有 RGB"),
]
COLOURS = {"只有 RGB": "#adb5bd", "真实深度": "#4dabf7",
           "生成深度": "#f76707", "对照": "#fa5252"}

# 三部位：深度清零 / 反转 的影响
CONDITIONS = [
    ("depth_zero", "深度清零"),
    ("depth_invert", "深度整体反转"),
    ("depth_flat_mean", "填成一块平均值"),
    ("depth_shuffle", "像素整体打乱"),
]


def setup_font() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False


def load(path: Path) -> dict:
    if not path.is_file():
        return {}
    return {r["condition"]: r for r in csv.DictReader(open(path, encoding="utf-8-sig"))}


def draw_scores() -> None:
    items = sorted(SCORES, key=lambda item: item[1])
    labels = [item[0] for item in items]
    values = [item[1] for item in items]
    colours = [COLOURS[item[2]] for item in items]
    figure, axis = plt.subplots(figsize=(9.8, 5.6))
    positions = np.arange(len(items))
    axis.barh(positions, values, color=colours)
    for position, value in zip(positions, values):
        axis.text(value + 0.0015, position, f"{value:.6f}", va="center", fontsize=8.5)
    axis.set_yticks(positions)
    axis.set_yticklabels(labels, fontsize=9)
    axis.invert_yaxis()
    axis.set_xlim(0, max(values) * 1.16)
    axis.set_xlabel("定位误差 NME（越小越好）")
    axis.set_title("手心测试集（100 张，跨人）十三组模型", fontsize=11)
    axis.grid(axis="x", alpha=0.3)
    figure.tight_layout()
    figure.savefig(FIGURES / "fig-18-palm-scores.png", dpi=150)
    plt.close(figure)
    print("已生成 fig-18-palm-scores.png")


def draw_cross_part() -> None:
    """三部位 × 四项深度改动的横向对照。"""
    # 第三个元素是该部位的测试集张数（命中数要按这个归一，不能写死 50）
    groups = [
        ("手背\nv8/11/26 真实深度", 100, [
            REPO / "runs/depth_battery/yolov8n_rgbd_4ch/summary.csv",
            REPO / "runs/depth_battery/yolo11n_rgbd_4ch/summary.csv",
            REPO / "runs/depth_battery/yolo26n_rgbd_4ch/summary.csv",
        ]),
        ("腹部\nv8/11/26 真实深度", 20, [
            REPO / "runs/abdomen_battery/yolov8n_abdomen_rgbd/summary.csv",
            REPO / "runs/abdomen_depth_check_v2/summary.csv",
            REPO / "runs/abdomen_battery/yolo26n_abdomen_rgbd/summary.csv",
        ]),
        ("手心\nv8-B/11-A/26-A 真实深度", 100, [
            REPO / "runs/palm_battery/yolov8n_palm_rgbd_B/summary.csv",
            REPO / "runs/palm_battery/yolo11n_palm_rgbd_A/summary.csv",
            REPO / "runs/palm_battery/yolo26n_palm_rgbd_A/summary.csv",
        ]),
    ]

    width = 0.20
    figure, axis = plt.subplots(figsize=(9.8, 4.8))
    for offset, (key, title) in enumerate(CONDITIONS):
        values = []
        for _name, total, paths in groups:
            row = []
            for path in paths:
                table = load(path)
                record = table.get(key)
                if record is None or record.get("delta_pct") in ("", None):
                    row.append(np.nan)
                    continue
                hits = int(record.get("hits") or 0)
                # 命中低于一半视为崩掉，误差不可比，剔除
                row.append(np.nan if hits < total * 0.5 else float(record["delta_pct"]))
            values.append(np.nanmean(row) if any(not np.isnan(v) for v in row) else np.nan)
        positions = np.arange(len(groups)) + (offset - 1.5) * width
        axis.bar(positions, values, width, label=title)
        for position, value in zip(positions, values):
            if not np.isnan(value) and abs(value) > 0.5:
                axis.text(position, value + (1.2 if value >= 0 else -2.6),
                          f"{value:+.1f}", ha="center", fontsize=8)

    axis.axhline(0, color="#495057", lw=1)
    axis.set_xticks(np.arange(len(groups)))
    axis.set_xticklabels([name for name, _t, _p in groups], fontsize=10)
    axis.set_ylabel("定位误差变化 %（正=变差）")
    axis.set_title("三个部位：三代模型对该改动影响的均值（命中不足一半的剔除）", fontsize=11)
    axis.legend(fontsize=9)
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(FIGURES / "fig-19-palm-vs-others.png", dpi=150)
    plt.close(figure)
    print("已生成 fig-19-palm-vs-others.png")


def main() -> None:
    setup_font()
    FIGURES.mkdir(parents=True, exist_ok=True)
    draw_scores()
    draw_cross_part()


if __name__ == "__main__":
    main()
