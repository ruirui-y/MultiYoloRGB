r"""三代 YOLO 的深度敏感度热力图。

9 个深度改动 × 3 个模型，绿色=变好、红色=变差、灰色=检出崩掉（不可比）。

数据：runs/depth_battery/<model>/summary.csv

用法：
    .venv\Scripts\python.exe render_generation_compare.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(r"H:\YJJ\Yolo_RGBD\MultiYoloRGB")
DOCS = ROOT / "docs"
MODELS = [("YOLOv8", ROOT / "runs/depth_battery/yolov8n_rgbd_4ch/summary.csv"),
          ("YOLO11", ROOT / "runs/depth_battery/yolo11n_rgbd_4ch/summary.csv"),
          ("YOLO26", ROOT / "runs/depth_battery/yolo26n_rgbd_4ch/summary.csv")]
CONDITIONS = [("depth_invert", "深度整体反转"),
              ("depth_flat_mean", "填成一块平均值"),
              ("depth_const255", "全填 255"),
              ("depth_zero", "深度清零"),
              ("depth_edges", "只留深度边缘"),
              ("depth_mirror_ud", "上下镜像"),
              ("depth_mirror_lr", "左右镜像"),
              ("depth_shuffle", "像素整体打乱"),
              ("l1x0.5_only", "第 1 层输出 ×0.5")]
# 命中低于 60 才算检出崩掉；60~99 只把命中数标出来
MIN_HITS = 60


def load(path: Path) -> dict[str, dict[str, str]]:
    return {row["condition"]: row for row in csv.DictReader(open(path, encoding="utf-8-sig"))}


def main() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    tables = {name: load(path) for name, path in MODELS}

    values = np.full((len(CONDITIONS), len(MODELS)), np.nan)
    labels = [["" for _ in MODELS] for _ in CONDITIONS]
    for row_index, (key, _label) in enumerate(CONDITIONS):
        for col_index, (name, _path) in enumerate(MODELS):
            row = tables[name].get(key)
            if not row:
                labels[row_index][col_index] = "未测"
                continue
            hits = int(row.get("hits") or 0)
            delta = row.get("delta_pct")
            if hits < MIN_HITS or delta in ("", None):
                labels[row_index][col_index] = f"崩\n{hits}/100"
                continue
            value = float(delta)
            values[row_index, col_index] = value
            labels[row_index][col_index] = f"{value:+.1f}%\n{hits}/100"

    limit = 30.0
    figure, axis = plt.subplots(figsize=(7.6, 7.4))
    image = axis.imshow(np.clip(values, -limit, limit), cmap="RdYlGn_r",
                        vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(np.arange(len(MODELS)))
    axis.set_xticklabels([name for name, _p in MODELS], fontsize=11)
    axis.set_yticks(np.arange(len(CONDITIONS)))
    axis.set_yticklabels([label for _k, label in CONDITIONS], fontsize=10)
    for row_index in range(len(CONDITIONS)):
        for col_index in range(len(MODELS)):
            text = labels[row_index][col_index]
            if "崩" in text or "未测" in text:
                axis.add_patch(plt.Rectangle((col_index - 0.5, row_index - 0.5), 1, 1,
                                             facecolor="#ced4da", edgecolor="white"))
                axis.text(col_index, row_index, text, ha="center", va="center",
                          fontsize=8.5, color="#495057")
            else:
                axis.text(col_index, row_index, text, ha="center", va="center", fontsize=10)
    axis.set_title("同一个深度改动，三代模型的反应（绿=变好，红=变差）", fontsize=11.5)
    figure.colorbar(image, ax=axis, label="定位误差变化 %（负数=变好）", shrink=0.8)
    axis.set_xticks(np.arange(-0.5, len(MODELS)), minor=True)
    axis.set_yticks(np.arange(-0.5, len(CONDITIONS)), minor=True)
    axis.grid(which="minor", color="white", lw=1.4)
    axis.tick_params(which="minor", length=0)
    figure.tight_layout()
    output = DOCS / "figures" / "fig-12-generation-compare.png"
    figure.savefig(output, dpi=150)
    plt.close(figure)
    print(f"已生成: {output}")


if __name__ == "__main__":
    main()
