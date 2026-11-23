r"""腹部实验的图：数据样例、九模型成绩、两组深度敏感度热力图。

产出（docs/）：
    fig-14-abdomen-depth.png       RGB / 真实深度 / 生成深度 三栏对照
    fig-15-abdomen-scores.png      九组模型成绩条形图（3 代 × 3 种输入）
    fig-16-abdomen-real-heat.png   真实深度：3 代模型 × 8 项改动的敏感度
    fig-17-abdomen-gen-heat.png    生成深度：同上

数据来源：
    runs/test_abdomen_all / runs/test_abdomen_gen / runs/test_abdomen_control
    runs/abdomen_battery/*/summary.csv、runs/abdomen_battery_gen/*/summary.csv

用法：
    .venv\Scripts\python.exe render_abdomen_figures.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(r"H:\YJJ\Yolo_RGBD\MultiYoloRGB")
DOCS = REPO / "docs"
REAL_DS = Path(r"H:\YJJ\Yolo_RGBD\Resource\abdomen\datasets\abdomen_dataset")
GEN_DS = Path(r"H:\YJJ\Yolo_RGBD\Resource\abdomen\datasets\abdomen_dataset_gen")

# (显示名, 定位误差, 命中, 分组)
SCORES = [
    ("26 + 生成深度", 0.013691, 20, "生成深度"),
    ("26 + RGB", 0.015003, 20, "只有 RGB"),
    ("11 + 生成深度", 0.016807, 20, "生成深度"),
    ("11 + 真实深度", 0.020831, 20, "真实深度"),
    ("v8 + 真实深度", 0.023182, 20, "真实深度"),
    ("v8 + RGB", 0.024689, 20, "只有 RGB"),
    ("11 + RGB", 0.028140, 20, "只有 RGB"),
    ("26 + 真实深度", 0.028887, 20, "真实深度"),
    ("v8 + 生成深度", 0.034434, 20, "生成深度"),
    ("11 + RGB（继续训，对照）", 0.042892, 20, "对照"),
    ("11 + 真实深度（通用起点，坏解）", 0.095627, 20, "对照"),
]
COLOURS = {"只有 RGB": "#adb5bd", "真实深度": "#4dabf7", "生成深度": "#f76707", "对照": "#ced4da"}

CONDITIONS = [("depth_invert", "深度整体反转"), ("depth_flat_mean", "填成一块平均值"),
              ("depth_zero", "深度清零"), ("depth_edges", "只留深度边缘"),
              ("depth_mirror_ud", "上下镜像"), ("depth_shuffle", "像素整体打乱"),
              ("l1x0.5_only", "第 1 层 ×0.5"), ("depth_as_gray", "换成原图黑白")]

REAL_TABLES = [("YOLOv8", REPO / "runs/abdomen_battery/yolov8n_abdomen_rgbd/summary.csv"),
               ("YOLO11", REPO / "runs/abdomen_depth_check_v2/summary.csv"),
               ("YOLO26", REPO / "runs/abdomen_battery/yolo26n_abdomen_rgbd/summary.csv")]
GEN_TABLES = [("YOLOv8", REPO / "runs/abdomen_battery_gen/yolov8n_abdomen_gen/summary.csv"),
              ("YOLO11", REPO / "runs/abdomen_battery_gen/yolo11n_abdomen_gen/summary.csv"),
              ("YOLO26", REPO / "runs/abdomen_battery_gen/yolo26n_abdomen_gen/summary.csv")]


def setup_font() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False


def label_panel(panel: np.ndarray, text: str, font) -> np.ndarray:
    canvas = np.zeros((panel.shape[0] + 30, panel.shape[1], 3), np.uint8)
    canvas[30:] = panel
    from PIL import Image, ImageDraw
    image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(image).text((6, 3), text, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def draw_samples() -> None:
    from PIL import ImageFont
    font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 17)
    rows = []
    for split, names in (("train", None),):
        real_images = sorted((REAL_DS / "rgb" / "images" / split).glob("*.png"))[:2]
        for real_path in real_images:
            gen_path = GEN_DS / "rgbd" / "images" / split / real_path.name
            rgb = cv2.imread(str(real_path), cv2.IMREAD_COLOR)
            real = cv2.imread(str(REAL_DS / "rgbd" / "images" / split / real_path.name), cv2.IMREAD_UNCHANGED)
            gen = cv2.imread(str(gen_path), cv2.IMREAD_UNCHANGED)
            if real is None or gen is None:
                continue
            panels = [
                label_panel(cv2.resize(rgb, (420, 236)), "RGB", font),
                label_panel(cv2.resize(cv2.cvtColor(real[:, :, 3], cv2.COLOR_GRAY2BGR), (420, 236)),
                            "传感器真实深度（亮=远）", font),
                label_panel(cv2.resize(cv2.cvtColor(gen[:, :, 3], cv2.COLOR_GRAY2BGR), (420, 236)),
                            "26 生成深度", font),
            ]
            rows.append(np.hstack(panels))
    if rows:
        cv2.imwrite(str(DOCS / "fig-14-abdomen-depth.png"), np.vstack(rows))
        print("已生成 fig-14-abdomen-depth.png")


def draw_scores() -> None:
    items = sorted(SCORES, key=lambda item: item[1])
    labels = [item[0] for item in items]
    values = [item[1] for item in items]
    colours = [COLOURS[item[3]] for item in items]
    figure, axis = plt.subplots(figsize=(9.6, 5.2))
    positions = np.arange(len(items))
    axis.barh(positions, values, color=colours)
    for position, value, item in zip(positions, values, items):
        axis.text(value + 0.0008, position, f"{value:.6f}  ({item[2]}/20)", va="center", fontsize=8.5)
    axis.set_yticks(positions)
    axis.set_yticklabels(labels, fontsize=9)
    axis.invert_yaxis()
    axis.set_xlim(0, max(values) * 1.28)
    axis.set_xlabel("定位误差 NME（越小越好）")
    axis.set_title("腹部测试集（20 张）九组模型 + 两个对照", fontsize=11)
    axis.grid(axis="x", alpha=0.3)
    figure.tight_layout()
    figure.savefig(DOCS / "fig-15-abdomen-scores.png", dpi=150)
    plt.close(figure)
    print("已生成 fig-15-abdomen-scores.png")


def draw_heatmap(tables: list[tuple[str, Path]], title: str, output: str) -> None:
    loaded = []
    for name, path in tables:
        if path.is_file():
            loaded.append((name, {r["condition"]: r for r in csv.DictReader(open(path, encoding="utf-8-sig"))}))
        else:
            print(f"[缺失] {path}")
    values = np.full((len(CONDITIONS), len(loaded)), np.nan)
    texts = [["" for _ in loaded] for _ in CONDITIONS]
    for row, (key, _label) in enumerate(CONDITIONS):
        for col, (_name, table) in enumerate(loaded):
            row_data = table.get(key)
            if not row_data or row_data.get("delta_pct") in ("", None):
                texts[row][col] = "未测"
                continue
            hits = int(row_data.get("hits") or 0)
            delta = float(row_data["delta_pct"])
            if hits < 10:
                texts[row][col] = f"崩\n{hits}/20"
                continue
            values[row, col] = delta
            texts[row][col] = f"{delta:+.1f}%"

    limit = 60.0
    figure, axis = plt.subplots(figsize=(6.8, 6.6))
    image = axis.imshow(np.clip(values, -limit, limit), cmap="RdYlGn_r", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(np.arange(len(loaded)))
    axis.set_xticklabels([name for name, _t in loaded], fontsize=11)
    axis.set_yticks(np.arange(len(CONDITIONS)))
    axis.set_yticklabels([label for _k, label in CONDITIONS], fontsize=10)
    for row in range(len(CONDITIONS)):
        for col in range(len(loaded)):
            text = texts[row][col]
            if "崩" in text or "未测" in text:
                axis.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, facecolor="#ced4da", edgecolor="white"))
                axis.text(col, row, text, ha="center", va="center", fontsize=8.5, color="#495057")
            else:
                axis.text(col, row, text, ha="center", va="center", fontsize=10)
    axis.set_title(title, fontsize=11)
    figure.colorbar(image, ax=axis, label="定位误差变化 %（负数=变好）", shrink=0.8)
    axis.set_xticks(np.arange(-0.5, len(loaded)), minor=True)
    axis.set_yticks(np.arange(-0.5, len(CONDITIONS)), minor=True)
    axis.grid(which="minor", color="white", lw=1.4)
    axis.tick_params(which="minor", length=0)
    figure.tight_layout()
    figure.savefig(DOCS / output, dpi=150)
    plt.close(figure)
    print(f"已生成 {output}")


def main() -> None:
    setup_font()
    draw_samples()
    draw_scores()
    draw_heatmap(REAL_TABLES, "腹部·真实深度：三代模型对同一改动的反应", "fig-16-abdomen-real-heat.png")
    draw_heatmap(GEN_TABLES, "腹部·生成深度：三代模型对同一改动的反应", "fig-17-abdomen-gen-heat.png")


if __name__ == "__main__":
    main()
