r"""把"第 4 通道的各种样子"和"深度的权重"画成图（图上直接标中文名与实测误差）。

产出：
    runs/depth_gallery/depth_variants.png   每种改动后第 4 通道长什么样（拼图，带标注）
    runs/depth_gallery/depth_weights.png    第 0 层里"听深度"的那组权重，在 0 / 1 / 2 倍下的样子
    runs/depth_gallery/variant_index.csv    编号对照（备用）

图上文字用中文字体（C:\Windows\Fonts\msyh.ttc）；读不到字体时退回英文编号。

用法：
    .venv\Scripts\python.exe render_depth_gallery.py
"""
from __future__ import annotations
import paths


import argparse
import csv
import glob
import os
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

import ablate_c2psa as ac
import depth_battery as db
import depth_separation_probe as dsp

DEFAULT_IMAGE_DIR = str(paths.DEFAULT_DATASET / "rgbd" / "images" / "test")
LABEL_DIR = str(paths.DEFAULT_DATASET / "rgb" / "labels" / "test")
BATTERY_CSV = r"H:\YJJ\Yolo_RGBD\MultiYoloRGB\runs\depth_battery\yolo11n_rgbd_4ch\summary.csv"
SEPARATION_CSV = r"H:\YJJ\Yolo_RGBD\MultiYoloRGB\runs\depth_separation_probe\summary.csv"
DIRECTION_CSV = r"H:\YJJ\Yolo_RGBD\MultiYoloRGB\runs\depth_direction\summary.csv"

PANEL_W, PANEL_H, HEADER = 340, 230, 46
COLUMNS = 5

SHORT_NAMES = {
    "baseline": "原样（基准）",
    "depth_invert": "数值全部反转",
    "depth_invert_shift+102": "反转后再抬高",
    "depth_shift+102": "整体 +102",
    "depth_shift+51": "整体 +51",
    "depth_shift-51": "整体 −51",
    "depth_zero": "深度填 0",
    "depth_const1": "全填最小值 1",
    "depth_const128": "全填 128",
    "depth_const255": "全填 255",
    "depth_flat_mean": "全填平均值",
    "depth_blur31": "模糊 31",
    "depth_blur15": "模糊 15",
    "depth_shuffle": "像素打乱",
    "depth_mirror_lr": "左右镜像",
    "depth_mirror_ud": "上下镜像",
    "depth_roll16": "右移 16 像素",
    "depth_roll64": "右移 64 像素",
    "depth_edges": "只留边缘",
    "depth_noise8": "加噪声",
    "depth_quant4": "只留 4 个等级",
    "depth_gamma0.5": "提亮 γ0.5",
    "depth_gamma2.0": "压暗 γ2",
    "depth_as_gray": "换成黑白图",
    "depth_as_rgbmean": "换成 RGB 平均",
    "rgb_invert": "颜色反转（对照）",
    "rgb_shift+51": "颜色 +51（对照）",
    "hand_high_only": "框内抬高",
    "bg_low_only": "框外压黑",
    "swap_means": "手/背景对调",
    "gt_near": "框内 255 / 框外 1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="画第 4 通道的图册（带中文标注）")
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--image", default=None)
    parser.add_argument("--output-dir", default=r"H:\YJJ\Yolo_RGBD\MultiYoloRGB\runs\depth_gallery")
    return parser.parse_args()


def load_font(size: int = 17):
    try:
        from PIL import ImageFont
        for candidate in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
            if Path(candidate).is_file():
                return ImageFont.truetype(candidate, size)
    except Exception:
        return None
    return None


def load_results(model: str = "yolo11n_rgbd_4ch") -> dict[str, tuple[float | None, int]]:
    """从两份 summary.csv 读回实测结果：{条件名: (误差变化%, 命中数)}。

    两份文件的列名不同：depth_battery 用 condition/delta_pct，
    depth_separation_probe 用 variant 且没有 delta_pct，需要按各自的基准行现算。
    两份里可能都混有别的模型（separation 那份同时跑了 v8），按 model 列过滤。
    """
    results: dict[str, tuple[float | None, int]] = {}
    for path in (BATTERY_CSV, SEPARATION_CSV, DIRECTION_CSV):
        if not os.path.isfile(path):
            continue
        all_rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
        rows = [r for r in all_rows if r.get("model") in (None, "", model)]
        baseline_nme = None
        parsed: list[tuple[str, float | None, int]] = []
        for row in rows:
            name = row.get("condition") or row.get("variant")
            if not name:
                continue
            try:
                nme = float(row["nme"]) if row.get("nme") not in ("", None) else None
            except ValueError:
                nme = None
            try:
                hits = int(row.get("hits") or 0)
            except ValueError:
                hits = 0
            if name in ("baseline", "true_depth") and nme:
                baseline_nme = nme
            delta = None
            if row.get("delta_pct") not in ("", None):
                delta = float(row["delta_pct"])
            parsed.append((name, delta, nme, hits))
            results[name] = (delta, hits)
        # 没有 delta_pct 的文件：用本文件里该模型的基准行现算
        if baseline_nme:
            for name, delta, nme, hits in parsed:
                if delta is None and nme and name not in ("baseline", "true_depth"):
                    results[name] = ((nme / baseline_nme - 1) * 100, hits)
    return results


def delta_text(name: str, results: dict[str, tuple[float | None, int]]) -> str:
    if name == "baseline":
        return "误差 0（基准）"
    if name not in results:
        return "误差未测"
    delta, hits = results[name]
    if delta is None:
        return "误差未测"
    text = f"误差 {delta:+.1f}%"
    if hits and hits < 100:
        text += f"（命中 {hits}）"
    return text


def draw_panel(panel: np.ndarray, code: str, title: str, subtitle: str, font) -> np.ndarray:
    """在图上加两行中文标注：编号+短名 / 实测误差。"""
    canvas = np.zeros((panel.shape[0] + HEADER, panel.shape[1], 3), dtype=np.uint8)
    canvas[HEADER:] = panel
    if font is None:
        cv2.putText(canvas, f"{code} {subtitle}", (6, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, canvas.shape[0] - 1), (80, 80, 80), 1)
        return canvas
    from PIL import Image, ImageDraw
    image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    line1 = f"{code}  {title}"
    draw.text((6, 3), line1, font=font, fill=(255, 255, 255))
    draw.text((6, HEADER - 20), subtitle, font=font, fill=(0, 220, 255))
    canvas = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, canvas.shape[0] - 1), (80, 80, 80), 1)
    return canvas


def depth_panel(channel: np.ndarray) -> np.ndarray:
    resized = cv2.resize(channel, (PANEL_W, PANEL_H), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)


def build_entries(image: np.ndarray, gt: dict[str, Any],
                  results: dict[str, tuple[float | None, int]]) -> tuple[list[tuple[str, str, str, np.ndarray]], list[dict[str, Any]]]:
    entries: list[tuple[str, str, str, np.ndarray]] = []
    index: list[dict[str, Any]] = []
    counter = 0

    for spec in db.build_conditions():
        if spec.l1_keep != 1.0:
            continue
        counter += 1
        code = f"A{counter:02d}"
        short = SHORT_NAMES.get(spec.name, spec.description)
        transformed = spec.transform(image)
        entries.append((code, short, delta_text(spec.name, results), depth_panel(transformed[:, :, 3])))
        index.append({"编号": code, "条件名": spec.name, "说明": short,
                      "实测": delta_text(spec.name, results)})

    mask = dsp.box_mask(image.shape[:2], np.asarray(gt["box"], dtype=np.float64))
    box_variants: list[tuple[str, Callable[[], np.ndarray]]] = [
        ("hand_high_only", lambda: dsp.hand_high_only(image, mask)),
        ("bg_low_only", lambda: dsp.background_low_only(image, mask)),
        ("swap_means", lambda: dsp.swap_region_means(image, mask)),
        ("gt_near", lambda: dsp.two_level_depth(image, mask, 255, 1)),
    ]
    for name, maker in box_variants:
        counter += 1
        code = f"B{counter:02d}"
        short = SHORT_NAMES.get(name, name)
        entries.append((code, short, delta_text(name, results), depth_panel(maker()[:, :, 3])))
        index.append({"编号": code, "条件名": name, "说明": short,
                      "实测": delta_text(name, results)})
    return entries, index


def compose(entries: list[tuple[str, str, str, np.ndarray]], font, header_title: str) -> np.ndarray:
    rows = []
    for start in range(0, len(entries), COLUMNS):
        chunk = entries[start:start + COLUMNS]
        panels = [draw_panel(panel, code, title, subtitle, font) for code, title, subtitle, panel in chunk]
        # 补齐到整行：尺寸跟着本行真实的图走（权重图和深度图尺寸不同）
        while len(panels) < COLUMNS:
            panels.append(np.zeros_like(panels[0]))
        rows.append(np.hstack(panels))
    body = np.vstack(rows)
    title_bar = np.zeros((40, body.shape[1], 3), dtype=np.uint8)
    if font is not None:
        from PIL import Image, ImageDraw
        image = Image.fromarray(cv2.cvtColor(title_bar, cv2.COLOR_BGR2RGB))
        ImageDraw.Draw(image).text((8, 8), header_title, font=font, fill=(255, 255, 255))
        title_bar = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return np.vstack([title_bar, body])


def render_weights(model_name: str, weight_dir: Path, scales: list[float]) -> list[tuple[str, str, str, np.ndarray]]:
    """把第 0 层里"看深度"的 16 个 3×3 数字画成小方格，返回可直接交给 compose 的条目。"""
    model = ac.YOLO(str(weight_dir / f"{model_name}.pt"))
    weight = model.model.model[0].conv.weight.detach().cpu().numpy()[:, 3]
    peak = float(np.abs(weight).max()) or 1.0
    captions = {0.0: "听不到深度", 1.0: "原样", 2.0: "放大一倍"}
    entries = []
    for scale in scales:
        cells = []
        for kernel in weight * scale:
            cell = np.clip((kernel / peak * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
            cells.append(cv2.resize(cell, (96, 96), interpolation=cv2.INTER_NEAREST))
        grid = np.vstack([np.hstack(cells[i:i + 4]) for i in range(0, 16, 4)])
        if grid.ndim == 2:
            grid = cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR)
        entries.append((f"x{scale:g}", captions.get(scale, f"x{scale:g}"),
                        "16 个 3×3 数字（灰=0 白=加分 黑=减分）", grid))
    return entries


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    font = load_font()
    results = load_results()

    image_path = Path(args.image) if args.image else Path(
        sorted(glob.glob(os.path.join(args.image_dir, "*.png")))[0])
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None or image.shape[2] != 4:
        raise ValueError(f"不是 4 通道图: {image_path}")
    height, width = image.shape[:2]
    stem = os.path.splitext(image_path.name)[0]
    gt = ac.ev.load_pose_label(Path(LABEL_DIR) / f"{stem}.txt", width, height)

    entries, index = build_entries(image, gt, results)
    gallery = compose(entries, font,
                      f"第 4 通道（深度）在各种改动下的样子    样本：{stem}    "
                      f"亮=远 暗=近    误差为 100 张跨人测试集上的实测变化（负数=变好）")
    variant_path = out_dir / "depth_variants.png"
    cv2.imwrite(str(variant_path), gallery)

    weights = compose(render_weights("yolo11n_rgbd_4ch", ac._REPO / "train_weight", [0.0, 1.0, 2.0]),
                      font, "深度的权重：把下面 16 个 3×3 数字改掉，等于调整网络听深度的音量")
    weights_path = out_dir / "depth_weights.png"
    cv2.imwrite(str(weights_path), weights)

    with (out_dir / "variant_index.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["编号", "条件名", "说明", "实测"])
        writer.writeheader()
        writer.writerows(index)

    print(f"样本图: {image_path.name}")
    print(f"图册: {variant_path}  ({len(entries)} 个面板)")
    print(f"权重图: {weights_path}")
    for row in index:
        print(f"  {row['编号']}  {row['说明']:<14} {row['实测']}")


if __name__ == "__main__":
    main()
