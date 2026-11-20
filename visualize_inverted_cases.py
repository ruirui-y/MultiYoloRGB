r"""训练期反转深度模型的可视化对比：原图 / 原模型预测 / 反转模型预测。

对每张图算两个定位误差：
    原模型吃原始深度   ← 基准
    反转模型吃反转深度 ← 待展示的方法
然后挑"改善最大"的 3 张与"变差最多/未命中"的 2 张，画成三栏图。

用法：
    .venv\Scripts\python.exe visualize_inverted_cases.py
    .venv\Scripts\python.exe visualize_inverted_cases.py --top 5 --bottom 3
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import ablate_c2psa as ac

ORIGINAL_ROOT = Path(r"H:\YJJ\Yolo_RGBD\Resource\session1_200146\dataset\hand_common")
INVERTED_ROOT = Path(r"H:\YJJ\Yolo_RGBD\Resource\session1_200146\dataset\hand_common_inv")
GT_COLOR = (0, 255, 0)      # 绿 = 标准答案
BASE_COLOR = (0, 165, 255)  # 橙 = 原模型
INV_COLOR = (255, 0, 255)   # 紫 = 反转模型


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="反转深度模型的可视化对比")
    parser.add_argument("--weights-dir", default=str(ac._REPO / "train_weight"))
    parser.add_argument("--base-model", default="yolo11n_rgbd_4ch")
    parser.add_argument("--inv-model", default="yolo11n_rgbd_4ch_inv")
    parser.add_argument("--split", default="test")
    parser.add_argument("--conf", type=float, default=0.01)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--top", type=int, default=3, help="展示改善最多的几张")
    parser.add_argument("--bottom", type=int, default=2, help="展示变差最多/未命中的几张")
    parser.add_argument("--output-dir", default=str(ac._REPO / "runs" / "inv_visualizations"))
    return parser.parse_args()


def draw(image: np.ndarray, gt: dict[str, Any], prediction: dict[str, Any],
         title: str, color: tuple[int, int, int]) -> np.ndarray:
    """在原图上画 GT（绿）与一组预测（指定颜色），并写标题。"""
    canvas = image[:, :, :3].copy()
    gt_box = np.asarray(gt["box"])
    cv2.rectangle(canvas, (int(gt_box[0]), int(gt_box[1])), (int(gt_box[2]), int(gt_box[3])), GT_COLOR, 2)
    for keypoint in gt["keypoints"]:
        if keypoint[2] > 0:
            cv2.circle(canvas, (int(keypoint[0]), int(keypoint[1])), 4, GT_COLOR, -1)
    if prediction["matched"]:
        box = np.asarray(prediction["box"])
        cv2.rectangle(canvas, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        for point in np.asarray(prediction["keypoints"]):
            cv2.drawMarker(canvas, (int(point[0]), int(point[1])), color, cv2.MARKER_CROSS, 12, 2)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(canvas, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return canvas


def main() -> None:
    args = parse_args()
    weights_dir = Path(args.weights_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    original_images = ac.stem_map(ORIGINAL_ROOT / "rgbd" / "images" / args.split)
    inverted_images = ac.stem_map(INVERTED_ROOT / "rgbd" / "images" / args.split)
    label_dir = ORIGINAL_ROOT / "rgb" / "labels" / args.split
    stems = sorted(s for s in original_images.keys() & inverted_images.keys()
                   if (label_dir / f"{s}.txt").is_file())

    base_model = ac.YOLO(str(weights_dir / f"{args.base_model}.pt"))
    inv_model = ac.YOLO(str(weights_dir / f"{args.inv_model}.pt"))

    rows: list[dict[str, Any]] = []
    for stem in stems:
        original = ac.read_image(original_images[stem], 4)
        inverted = ac.read_image(inverted_images[stem], 4)
        height, width = original.shape[:2]
        gt = ac.ev.load_pose_label(label_dir / f"{stem}.txt", width, height)

        base_result = ac.ev.run_pose_model(base_model, original, args.conf, args.imgsz)
        base_pred = ac.ev.find_best_prediction(base_result, gt["box"], args.match_iou)
        inv_result = ac.ev.run_pose_model(inv_model, inverted, args.conf, args.imgsz)
        inv_pred = ac.ev.find_best_prediction(inv_result, gt["box"], args.match_iou)

        base_nme = None
        inv_nme = None
        if base_pred["matched"]:
            try:
                base_nme = float(ac.ev.calculate_nme(gt["box"], gt["keypoints"], base_pred["keypoints"])["nme"])
            except ValueError:
                base_nme = None
        if inv_pred["matched"]:
            try:
                inv_nme = float(ac.ev.calculate_nme(gt["box"], gt["keypoints"], inv_pred["keypoints"])["nme"])
            except ValueError:
                inv_nme = None

        rows.append({"stem": stem, "base_nme": base_nme, "inv_nme": inv_nme,
                     "base_matched": bool(base_pred["matched"]),
                     "inv_matched": bool(inv_pred["matched"])})

        # 只在需要展示的图上花绘制成本
        delta = None
        if base_nme is not None and inv_nme is not None:
            delta = inv_nme - base_nme
        rows[-1]["delta"] = delta

    paired = [r for r in rows if r["base_nme"] is not None and r["inv_nme"] is not None]
    paired.sort(key=lambda r: r["delta"])
    best = paired[:args.top]
    worst = sorted(paired, key=lambda r: r["delta"], reverse=True)[:args.bottom]
    missed = [r for r in rows if r["base_matched"] and not r["inv_matched"]][:args.bottom]

    def render(selection: list[dict[str, Any]], tag: str) -> None:
        for row in selection:
            stem = row["stem"]
            original = ac.read_image(original_images[stem], 4)
            inverted = ac.read_image(inverted_images[stem], 4)
            height, width = original.shape[:2]
            gt = ac.ev.load_pose_label(label_dir / f"{stem}.txt", width, height)
            base_result = ac.ev.run_pose_model(base_model, original, args.conf, args.imgsz)
            base_pred = ac.ev.find_best_prediction(base_result, gt["box"], args.match_iou)
            inv_result = ac.ev.run_pose_model(inv_model, inverted, args.conf, args.imgsz)
            inv_pred = ac.ev.find_best_prediction(inv_result, gt["box"], args.match_iou)
            base_text = "base NME=%.4f" % row["base_nme"] if row["base_nme"] is not None else "base MISS"
            inv_text = "inv NME=%.4f" % row["inv_nme"] if row["inv_nme"] is not None else "inv MISS"
            panels = [
                draw(original, gt, base_pred, base_text, BASE_COLOR),
                draw(inverted, gt, inv_pred, "inverted-depth | " + inv_text, INV_COLOR),
            ]
            combined = np.hstack(panels)
            path = out_dir / f"{tag}_{stem[-17:]}.png"
            cv2.imwrite(str(path), combined)

    render(best, "best")
    render(worst, "worst")
    render(missed, "missed")

    with (out_dir / "per_image.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stem", "base_matched", "inv_matched",
                                                    "base_nme", "inv_nme", "delta"])
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 96)
    print(f"图片 {len(rows)} 张，conf={args.conf}，权重 {args.base_model} vs {args.inv_model}")
    print(f"两者都命中 {len(paired)} 张；平均误差 base={np.mean([r['base_nme'] for r in paired]):.6f} "
          f"inv={np.mean([r['inv_nme'] for r in paired]):.6f}")
    print("\n改善最多：")
    for row in best:
        print(f"  {row['stem'][-17:]}  base={row['base_nme']:.4f} -> inv={row['inv_nme']:.4f} "
              f"({row['delta']:+.4f})")
    print("\n变差最多：")
    for row in worst:
        print(f"  {row['stem'][-17:]}  base={row['base_nme']:.4f} -> inv={row['inv_nme']:.4f} "
              f"({row['delta']:+.4f})")
    if missed:
        print("\n原地命中、反转后未命中：")
        for row in missed:
            print(f"  {row['stem'][-17:]}")
    print(f"\n图已保存: {out_dir}")


if __name__ == "__main__":
    main()
