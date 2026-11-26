r"""把 g01 的 YOLO-Pose 标签画回图上，检查关键点编号有没有串。

产出：
    每张图一张拼图（原图 + 编号标注），抽 N 张
    一份编号统计（每个编号出现的坐标范围）

用法：
    .venv\Scripts\python.exe check_palm_labels.py
    .venv\Scripts\python.exe check_palm_labels.py --count 8
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

RAW = Path(r"H:\YJJ\Yolo_RGBD\Resource\hand_palm\raw\palm_0916\g01")
OUT = Path(r"H:\YJJ\Yolo_RGBD\MultiYoloRGB\runs\palm_label_check")

COLOURS = [
    (0, 0, 255), (0, 165, 255), (0, 255, 255),
    (0, 255, 0), (255, 200, 0), (255, 0, 0), (255, 0, 255),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="手心标签编号检查")
    parser.add_argument("--raw", default=str(RAW))
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--count", type=int, default=6, help="抽多少张画图")
    return parser.parse_args()


def draw(image_path: Path, label_path: Path) -> np.ndarray | None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    height, width = image.shape[:2]
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    parts = text.split()
    # 格式：cls cx cy w h  kx1 ky1 v1  kx2 ky2 v2 ...
    values = [float(v) for v in parts]
    keypoints = values[5:]
    for index in range(0, len(keypoints) - 2, 3):
        x_norm, y_norm, visible = keypoints[index], keypoints[index + 1], keypoints[index + 2]
        if visible <= 0:
            continue
        px, py = int(x_norm * width), int(y_norm * height)
        number = index // 3
        colour = COLOURS[number % len(COLOURS)]
        cv2.circle(image, (px, py), 9, colour, -1)
        cv2.circle(image, (px, py), 9, (255, 255, 255), 1)
        cv2.putText(image, str(number), (px - 5, py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, str(number), (px - 5, py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def main() -> None:
    args = parse_args()
    raw = Path(args.raw)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    label_dir = raw / "image" / "labels"
    labels = sorted(label_dir.glob("*.txt"))
    images = {p.stem: p for p in (raw / "image").glob("*.jpg")}
    print(f"标签 {len(labels)} 个 / 图片 {len(images)} 张")

    points = []
    missing_image = 0
    for label_path in labels:
        image_path = images.get(label_path.stem)
        if image_path is None:
            missing_image += 1
            continue
        values = [float(v) for v in label_path.read_text(encoding="utf-8").split()]
        keypoints = values[5:]
        points.append([(keypoints[i], keypoints[i + 1], int(keypoints[i + 2]))
                       for i in range(0, len(keypoints) - 2, 3)])

    if not points:
        print("没有可分析的标签")
        return

    keypoint_count = {len(p) for p in points}
    print(f"每张图的点数：{keypoint_count}")
    print(f"有标签但缺图片：{missing_image}")

    array = np.array([[kp[2] for kp in p] for p in points])
    print(f"每个点的可见标记(0/1/2)：最小值 {array.min()} 最大值 {array.max()}")

    normalised = np.array([[[kp[0], kp[1]] for kp in p] for p in points])
    print("\n各编号的归一化坐标范围（x, y）与均值：")
    for index in range(normalised.shape[1]):
        xs = normalised[:, index, 0]
        ys = normalised[:, index, 1]
        print(f"  点 {index}: x {xs.min():.3f}~{xs.max():.3f} (均值 {xs.mean():.3f})"
              f"   y {ys.min():.3f}~{ys.max():.3f} (均值 {ys.mean():.3f})")

    picks = [labels[int(i)] for i in np.linspace(0, len(labels) - 1, min(args.count, len(labels))).astype(int)]
    panels = []
    for label_path in picks:
        image_path = images.get(label_path.stem)
        if image_path is None:
            continue
        drawn = draw(image_path, label_path)
        if drawn is None:
            continue
        panels.append(cv2.resize(drawn, (640, 360)))
        print(f"已画: {label_path.stem}")
    if panels:
        columns = 2
        while len(panels) % columns:
            panels.append(np.zeros_like(panels[0]))
        rows = [np.hstack(panels[i:i + columns]) for i in range(0, len(panels), columns)]
        cv2.imwrite(str(out / "g01_label_check.png"), np.vstack(rows))
        print(f"\n拼图: {out / 'g01_label_check.png'}")


if __name__ == "__main__":
    main()
