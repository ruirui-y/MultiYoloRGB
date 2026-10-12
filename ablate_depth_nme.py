"""Depth 消融（NME 口径）：4 通道模型"用没用"以及"怎么用"第 4 通道。

第一轮（回答"有没有学会用深度"）
--------------------------------
    true_depth : 原始 4 通道
    zero_depth : 第 4 通道全部置 0
    判据：去掉深度后 NME 明显变差 → 模型确实在用深度

第二轮（回答"怎么用深度"）
--------------------------
    spatial_shuffle_depth : 打乱第 4 通道的像素位置，保留 min/max/mean/std/直方图
                            → 若 NME 变差，说明模型依赖深度的【空间结构】，而非"有没有深度"
    flat_valid_depth      : 保留有效 mask，有效区内部全填成该图深度均值
                            → 若 NME 明显变差，说明模型用了【连续几何】，不只是轮廓
    blurred_depth         : 对第 4 通道做 masked Gaussian blur（31×31）
                            → 若 NME 变化小，说明模型只用【粗尺度结构】；变大则依赖【局部高频】

变体算法与 `ablate_rgbd_depth.py` 同实现（保证口径一致），本脚本只把评估改成 NME。

用法
----
    cd /d H:\\YJJ\\Yolo_RGBD\\MultiYoloRGB
    .venv\\Scripts\\python.exe ablate_depth_nme.py
    .venv\\Scripts\\python.exe ablate_depth_nme.py --variants true_depth zero_depth

输出
----
    runs\\ablate_depth_nme\\summary.csv      每个权重 × 每个变体
    runs\\ablate_depth_nme\\per_image.csv    逐图明细
    runs\\ablate_depth_nme\\run_config.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_HERE: Path = Path(__file__).resolve()
_REPO: Path = _HERE.parent
_SIBLING: Path = _REPO.parent / "yolov8_rgbd_detection"
_EVAL_SCRIPT: Path = _SIBLING / "YJJ_Pose_Scripts" / "04_eval" / "eval_rgb_vs_rgbd_nme.py"
_DEFAULT_DATASET: Path = Path(
    r"H:\YJJ\Yolo_RGBD\Resource\session1_200146\dataset\hand_common"
)

# 先剥离"项目根"再 import（否则会命中项目内 vendored ultralytics 8.3.176）
_SIBLING_NORM: str = os.path.normcase(os.path.abspath(str(_SIBLING)))
sys.path[:] = [
    entry for entry in sys.path
    if entry and os.path.normcase(os.path.abspath(entry)) != _SIBLING_NORM
]

import ultralytics  # noqa: E402

_LOCKED_VERSION: str = ultralytics.__version__

if not _EVAL_SCRIPT.is_file():
    raise FileNotFoundError(f"第一阶段评估脚本不存在: {_EVAL_SCRIPT}")

sys.path.insert(0, str(_EVAL_SCRIPT.parent))
import eval_rgb_vs_rgbd_nme as ev  # noqa: E402

if ultralytics.__version__ != _LOCKED_VERSION:
    raise RuntimeError(
        f"ultralytics 被换掉了: {_LOCKED_VERSION} -> {ultralytics.__version__}"
    )

sys.path[:] = [
    entry for entry in sys.path
    if not (entry and os.path.normcase(os.path.abspath(entry)) == _SIBLING_NORM)
]

from ultralytics import YOLO  # noqa: E402


IMAGE_EXTS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
)
KEYPOINT_COUNT: int = 7
ALL_VARIANTS: tuple[str, ...] = (
    "true_depth",
    "zero_depth",
    "spatial_shuffle_depth",
    "flat_valid_depth",
    "blurred_depth",
)
VARIANT_LABELS: dict[str, str] = {
    "true_depth": "原始 4 通道",
    "zero_depth": "第 4 通道置 0",
    "spatial_shuffle_depth": "打乱深度像素位置",
    "flat_valid_depth": "有效区填均值",
    "blurred_depth": "深度高斯模糊 31",
}
BLUR_KERNEL: int = 31
SHUFFLE_SEED_BASE: int = 20261011


# ============================================================
# 变体算法（与 ablate_rgbd_depth.py 同实现，保证口径一致）
# ============================================================
def spatial_shuffle_depth_channel(ch4: np.ndarray, seed: int) -> np.ndarray:
    """仅对像素位置做固定 seed 的随机 permutation，保留全部像素值（统计量不变）。"""
    rng = np.random.default_rng(seed)
    flat = ch4.ravel()
    perm = rng.permutation(flat.size)
    return flat[perm].reshape(ch4.shape).astype(ch4.dtype)


def blurred_depth_channel(ch4: np.ndarray, kernel_size: int) -> np.ndarray:
    """masked Gaussian blur：只让有效像素参与平均，0=invalid 保持 0。"""
    kernel = (int(kernel_size), int(kernel_size))
    valid = (ch4 > 0).astype(np.float32)
    depth = ch4.astype(np.float32)
    weighted = cv2.GaussianBlur(depth * valid, kernel, 0)
    weights = cv2.GaussianBlur(valid, kernel, 0)
    blurred = np.divide(weighted, weights, out=np.zeros_like(weighted),
                        where=weights > 1e-6)
    blurred[ch4 == 0] = 0
    return np.clip(np.rint(blurred), 0, 255).astype(np.uint8)


def flat_valid_depth_channel(ch4: np.ndarray) -> np.ndarray:
    """有效区全部填成该图有效区均值；invalid 严格保持 0。"""
    valid_mask = ch4 > 0
    flat = np.zeros_like(ch4, dtype=np.uint8)
    if not np.any(valid_mask):
        return flat
    mean_value = int(np.clip(np.rint(float(ch4[valid_mask].mean())), 1, 255))
    flat[valid_mask] = mean_value
    return flat


def apply_variant(image: np.ndarray, variant: str, seed: int) -> np.ndarray:
    if variant == "true_depth":
        return image
    out = image.copy()
    ch4 = out[:, :, 3]
    if variant == "zero_depth":
        out[:, :, 3] = 0
    elif variant == "spatial_shuffle_depth":
        out[:, :, 3] = spatial_shuffle_depth_channel(ch4, seed)
    elif variant == "flat_valid_depth":
        out[:, :, 3] = flat_valid_depth_channel(ch4)
    elif variant == "blurred_depth":
        out[:, :, 3] = blurred_depth_channel(ch4, BLUR_KERNEL)
    else:
        raise ValueError(f"未知变体: {variant}")
    return out


# ============================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Depth 消融（NME 口径）")
    p.add_argument("--dataset-root", default=str(_DEFAULT_DATASET))
    p.add_argument("--weights-dir", default=str(_REPO / "train_weight"))
    p.add_argument("--weights", nargs="+", default=None)
    p.add_argument("--variants", nargs="+", default=list(ALL_VARIANTS),
                   choices=list(ALL_VARIANTS), help="要跑的变体，默认全部")
    p.add_argument("--match-iou", type=float, default=0.50)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--output-dir", default=str(_REPO / "runs" / "ablate_depth_nme"))
    return p.parse_args()


def stem_map(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        return {}
    return {
        p.stem: p
        for p in sorted(directory.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    }


def read_rgbd(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"图像读取失败: {path}")
    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError(f"图像不是 4 通道: {path} shape={image.shape}")
    return image


def mean_of(values: list[Any]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return float(np.mean(clean)) if clean else None


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root).expanduser()
    out_dir = Path(args.output_dir).expanduser()
    variants: list[str] = list(dict.fromkeys(args.variants))

    print("=" * 104)
    print("Depth 消融（NME 口径）：4 通道模型「用没用」与「怎么用」第 4 通道")
    print(f"测试集 : {root}")
    print(f"变体   : {', '.join(variants)}")
    print(f"口径   : IoU >= {args.match_iou}   conf={args.conf}   imgsz={args.imgsz}")
    print(f"实现   : {_EVAL_SCRIPT.name}（ultralytics {_LOCKED_VERSION}）")
    print("=" * 104)

    rgbd_map = stem_map(root / "rgbd" / "images" / "test")
    if not rgbd_map:
        print(f"[STOP] 找不到 4 通道测试图: {root / 'rgbd' / 'images' / 'test'}")
        return
    label_dir = root / "rgb" / "labels" / "test"
    if not label_dir.is_dir():
        label_dir = root / "rgbd" / "labels" / "test"
    stems = sorted(rgbd_map)
    print(f"测试图片: rgbd={len(stems)} 张   标签: {label_dir}")

    if args.weights:
        candidates = [Path(w).expanduser() for w in args.weights]
    else:
        weights_dir = Path(args.weights_dir).expanduser()
        candidates = sorted(
            p for p in weights_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".pt"
        ) if weights_dir.is_dir() else []

    models: list[dict[str, Any]] = []
    for weights in candidates:
        try:
            model = YOLO(str(weights))
        except Exception as exc:  # noqa: BLE001
            print(f"[跳过] {weights.name} 加载失败: {type(exc).__name__}: {exc}")
            continue
        channels = int(model.model.model[0].conv.weight.shape[1])
        if channels != 4:
            print(f"[跳过] {weights.name} 是 {channels} 通道（本消融只针对 4 通道）")
            continue
        models.append({"name": weights.stem, "path": weights, "model": model})
        print(f"已加载: {weights.stem:24s} 4ch")
    if not models:
        print("[STOP] 没有可用的 4 通道权重")
        return

    rows: list[dict[str, Any]] = []
    for item in models:
        for variant in variants:
            started = time.perf_counter()
            matched = 0
            for index, stem in enumerate(stems, start=1):
                label_path = label_dir / f"{stem}.txt"
                if not label_path.is_file():
                    continue
                image = read_rgbd(rgbd_map[stem])
                image = apply_variant(image, variant,
                                      SHUFFLE_SEED_BASE + index)
                height, width = image.shape[:2]
                gt = ev.load_pose_label(label_path, width, height)
                result = ev.run_pose_model(item["model"], image, args.conf, args.imgsz)
                pred = ev.find_best_prediction(result, gt["box"], args.match_iou)
                record: dict[str, Any] = {
                    "model": item["name"], "variant": variant, "image": stem,
                    "matched": bool(pred["matched"]),
                    "iou": round(float(pred["iou"]), 6), "nme": None,
                }
                for k in range(KEYPOINT_COUNT):
                    record[f"k{k}_px"] = None
                if pred["matched"]:
                    matched += 1
                    nme_result = ev.calculate_nme(gt["box"], gt["keypoints"],
                                                  pred["keypoints"])
                    record["nme"] = float(nme_result["nme"])
                    errors = ev.extract_keypoint_pixel_errors(nme_result, KEYPOINT_COUNT)
                    for k, value in enumerate(errors):
                        record[f"k{k}_px"] = (None if value is None
                                              else round(float(value), 4))
                rows.append(record)
                if index % 25 == 0 or index == len(stems):
                    print(f"  {item['name']:22s} {variant:22s} 进度 {index}/{len(stems)}",
                          flush=True)
            print(f"  -> {item['name']} / {variant}: 命中 {matched}/{len(stems)}  "
                  f"用时 {time.perf_counter() - started:.1f}s")

    summary: list[dict[str, Any]] = []
    for item in models:
        entry: dict[str, Any] = {"model": item["name"]}
        base_nme: float | None = None
        for variant in variants:
            subset = [r for r in rows
                      if r["model"] == item["name"] and r["variant"] == variant]
            nme = mean_of([r["nme"] for r in subset])
            entry[f"{variant}_matched"] = sum(1 for r in subset if r["matched"])
            entry[f"{variant}_nme"] = nme
            for k in range(KEYPOINT_COUNT):
                entry[f"{variant}_k{k}_px"] = mean_of([r[f"k{k}_px"] for r in subset])
            if variant == "true_depth":
                base_nme = nme
        for variant in variants:
            current = entry.get(f"{variant}_nme")
            if base_nme and current is not None:
                entry[f"{variant}_delta"] = current - base_nme
                entry[f"{variant}_pct"] = (current - base_nme) / base_nme * 100.0
            else:
                entry[f"{variant}_delta"] = None
                entry[f"{variant}_pct"] = None
        summary.append(entry)

    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["model"]
    for variant in variants:
        fields += [f"{variant}_matched", f"{variant}_nme"]
        fields += [f"{variant}_k{k}_px" for k in range(KEYPOINT_COUNT)]
        fields += [f"{variant}_delta", f"{variant}_pct"]
    with (out_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)

    per_fields = ["model", "variant", "image", "matched", "iou", "nme"] + \
        [f"k{k}_px" for k in range(KEYPOINT_COUNT)]
    with (out_dir / "per_image.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    (out_dir / "run_config.json").write_text(
        json.dumps({
            "dataset_root": str(root), "stems": len(stems),
            "variants": variants, "blur_kernel": BLUR_KERNEL,
            "shuffle_seed_base": SHUFFLE_SEED_BASE,
            "match_iou": args.match_iou, "conf": args.conf, "imgsz": args.imgsz,
            "weights": [{"name": m["name"], "path": str(m["path"])} for m in models],
            "ultralytics": _LOCKED_VERSION, "eval_impl": _EVAL_SCRIPT.name,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ---- 终端表 ----
    print("\n" + "=" * 104)
    print(f"NME 对比（测试集 = {root.name}/test，{len(stems)} 张）")
    print("=" * 104)
    head = f"{'模型':22s} " + " ".join(f"{v.replace('_depth', ''):>20s}" for v in variants)
    print(head)
    for entry in summary:
        cells: list[str] = []
        for variant in variants:
            nme = entry.get(f"{variant}_nme")
            pct = entry.get(f"{variant}_pct")
            cells.append(f"{'%.4f' % nme if nme is not None else '-':>12s}"
                         f"{'%+.1f%%' % pct if pct is not None else '':>8s}")
        print(f"{entry['model']:22s} " + " ".join(f"{c:>20s}" for c in cells))

    print("\n[判据]")
    total = len(stems)
    for entry in summary:
        print(f"  {entry['model']}")
        for variant in variants:
            if variant == "true_depth":
                continue
            matched = int(entry.get(f"{variant}_matched", 0) or 0)
            pct = entry.get(f"{variant}_pct")
            # 命中数过低不是"变差"，而是"彻底失效"，必须单独报出（否则会被当成无数据跳过）
            if matched * 2 <= total:
                print(f"      {variant:22s} 命中 {matched}/{total}"
                      f"   → 【检出失效】该输入下模型基本无法定位")
                continue
            if pct is None:
                continue
            if variant == "zero_depth":
                verdict = ("明显变差 → 在用深度" if pct > 5
                           else "基本不变 → 没用深度" if pct > -5 else "反而变好 → 深度是干扰")
            elif variant == "spatial_shuffle_depth":
                verdict = ("破坏空间结构后变差 → 依赖深度的【空间对应关系】" if pct > 5
                           else "基本不变 → 只把深度当【有没有】的开关，不用其空间结构")
            elif variant == "flat_valid_depth":
                verdict = ("填平后变差 → 依赖【连续几何/梯度】" if pct > 5
                           else "基本不变 → 只用【有效区轮廓】，不用内部几何")
            else:
                verdict = ("模糊后变差 → 依赖【局部高频细节】" if pct > 5
                           else "基本不变 → 只用【粗尺度结构】")
            print(f"      {variant:22s} {pct:+7.1f}%   {verdict}")

    print(f"\n结果文件:\n  {out_dir / 'summary.csv'}\n  {out_dir / 'per_image.csv'}")


if __name__ == "__main__":
    main()
