r"""YOLO11-RGBD 第 1 层通道归因：ch09 / ch13 / ch15 到底在干什么。

上一轮已经知道「单独削弱 ch09 或 ch15 会让 NME 变好，单独恢复 ch13 会让 NME 变好」，
但不知道原因。这个脚本分两段拿证据：

  阶段 A 观察（这个通道长什么样、跟谁对齐）
    - 逐通道激活与四种参照图的相关：输入深度通道 / RGB 灰度 / RGB 梯度 / GT 关键点热区
    - 把深度换成 zero / flat / blur 后，逐通道激活掉了多少 → 这个通道是不是深度喂出来的
    - 峰值位置离最近的 GT 关键点有多远 → 它的响应是不是落在关键点上

  阶段 B 干预（把「空间内容」和「整体幅度」拆开）
    - 空间置换：保留该通道的取值集合 / 均值 / 方差，只把 H×W 打乱
        置换后变好 → 有害的是它携带的空间内容
        置换后没变 → 有害的是它的整体幅度
    - 整通道压成空间均值：幅度留着、空间内容抹掉，作为对照
    - 安慰剂：同样置换一个非目标通道（默认 ch01），证明「置换动作本身」没影响

不重写评估：读图、命中判定、NME、深度变体全部复用现有实现
（ablate_c2psa / ablate_layers / ablate_l1_channel_groups / ablate_depth_nme）。

跑法（在 MultiYoloRGB 根目录）：
    .venv\Scripts\python.exe l1_channel_attribution.py
    .venv\Scripts\python.exe l1_channel_attribution.py --limit 10          # 冒烟
    .venv\Scripts\python.exe l1_channel_attribution.py --channels 9 13 15

输出目录默认 runs/l1_channel_attribution：
    channel_reference.csv     通道 × 参照 的相关
    depth_dependence.csv      深度变体下每个通道的激活衰减
    peak_alignment.csv        峰值到最近 GT 点的距离
    intervene_summary.csv     每个干预条件的命中 / NME / 逐点误差
    intervene_paired.csv      逐图配对（只算两边都命中的图）
    same_frames_summary.csv   全部条件共同命中集的 NME
    channel_profile.csv       通道号与分组对照
    run_config.json           全部参数与随机种子
    figures/                  三张图（matplotlib，缺了只跳过图的生成）
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn

import ablate_c2psa as ac
import ablate_depth_nme as ad
from ablate_l1_channel_groups import (
    channel_groups,
    check_l1_and_get_channels,
    fmt,
    hits_map,
    write_csv,
)
from ablate_layers import load_items

LAYER_INDEX = 1
KEYPOINTS = ac.KEYPOINT_COUNT
DEPTH_VARIANTS = ("true_depth", "zero_depth", "flat_valid_depth", "blurred_depth")
REFERENCES = ("depth", "gray", "grad", "gt_heat")
PLACEBO_CHANNEL_1BASED = 1
FIGURE_IMAGES = 6


# ============================================================
# 包装层：都套在第 1 层外面，保留 .f / .i 供 ultralytics 排拓扑顺序
# ============================================================
class L1Recorder(nn.Module):
    """照常出结果，顺手把 L1 输出抄一份存起来（C,H,W）。"""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.source = source
        self.f = source.f
        self.i = source.i
        self.type = "L1Recorder"
        self.last: np.ndarray | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.source(x)
        if isinstance(y, torch.Tensor) and y.ndim == 4:
            # 评估循环一次只喂一张图，[0] 取掉 batch 维；转 numpy 是为了后面算相关
            self.last = y.detach().float().cpu().numpy()[0]
        else:
            self.last = None
        return y


class L1ChannelSpatialShuffle(nn.Module):
    """把指定输出通道的 H×W 网格按同一个固定置换重排。

    取值集合、均值、方差全都不变，只破坏「哪个位置是什么」。
    """

    def __init__(self, source: nn.Module, channels: tuple[int, ...], seed: int) -> None:
        super().__init__()
        self.source = source
        self.f = source.f
        self.i = source.i
        self.type = "L1ChannelSpatialShuffle"
        self.channels = channels
        self.seed = seed
        self.perm: torch.Tensor | None = None
        self.grid: tuple[int, int] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.source(x)
        self._check(y)
        batch, chans, height, width = y.shape
        if self.perm is None or self.grid != (height, width):
            rng = np.random.default_rng(self.seed)
            self.perm = torch.as_tensor(rng.permutation(height * width), dtype=torch.long, device=y.device)
            self.grid = (height, width)
        out = y.clone()
        for channel in self.channels:
            flat = out[:, channel].reshape(batch, -1)
            out[:, channel] = flat[:, self.perm].reshape(batch, height, width)
        return out

    def _check(self, y: Any) -> None:
        if not isinstance(y, torch.Tensor) or y.ndim != 4:
            raise RuntimeError(f"L1 输出不是 4 维张量: {getattr(y, 'shape', None)}")
        if max(self.channels) >= y.shape[1]:
            raise ValueError(f"通道 {max(self.channels)} 超出 L1 输出通道数 {y.shape[1]}")


class L1ChannelFlatten(nn.Module):
    """把指定通道压成它自己的空间均值：幅度留着，空间内容抹掉。"""

    def __init__(self, source: nn.Module, channels: tuple[int, ...]) -> None:
        super().__init__()
        self.source = source
        self.f = source.f
        self.i = source.i
        self.type = "L1ChannelFlatten"
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.source(x)
        if not isinstance(y, torch.Tensor) or y.ndim != 4:
            raise RuntimeError(f"L1 输出不是 4 维张量: {getattr(y, 'shape', None)}")
        if max(self.channels) >= y.shape[1]:
            raise ValueError(f"通道 {max(self.channels)} 超出 L1 输出通道数 {y.shape[1]}")
        out = y.clone()
        for channel in self.channels:
            mean = out[:, channel].mean(dim=(1, 2), keepdim=True)
            out[:, channel] = mean.expand_as(out[:, channel])
        return out


@dataclass(frozen=True)
class Condition:
    name: str
    kind: str                    # baseline / shuffle / flatten
    channels: tuple[int, ...]    # 0-based


# ============================================================
# 参数
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO11-RGBD 第 1 层通道归因与空间置换干预")
    parser.add_argument("--dataset-root", default=str(ac._DEFAULT_DATASET))
    parser.add_argument("--weights-dir", default=str(ac._REPO / "train_weight"))
    parser.add_argument("--weights-name", default="yolo11n_rgbd_4ch")
    parser.add_argument("--split", default="test")
    parser.add_argument("--channels", nargs="+", type=int, default=[9, 13, 15],
                        help="要解释的通道号，1-based，默认 ch09 ch13 ch15")
    parser.add_argument("--placebo-channel", type=int, default=PLACEBO_CHANNEL_1BASED,
                        help="安慰剂通道（1-based），用来证明置换动作本身没影响")
    parser.add_argument("--depth-variants", nargs="+", default=list(DEPTH_VARIANTS),
                        choices=list(DEPTH_VARIANTS))
    parser.add_argument("--skip-intervene", action="store_true", help="只跑观察阶段")
    parser.add_argument("--skip-observe", action="store_true", help="只跑干预阶段")
    parser.add_argument("--expected-count", type=int, default=100, help="0 表示不校验张数")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 张，0 表示全部（冒烟用）")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--figures", type=int, default=1, help="1 生成图片，0 跳过")
    parser.add_argument("--output-dir", default=str(ac._REPO / "runs" / "l1_channel_attribution"))
    return parser.parse_args()


# ============================================================
# 数据准备
# ============================================================
def collect_stems(root: Path, split: str, expected: int) -> list[str]:
    """取 rgb / rgbd / labels 三者都对齐的 stem 列表。"""
    rgb = ac.stem_map(root / "rgb" / "images" / split)
    rgbd = ac.stem_map(root / "rgbd" / "images" / split)
    label_dir = root / "rgb" / "labels" / split
    if not label_dir.is_dir():
        label_dir = root / "rgbd" / "labels" / split
    if not label_dir.is_dir():
        raise FileNotFoundError(label_dir)
    stems = sorted(s for s in rgb.keys() & rgbd.keys() if (label_dir / f"{s}.txt").is_file())
    if not stems:
        raise RuntimeError(f"没有找到对齐的图: {root} / {split}")
    if expected and len(stems) != expected:
        raise RuntimeError(f"期望 {expected} 张对齐图，实际 {len(stems)} 张")
    return stems


# ============================================================
# 参照图（都缩放到 L1 输出的空间尺度再算相关）
# ============================================================
def resize_map(source: np.ndarray, height: int, width: int) -> np.ndarray:
    return cv2.resize(source.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA)


def reference_maps(image: np.ndarray, gt: dict[str, Any],
                   height: int, width: int) -> dict[str, np.ndarray]:
    """四种参照：深度通道、灰度、梯度幅值、GT 关键点热区。"""
    image_h, image_w = image.shape[:2]
    rgb = image[:, :, :3]
    gray_full = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    grad_full = cv2.magnitude(cv2.Sobel(gray_full, cv2.CV_32F, 1, 0, ksize=3),
                              cv2.Sobel(gray_full, cv2.CV_32F, 0, 1, ksize=3))
    box = np.asarray(gt["box"], dtype=np.float64)
    diagonal = float(np.hypot(box[2] - box[0], box[3] - box[1])) or 1.0
    return {
        "depth": resize_map(image[:, :, 3], height, width),
        "gray": resize_map(gray_full, height, width),
        "grad": resize_map(grad_full, height, width),
        "gt_heat": gaussian_heat(gt["keypoints"], image_w, image_h, width, height, diagonal),
    }


def gaussian_heat(keypoints: np.ndarray, image_w: int, image_h: int,
                  width: int, height: int, diagonal: float) -> np.ndarray:
    """把 7 个 GT 点摊成一张高斯热区，只在特征尺度上算，省一次缩放。"""
    heat = np.zeros((height, width), dtype=np.float32)
    sigma_img = max(diagonal * 0.05, 1.0)
    sigma = sigma_img * width / float(image_w)
    if sigma <= 0:
        return heat
    grid_y, grid_x = np.mgrid[0:height, 0:width].astype(np.float32)
    for keypoint in keypoints:
        if keypoint[2] <= 0:
            continue
        fx = float(keypoint[0]) * width / float(image_w)
        fy = float(keypoint[1]) * height / float(image_h)
        heat += np.exp(-(((grid_x - fx) ** 2 + (grid_y - fy) ** 2) / (2.0 * sigma ** 2)))
    return heat


def rankdata(values: np.ndarray) -> np.ndarray:
    """平均秩次（并列取平均）；只为了手算 Spearman，避免再引一个依赖。

    用 np.unique 一次算出每个取值的起始秩和重复次数，再散射回去——
    比逐点循环快两个数量级，32 个通道 × 100 张才跑得动。
    """
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    starts = np.concatenate(([0.0], np.cumsum(counts)[:-1].astype(np.float64)))
    return (starts + (counts.astype(np.float64) - 1.0) / 2.0)[inverse.ravel()]


def correlations(activation: np.ndarray, reference: np.ndarray) -> tuple[float | None, float | None]:
    """返回 (pearson, spearman)；任一侧没有变化就返回 None。"""
    a = activation.astype(np.float64).ravel()
    b = reference.astype(np.float64).ravel()
    if a.size < 3 or a.std() < 1e-9 or b.std() < 1e-9:
        return None, None
    pearson = float(np.corrcoef(a, b)[0, 1])
    spearman = float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])
    return pearson, spearman


def resolve_active_layer(model: Any) -> Any | None:
    """找到推理里真正复用的那个 L1Recorder。

    predictor 建模型时会 deepcopy 一份（predictor.py:466 `model = deepcopy(model)`），
    所以挂在我们手上的 recorder 和真正参与前向的是两个对象。这里从 predictor 侧
    往下扫，按 isinstance 认领——不写死 `predictor.model.model` 这种路径，
    免得 ultralytics 换一层包装就失效。
    """
    candidates: list[Any] = []
    predictor = getattr(model, "predictor", None)
    backend = getattr(predictor, "model", None) if predictor is not None else None
    for holder in (backend, getattr(backend, "model", None) if backend is not None else None):
        if holder is not None and hasattr(holder, "modules"):
            candidates.extend(module for module in holder.modules()
                              if isinstance(module, L1Recorder))
    candidates.extend(module for module in model.model.modules()
                      if isinstance(module, L1Recorder))
    for module in candidates:
        if module.last is not None:
            return module
    return None


# ============================================================
# 单张图评估（与 ac.run_condition 同口径，额外把激活读出来）
# ============================================================
def evaluate_one(model: Any, image: np.ndarray, gt: dict[str, Any],
                 conf: float, imgsz: int, match_iou: float,
                 want_activation: bool = False) -> tuple[dict[str, Any], np.ndarray | None]:
    result = ac.ev.run_pose_model(model, image, conf, imgsz)
    activation = None
    if want_activation:
        layer = resolve_active_layer(model)
        activation = None if layer is None else layer.last
    n_boxes = 0 if result.boxes is None else len(result.boxes)
    prediction = ac.ev.find_best_prediction(result, gt["box"], match_iou)
    nme: float | None = None
    pixel_errors: list[float | None] = [None] * KEYPOINTS
    if prediction["matched"]:
        try:
            nme_result = ac.ev.calculate_nme(gt["box"], gt["keypoints"], prediction["keypoints"])
            nme = float(nme_result["nme"])
            errors = ac.ev.extract_keypoint_pixel_errors(nme_result, KEYPOINTS)
            pixel_errors = [None if v is None else float(v) for v in errors]
        except ValueError:
            nme = None
    row = {
        "n_boxes": n_boxes,
        "matched": bool(prediction["matched"]),
        "iou": round(float(prediction["iou"]), 6),
        "nme": None if nme is None else round(nme, 6),
        **{f"k{k}_px": None if pixel_errors[k] is None else round(pixel_errors[k], 4)
           for k in range(KEYPOINTS)},
    }
    return row, activation


# ============================================================
# 阶段 A：观察
# ============================================================
def observe(weight: Path, items: list[tuple[str, np.ndarray, dict[str, Any]]],
            target_channels: tuple[int, ...], variants: list[str],
            args: argparse.Namespace) -> dict[str, Any]:
    """逐个深度变体跑一遍，收集通道激活的三种证据。"""
    stats: dict[str, Any] = {
        "activation": {},   # variant -> channel(1based) -> [abs_mean...]
        "correlation": {},  # ref -> channel(1based) -> {"pearson": [], "spearman": []}
        "peak": {},         # channel(1based) -> [dist_norm...]
        "nme": {},          # variant -> list[float]
        "hits": {},         # variant -> dict[stem -> row]
        "grid": None,
    }
    for ref in REFERENCES:
        stats["correlation"][ref] = {ch: {"pearson": [], "spearman": []} for ch in target_channels}

    for variant in variants:
        model = ac.YOLO(str(weight))
        model.model.model[LAYER_INDEX] = L1Recorder(model.model.model[LAYER_INDEX])
        started = time.perf_counter()
        variant_nme: list[float] = []
        hit_rows: dict[str, dict[str, Any]] = {}
        try:
            for stem, image, gt in items:
                variant_image = ad.apply_variant(image, variant, args.seed)
                row, activation = evaluate_one(model, variant_image, gt, args.conf,
                                               args.imgsz, args.match_iou,
                                               want_activation=True)
                hit_rows[stem] = {"image": stem, **row}
                if row["nme"] is not None:
                    variant_nme.append(row["nme"])
                if activation is None:
                    raise RuntimeError(
                        f"{variant}/{stem}: 没拿到 L1 激活。多半是包装层没跟着 predictor 的副本走，"
                        "检查 resolve_active_layer 是否认到了 L1Recorder")
                height, width = activation.shape[1], activation.shape[2]
                stats["grid"] = (height, width)
                abs_mean = np.abs(activation.astype(np.float32)).mean(axis=(1, 2))
                for channel in target_channels:
                    stats["activation"].setdefault(variant, {}).setdefault(channel, []).append(
                        float(abs_mean[channel - 1]))
                if variant != "true_depth":
                    continue
                # 参照相关与峰值位置只在真实深度上算（变体阶段只关心激活强度怎么变）
                maps = reference_maps(variant_image, gt, height, width)
                for channel in target_channels:
                    plane = activation[channel - 1].astype(np.float32)
                    for ref in REFERENCES:
                        pearson, spearman = correlations(plane, maps[ref])
                        if pearson is None:
                            continue
                        stats["correlation"][ref][channel]["pearson"].append(pearson)
                        stats["correlation"][ref][channel]["spearman"].append(spearman)
                    stats["peak"].setdefault(channel, []).append(
                        peak_distance(plane, gt["keypoints"],
                                      image.shape[0], image.shape[1], args.imgsz))
        finally:
            del model
            gc.collect()
        stats["nme"][variant] = variant_nme
        stats["hits"][variant] = hit_rows
        print(f"[observe] {variant:<19} hits {len(variant_nme):>3}/{len(items)} "
              f"NME={fmt(ac.mean_of(variant_nme))}  {time.perf_counter() - started:.1f}s")
    return stats


def letterbox_transform(image_h: int, image_w: int, imgsz: int) -> tuple[float, float, float]:
    """照着 ultralytics 的 letterbox 复算一次缩放与左右/上下补边，返回 (缩放倍率, 横向补边, 纵向补边)。

    特征图和原图之间不是简单的宽高比关系——中间隔着一层 640 等边画布和居中的灰边，
    直接把特征坐标乘回去在非正方形图上会偏，所以这里显式算一遍。
    """
    ratio = min(imgsz / float(image_h), imgsz / float(image_w))
    pad_x = (imgsz - image_w * ratio) / 2.0
    pad_y = (imgsz - image_h * ratio) / 2.0
    return ratio, pad_x, pad_y


def peak_distance(plane: np.ndarray, keypoints: np.ndarray,
                  image_h: int, image_w: int, imgsz: int) -> float | None:
    """激活峰值到最近 GT 关键点的距离（原图像素）。

    先把峰值从特征图坐标还原到 letterbox 画布，再把 GT 点搬到同一画布上比，
    最后除以缩放倍率换回原图像素——等比缩放，所以距离可以直接除。
    """
    visible = keypoints[keypoints[:, 2] > 0]
    if visible.size == 0:
        return None
    feature_h, feature_w = plane.shape
    stride = imgsz / float(feature_w)
    ratio, pad_x, pad_y = letterbox_transform(image_h, image_w, imgsz)
    peak_row, peak_col = np.unravel_index(int(np.argmax(plane)), plane.shape)
    peak_x = peak_col * stride
    peak_y = peak_row * stride
    gt_x = visible[:, 0] * ratio + pad_x
    gt_y = visible[:, 1] * ratio + pad_y
    distances = np.hypot(gt_x - peak_x, gt_y - peak_y)
    return float(distances.min() / ratio)


# ============================================================
# 阶段 B：干预
# ============================================================
def build_conditions(target_channels: tuple[int, ...],
                     placebo: int) -> list[Condition]:
    """基线 + 安慰剂 + 每个目标通道的置换/压平 + 三个一起置换。"""
    conditions = [Condition("baseline", "baseline", ())]
    conditions.append(Condition(f"shuffle_ch{placebo:02d}", "shuffle", (placebo - 1,)))
    for channel in target_channels:
        conditions.append(Condition(f"shuffle_ch{channel:02d}", "shuffle", (channel - 1,)))
    for channel in target_channels:
        conditions.append(Condition(f"flatten_ch{channel:02d}", "flatten", (channel - 1,)))
    conditions.append(Condition(
        "shuffle_all_" + "_".join(f"{c:02d}" for c in target_channels),
        "shuffle",
        tuple(c - 1 for c in target_channels),
    ))
    return conditions


def wrap_layer(model: Any, condition: Condition, channels: int, seed: int) -> None:
    """按条件把第 1 层换掉；必须在 YOLO 首次 predict 之前调用。"""
    if condition.kind == "baseline":
        return
    if condition.channels and max(condition.channels) >= channels:
        raise ValueError(f"通道 {max(condition.channels)} 超出 L1 输出通道数 {channels}")
    original = model.model.model[LAYER_INDEX]
    if condition.kind == "shuffle":
        model.model.model[LAYER_INDEX] = L1ChannelSpatialShuffle(original, condition.channels, seed)
        return
    if condition.kind == "flatten":
        model.model.model[LAYER_INDEX] = L1ChannelFlatten(original, condition.channels)
        return
    raise ValueError(f"未知干预类型: {condition.kind}")


def intervene(weight: Path, items: list[tuple[str, np.ndarray, dict[str, Any]]],
              conditions: list[Condition], channels: int,
              args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    """每个条件都重新加载权重再套包装层（predictor 会 deepcopy，晚改无效）。"""
    rows_by_condition: dict[str, list[dict[str, Any]]] = {}
    for condition in conditions:
        model = ac.YOLO(str(weight))
        wrap_layer(model, condition, channels, args.seed)
        started = time.perf_counter()
        rows: list[dict[str, Any]] = []
        try:
            for stem, image, gt in items:
                row, _ = evaluate_one(model, image, gt, args.conf, args.imgsz, args.match_iou)
                rows.append({"image": stem, **row})
        finally:
            del model
            gc.collect()
        rows_by_condition[condition.name] = rows
        hits = len(hits_map(rows))
        print(f"[intervene] {condition.name:<22} hits {hits:>3}/{len(rows)} "
              f"NME={fmt(ac.mean_of([r['nme'] for r in rows]))}  {time.perf_counter() - started:.1f}s")
    return rows_by_condition


# ============================================================
# 汇总
# ============================================================
def summarise_intervention(rows_by_condition: dict[str, list[dict[str, Any]]],
                           conditions: list[Condition]) -> tuple[list[dict[str, Any]],
                                                                  list[dict[str, Any]]]:
    """条件级汇总 + 逐图配对（只算与基线都命中的图）。"""
    baseline = {r["image"]: r for r in rows_by_condition["baseline"]}
    summary: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    for condition in conditions:
        rows = rows_by_condition[condition.name]
        hits = hits_map(rows)
        pairs = [stem for stem in hits if stem in baseline and baseline[stem]["nme"] is not None]
        deltas = [hits[stem]["nme"] - baseline[stem]["nme"] for stem in pairs]
        entry: dict[str, Any] = {
            "condition": condition.name,
            "kind": condition.kind,
            "channels_1based": ",".join(str(c + 1) for c in condition.channels),
            "frames": len(rows),
            "hits": len(hits),
            "nme_matched_only": ac.mean_of([r["nme"] for r in rows]),
            "paired_frames": len(pairs),
            "paired_baseline_nme": ac.mean_of([baseline[s]["nme"] for s in pairs]),
            "paired_case_nme": ac.mean_of([hits[s]["nme"] for s in pairs]),
            "paired_delta_nme": ac.mean_of(deltas),
            "paired_win_images": sum(1 for d in deltas if d < 0),
            "paired_lose_images": sum(1 for d in deltas if d > 0),
        }
        if entry["paired_baseline_nme"]:
            entry["paired_delta_pct"] = (entry["paired_delta_nme"] /
                                         entry["paired_baseline_nme"] * 100.0)
        else:
            entry["paired_delta_pct"] = None
        for k in range(KEYPOINTS):
            base_values = [baseline[s][f"k{k}_px"] for s in pairs]
            case_values = [hits[s][f"k{k}_px"] for s in pairs]
            entry[f"k{k}_baseline_px"] = ac.mean_of(base_values)
            entry[f"k{k}_case_px"] = ac.mean_of(case_values)
            valid = [(c, b) for c, b in zip(case_values, base_values) if c is not None and b is not None]
            entry[f"k{k}_delta_px"] = ac.mean_of([c - b for c, b in valid])
        summary.append(entry)
        for stem in sorted(pairs):
            paired.append({
                "condition": condition.name,
                "image": stem,
                "baseline_nme": baseline[stem]["nme"],
                "case_nme": hits[stem]["nme"],
                "delta_nme": hits[stem]["nme"] - baseline[stem]["nme"],
            })
    return summary, paired


def same_frame_summary(rows_by_condition: dict[str, list[dict[str, Any]]],
                       conditions: list[Condition]) -> tuple[list[dict[str, Any]], int]:
    """所有条件都命中的交集上重算 NME，避免命中集不同导致不可比。"""
    maps = {name: hits_map(rows) for name, rows in rows_by_condition.items()}
    common = sorted(set.intersection(*(set(m) for m in maps.values())))
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        subset = [maps[condition.name][stem] for stem in common]
        entry: dict[str, Any] = {
            "condition": condition.name,
            "common_frames": len(common),
            "nme_common": ac.mean_of([r["nme"] for r in subset]),
        }
        for k in range(KEYPOINTS):
            entry[f"k{k}_px_common"] = ac.mean_of([r[f"k{k}_px"] for r in subset])
        rows.append(entry)
    return rows, len(common)


# ============================================================
# 图片（matplotlib 缺失时只告警，不影响 CSV）
# ============================================================
def plot_figures(figure_dir: Path, channel_rows: list[dict[str, Any]],
                 depth_rows: list[dict[str, Any]],
                 summary: list[dict[str, Any]],
                 variants: list[str]) -> list[str]:
    """三张图：通道级深度依赖、通道×参照相关热图、干预前后 NME 对比。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        print("[figures] 没有 matplotlib，跳过出图")
        return []
    figure_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    channels = [row["channel_1based"] for row in channel_rows]
    labels = [f"ch{channel:02d}" for channel in channels]
    by_variant: dict[str, list[float]] = {}
    for variant in variants:
        by_variant[variant] = [
            next((row["abs_mean"] for row in depth_rows
                  if row["variant"] == variant and row["channel_1based"] == channel), 0.0)
            for channel in channels
        ]
    figure, axis = plt.subplots(figsize=(12, 3.6))
    width = 0.2
    positions = np.arange(len(by_variant["true_depth"]))
    for offset, (variant, values) in enumerate(by_variant.items()):
        axis.bar(positions + offset * width, values, width, label=variant)
    axis.set_xticks(positions + width * 1.5)
    axis.set_xticklabels(labels, rotation=90, fontsize=7)
    axis.set_ylabel("mean |activation|")
    axis.set_title("Depth dependence of L1 output channels")
    axis.legend(fontsize=7)
    path = figure_dir / "channel_depth_dependence.png"
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    written.append(path.name)

    matrix = np.full((len(channels), len(REFERENCES)), np.nan)
    for row_index, channel in enumerate(channels):
        for column, ref in enumerate(REFERENCES):
            value = next((r[f"{ref}_spearman_mean"] for r in channel_rows
                          if r["channel_1based"] == channel), None)
            if value is not None:
                matrix[row_index, column] = value
    figure, axis = plt.subplots(figsize=(4.6, 8.0))
    image = axis.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axis.set_xticks(range(len(REFERENCES)))
    axis.set_xticklabels(REFERENCES, rotation=45, ha="right", fontsize=8)
    axis.set_yticks(range(len(channels)))
    axis.set_yticklabels(labels, fontsize=7)
    axis.set_title("Spearman vs reference maps", fontsize=9)
    figure.colorbar(image, ax=axis, fraction=0.05)
    path = figure_dir / "channel_reference_heat.png"
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    written.append(path.name)

    names = [row["condition"] for row in summary]
    values = [row["nme_matched_only"] if row["nme_matched_only"] is not None else np.nan
              for row in summary]
    figure, axis = plt.subplots(figsize=(9, 3.2))
    colors = ["#888888" if n == "baseline" else "#4a7ebb" if n.startswith("shuffle") else "#c25b4a"
              for n in names]
    axis.bar(range(len(names)), values, color=colors)
    axis.set_xticks(range(len(names)))
    axis.set_xticklabels(names, rotation=35, ha="right", fontsize=7)
    axis.set_ylabel("NME (matched only)")
    axis.set_title("Intervention effect on L1 channels")
    path = figure_dir / "intervention_nme.png"
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    written.append(path.name)
    return written


def save_activation_overlays(figure_dir: Path,
                             items: list[tuple[str, np.ndarray, dict[str, Any]]],
                             args: argparse.Namespace) -> list[str]:
    """挑前几张图，把目标通道的激活叠在 RGB 上看落点。"""
    if not args.figures:
        return []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        return []
    targets = args.channels
    figure_dir.mkdir(parents=True, exist_ok=True)
    weight = Path(args.weights_dir) / (args.weights_name.removesuffix(".pt") + ".pt")
    model = ac.YOLO(str(weight))
    model.model.model[LAYER_INDEX] = L1Recorder(model.model.model[LAYER_INDEX])
    written: list[str] = []
    try:
        for stem, image, gt in items[:FIGURE_IMAGES]:
            _, activation = evaluate_one(model, image, gt, args.conf, args.imgsz,
                                          args.match_iou, want_activation=True)
            if activation is None:
                continue
            rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
            figure, axes = plt.subplots(1, len(targets) + 1, figsize=(3.4 * (len(targets) + 1), 3.0))
            axes[0].imshow(rgb)
            boxes = np.asarray(gt["box"])
            axes[0].add_patch(Rectangle(
                (boxes[0], boxes[1]), boxes[2] - boxes[0], boxes[3] - boxes[1],
                fill=False, edgecolor="lime", linewidth=1.2))
            for keypoint in gt["keypoints"]:
                if keypoint[2] > 0:
                    axes[0].plot(keypoint[0], keypoint[1], "o", color="yellow", markersize=3)
            axes[0].set_title(f"{stem[-17:]}", fontsize=8)
            axes[0].axis("off")
            for index, channel in enumerate(targets, start=1):
                plane = activation[channel - 1]
                plane = cv2.resize(plane, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
                # 按 99 分位归一化：少数极强响应不至于把整张图压成一片黑
                scale = float(np.percentile(np.abs(plane), 99)) or 1.0
                normalized = np.clip(np.abs(plane) / scale, 0, 1)
                heat_map = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
                blended = cv2.addWeighted(rgb, 0.55, cv2.cvtColor(heat_map, cv2.COLOR_BGR2RGB), 0.45, 0)
                axes[index].imshow(blended)
                for keypoint in gt["keypoints"]:
                    if keypoint[2] > 0:
                        axes[index].plot(keypoint[0], keypoint[1], "+", color="white", markersize=4)
                axes[index].set_title(f"ch{channel:02d}  p99={scale:.3f}", fontsize=8)
                axes[index].axis("off")
            path = figure_dir / f"overlay_{stem[-17:]}.png"
            figure.tight_layout()
            figure.savefig(path, dpi=140)
            plt.close(figure)
            written.append(path.name)
    finally:
        del model
        gc.collect()
    return written


def verdict_lines(summary: list[dict[str, Any]], depth_rows: list[dict[str, Any]],
                  target_channels: tuple[int, ...]) -> list[str]:
    """把「置换变好 = 空间内容有害」这条判据翻成人话。"""
    lines: list[str] = []
    by_name = {row["condition"]: row for row in summary}
    for name in sorted(by_name):
        row = by_name[name]
        if row["kind"] == "baseline" or row["paired_delta_pct"] is None:
            continue
        delta = row["paired_delta_pct"]
        if name.startswith("shuffle_ch") or name.startswith("shuffle_all"):
            if delta <= -2.0:
                lines.append(f"{name}: NME {delta:+.2f}% → 这个通道的空间内容在拖后腿")
            elif delta >= 2.0:
                lines.append(f"{name}: NME {delta:+.2f}% → 这个通道的空间内容有用，打乱就变差")
            else:
                lines.append(f"{name}: NME {delta:+.2f}% → 空间内容基本无关")
        else:
            lines.append(f"{name}: NME {delta:+.2f}% → 抹掉空间只剩幅度时的净效果")
    for row in depth_rows:
        if row["variant"] == "true_depth":
            continue
        change = row["rel_change_vs_true"]
        if change is None:
            continue
        if row["channel_1based"] in target_channels and abs(change) >= 0.20:
            lines.append(f"ch{row['channel_1based']:02d} 在 {row['variant']} 下激活 "
                         f"{change * 100:+.0f}% → 强依赖深度")
    return lines


# ============================================================
def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root).expanduser()
    out_dir = Path(args.output_dir).expanduser()
    weights_dir = Path(args.weights_dir).expanduser()
    weight = weights_dir / (args.weights_name.removesuffix(".pt") + ".pt")
    if not weight.is_file():
        raise FileNotFoundError(weight)
    target_channels = tuple(args.channels)
    if any(ch <= 0 for ch in target_channels):
        raise ValueError("--channels 用 1-based 编号，必须 > 0")

    stems = collect_stems(root, args.split, args.expected_count)
    if args.limit:
        stems = stems[:args.limit]
    items = load_items(root, args.split, 4, stems)

    probe = ac.YOLO(str(weight))
    channels = check_l1_and_get_channels(probe)
    del probe
    gc.collect()
    if max(target_channels) > channels:
        raise ValueError(f"通道 {max(target_channels)} 超出 L1 输出通道数 {channels}")
    groups = channel_groups(channels, 4)

    print("=" * 108)
    print("YOLO11-RGBD L1 CHANNEL ATTRIBUTION")
    print(f"weights: {weight}")
    print(f"images: {len(items)}  conf={args.conf} imgsz={args.imgsz} match_iou={args.match_iou}")
    print(f"L1 channels={channels}  groups={groups}  targets={target_channels}  seed={args.seed}")
    print("=" * 108)

    stats = None
    if not args.skip_observe:
        stats = observe(weight, items, target_channels, args.depth_variants, args)

    rows_by_condition: dict[str, list[dict[str, Any]]] = {}
    conditions: list[Condition] = []
    summary: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    same_frames: list[dict[str, Any]] = []
    common_frames = 0
    if not args.skip_intervene:
        conditions = build_conditions(target_channels, args.placebo_channel)
        rows_by_condition = intervene(weight, items, conditions, channels, args)
        summary, paired = summarise_intervention(rows_by_condition, conditions)
        same_frames, common_frames = same_frame_summary(rows_by_condition, conditions)

    out_dir.mkdir(parents=True, exist_ok=True)

    channel_rows: list[dict[str, Any]] = []
    for channel in target_channels:
        entry: dict[str, Any] = {
            "channel_1based": channel,
            "channel_0based": channel - 1,
            "group": next((f"G{i}" for i, (lo, hi) in enumerate(groups, 1)
                           if lo <= channel - 1 < hi), ""),
        }
        for ref in REFERENCES:
            entry[f"{ref}_pearson_mean"] = None
            entry[f"{ref}_spearman_mean"] = None
            entry[f"{ref}_spearman_median"] = None
            if stats is None:
                continue
            values = stats["correlation"][ref][channel]
            entry[f"{ref}_pearson_mean"] = ac.mean_of(values["pearson"])
            entry[f"{ref}_spearman_mean"] = ac.mean_of(values["spearman"])
            entry[f"{ref}_spearman_median"] = (
                None if not values["spearman"] else float(np.median(values["spearman"])))
        channel_rows.append(entry)

    depth_rows: list[dict[str, Any]] = []
    if stats is not None:
        for variant in args.depth_variants:
            for channel in target_channels:
                true_values = stats["activation"].get("true_depth", {}).get(channel, [])
                values = stats["activation"].get(variant, {}).get(channel, [])
                true_mean = ac.mean_of(true_values)
                variant_mean = ac.mean_of(values)
                change = None
                if true_mean and variant_mean is not None:
                    change = (variant_mean - true_mean) / true_mean
                depth_rows.append({
                    "variant": variant,
                    "channel_1based": channel,
                    "abs_mean": variant_mean,
                    "abs_mean_true": true_mean,
                    "rel_change_vs_true": change,
                })

    peak_rows: list[dict[str, Any]] = []
    if stats is not None:
        for channel in target_channels:
            distances = stats["peak"].get(channel, [])
            peak_rows.append({
                "channel_1based": channel,
                "group": next((f"G{i}" for i, (lo, hi) in enumerate(groups, 1)
                               if lo <= channel - 1 < hi), ""),
                "frames": len(distances),
                "mean_dist_px": ac.mean_of(distances),
                "median_dist_px": None if not distances else float(np.median(distances)),
            })

    channel_profile = []
    for group_index, (lo, hi) in enumerate(groups, 1):
        for channel in range(lo, hi):
            channel_profile.append({
                "channel_0based": channel,
                "channel_1based": channel + 1,
                "group": f"G{group_index}",
                "is_target": (channel + 1) in target_channels,
            })

    if channel_rows:
        write_csv(out_dir / "channel_reference.csv", channel_rows, list(channel_rows[0]))
    if depth_rows:
        write_csv(out_dir / "depth_dependence.csv", depth_rows, list(depth_rows[0]))
    if peak_rows:
        write_csv(out_dir / "peak_alignment.csv", peak_rows, list(peak_rows[0]))
    write_csv(out_dir / "channel_profile.csv", channel_profile, list(channel_profile[0]))
    if summary:
        fields = ["condition", "kind", "channels_1based", "frames", "hits", "nme_matched_only",
                  "paired_frames", "paired_baseline_nme", "paired_case_nme", "paired_delta_nme",
                  "paired_delta_pct", "paired_win_images", "paired_lose_images"]
        for k in range(KEYPOINTS):
            fields += [f"k{k}_baseline_px", f"k{k}_case_px", f"k{k}_delta_px"]
        write_csv(out_dir / "intervene_summary.csv", summary, fields)
    if paired:
        write_csv(out_dir / "intervene_paired.csv", paired, list(paired[0]))
    if same_frames:
        write_csv(out_dir / "same_frames_summary.csv", same_frames, list(same_frames[0]))

    figures: list[str] = []
    if args.figures and channel_rows and depth_rows and summary:
        figures = plot_figures(out_dir / "figures", channel_rows, depth_rows,
                               summary, args.depth_variants)
    if args.figures and stats is not None:
        overlays = save_activation_overlays(out_dir / "figures", items, args)
        figures.extend(overlays)

    config = {
        "dataset_root": str(root),
        "weights": str(weight),
        "split": args.split,
        "stems": len(items),
        "ultralytics": ac._LOCKED_VERSION,
        "eval_impl": ac._EVAL_SCRIPT.name,
        "layer_index": LAYER_INDEX,
        "l1_channels": channels,
        "groups": [{"name": f"G{i}", "start_inclusive": lo, "stop_exclusive": hi}
                   for i, (lo, hi) in enumerate(groups, 1)],
        "target_channels_1based": list(target_channels),
        "placebo_channel_1based": args.placebo_channel,
        "depth_variants": args.depth_variants,
        "seed": args.seed,
        "conf": args.conf,
        "imgsz": args.imgsz,
        "match_iou": args.match_iou,
        "conditions": [{"name": c.name, "kind": c.kind,
                        "channels_0based": list(c.channels)} for c in conditions],
        "common_frames_all_conditions": common_frames,
        "figures": figures,
        "note": "激活与干预都不改权重；每个条件重新加载 .pt 后再套包装层（predictor 会 deepcopy）。"
                "空间置换对每张图共用同一个固定排列，保证各条件只差「哪个通道被打乱」。",
    }
    (out_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 108)
    print("通道归因（与参照图的 Spearman 均值 / 峰值到最近 GT 点的距离）")
    print(f"{'channel':<9}{'group':<7}{'depth':>9}{'gray':>9}{'grad':>9}{'gt_heat':>9}{'peak_px':>10}")
    for row in channel_rows:
        peak = next((p["mean_dist_px"] for p in peak_rows
                     if p["channel_1based"] == row["channel_1based"]), None)
        print(f"ch{row['channel_1based']:02d}     {row['group']:<7}"
              f"{fmt(row['depth_spearman_mean'],3):>9}{fmt(row['gray_spearman_mean'],3):>9}"
              f"{fmt(row['grad_spearman_mean'],3):>9}{fmt(row['gt_heat_spearman_mean'],3):>9}"
              f"{fmt(peak,2):>10}")
    if summary:
        print("\n干预（都对比同一条基线，只算两边都命中的图）")
        print(f"{'condition':<24}{'hits':>5}{'pairs':>7}{'NME':>9}{'delta%':>9}{'wins':>6}{'loses':>7}")
        for row in summary:
            print(f"{row['condition']:<24}{row['hits']:>5}{row['paired_frames']:>7}"
                  f"{fmt(row['paired_case_nme']):>9}{fmt(row['paired_delta_pct'],2):>9}"
                  f"{row['paired_win_images']:>6}{row['paired_lose_images']:>7}")
        lines = verdict_lines(summary, depth_rows, target_channels)
        if lines:
            print("\n判据（置换变好 = 空间内容有害；变差 = 空间内容有用）")
            for line in lines:
                print("  - " + line)
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
