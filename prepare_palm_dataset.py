r"""准备手心（palm）数据集：g01 做训练+验证，g02 只做测试。

为什么不直接用主仓的 ``prepare_pose_rgbd_dataset.py``：
那个脚本固定按「时间戳 70/15/15」切一个目录，会把 g01 和 g02 混进同一个 split。
手心这里是**按人分批**的：
    g01（101 张，同一个人）→ train 81 / val 20（按时间戳顺序切）
    g02（100 张，另一个人）→ test 100（训练全程不看）

融合与校验逻辑直接复用主仓脚本的 5 个函数，不复制实现：
    collect / fuse_pair_rawdepth / verify_fused_png / validate_labels / build_yaml

产物（``paths.PALM_DATASET``）：
    rgb/{images,labels}/{train,val,test}      3 通道
    rgbd/{images,labels}/{train,val,test}     4 通道（第 4 层 = 深度）
    rgb/data_rgb.yaml、rgbd/data_rgbd.yaml
    splits.json、dataset_report.json

用法：
    .venv\Scripts\python.exe prepare_palm_dataset.py
    .venv\Scripts\python.exe prepare_palm_dataset.py --force
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path

import paths

# 主仓的数据准备脚本：复用它的融合与标签校验，避免两份实现漂移
_MAIN_SCRIPT = (paths.REPO.parent / "yolov8_rgbd_detection" / "YJJ_Pose_Scripts"
                / "01_data" / "prepare_pose_rgbd_dataset.py")

# 手心实测：两组 1%~99% 都落在 1488~1925mm，取整到 1500~1920
DEFAULT_LOW = 1500.0
DEFAULT_HIGH = 1920.0
TRAIN_GROUP = "g01"     # 出 train / val
TEST_GROUP = "g02"      # 只出 test
VAL_RATIO = 0.20        # g01 内部切 20% 做验证


def load_main_module():
    """按文件路径加载主仓脚本，返回模块对象。"""
    if not _MAIN_SCRIPT.is_file():
        raise FileNotFoundError(f"找不到主仓数据准备脚本: {_MAIN_SCRIPT}")
    spec = importlib.util.spec_from_file_location("prepare_pose_rgbd_dataset", _MAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="手心数据集（g01 训验 / g02 测试）")
    parser.add_argument("--output", default=str(paths.PALM_DATASET))
    parser.add_argument("--class-name", default="palm")
    parser.add_argument("--depth-low", type=float, default=DEFAULT_LOW)
    parser.add_argument("--depth-high", type=float, default=DEFAULT_HIGH)
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--force", action="store_true", help="目标已存在时先删掉")
    return parser.parse_args()


def group_files(group_dir: Path) -> tuple[dict, dict, dict]:
    """收集一组里的 rgb / npy / label，按时间戳做键（不含前缀）。"""
    rgb_dir = group_dir / "image"
    npy_dir = group_dir / "npy"
    label_dir = group_dir / "image" / "labels"
    rgb = {p.stem.replace("color_", ""): p for p in rgb_dir.glob("*.jpg")}
    rgb.update({p.stem.replace("color_", ""): p for p in rgb_dir.glob("*.png")})
    npy = {p.stem.replace("depth_", ""): p for p in npy_dir.glob("*.npy")}
    label = {p.stem.replace("color_", ""): p for p in label_dir.glob("*.txt")}
    return rgb, npy, label


def main() -> None:
    args = parse_args()
    module = load_main_module()

    output = Path(args.output)
    staging = output.parent / f".{output.name}_tmp"
    fused_dir = staging / "fused"

    train_group = paths.PALM_GROUPS[TRAIN_GROUP]
    test_group = paths.PALM_GROUPS[TEST_GROUP]

    # ---- 1. 收集并确认每组配对完整 ----
    collected = {}
    for name, group_dir in ((TRAIN_GROUP, train_group), (TEST_GROUP, test_group)):
        rgb, npy, label = group_files(group_dir)
        paired = sorted(set(rgb) & set(npy) & set(label))
        dropped = sorted((set(rgb) & set(npy)) - set(label))
        print(f"[{name}] 图 {len(rgb)} 深度 {len(npy)} 标签 {len(label)} → 可用 {len(paired)}"
              + (f"（未标注 {len(dropped)} 张，跳过）" if dropped else ""))
        if not paired:
            raise RuntimeError(f"{name} 没有任何配对完整的样本")
        collected[name] = {"dir": group_dir, "ids": paired, "rgb": rgb, "npy": npy, "label": label}

    # ---- 2. 校验标签，推断关键点数量 ----
    all_labels = {}
    for name, data in collected.items():
        for ts in data["ids"]:
            all_labels[ts] = data["label"][ts]
    keypoint_count = module.validate_labels(all_labels)
    print(f"[标签] 关键点数量 K={keypoint_count}，总样本 {len(all_labels)}")

    # ---- 3. 划分：g01 内部切 train/val，g02 全给 test ----
    train_ids = collected[TRAIN_GROUP]["ids"]                    # 已按时间戳升序
    val_count = round(len(train_ids) * args.val_ratio)
    split_train = train_ids[:len(train_ids) - val_count]
    split_val = train_ids[len(train_ids) - val_count:]
    split_test = collected[TEST_GROUP]["ids"]
    print(f"[划分] train={len(split_train)} val={len(split_val)} test={len(split_test)}"
          f"（train/val 来自 {TRAIN_GROUP}，test 来自 {TEST_GROUP}）")

    # ---- 4. 融合 4 通道图（写进 staging，校验通过后才发布）----
    if staging.exists():
        shutil.rmtree(staging)
    fused_dir.mkdir(parents=True, exist_ok=True)
    fused = 0
    for name, data in collected.items():
        for ts in data["ids"]:
            out_path = fused_dir / f"color_{ts}.png"
            module.fuse_pair_rawdepth(data["rgb"][ts], data["npy"][ts], out_path,
                                      args.depth_low, args.depth_high)
            module.verify_fused_png(data["rgb"][ts], data["npy"][ts], out_path,
                                    args.depth_low, args.depth_high)
            fused += 1
    print(f"[融合] {fused} 张 4 通道 PNG（深度 {args.depth_low:.0f}~{args.depth_high:.0f} mm）")

    # ---- 5. 复制到 rgb / rgbd 的 train/val/test ----
    dataset = staging / "dataset"
    for split, ids in (("train", split_train), ("val", split_val), ("test", split_test)):
        rgb_images = dataset / "rgb" / "images" / split
        rgbd_images = dataset / "rgbd" / "images" / split
        rgb_labels = dataset / "rgb" / "labels" / split
        rgbd_labels = dataset / "rgbd" / "labels" / split
        for directory in (rgb_images, rgbd_images, rgb_labels, rgbd_labels):
            directory.mkdir(parents=True, exist_ok=True)
        for ts in ids:
            # 找出这个 ts 属于哪一组，取对应的源图与标签
            data = next(d for d in collected.values() if ts in d["npy"])
            shutil.copy2(data["rgb"][ts], rgb_images / f"color_{ts}{data['rgb'][ts].suffix}")
            shutil.copy2(fused_dir / f"color_{ts}.png", rgbd_images / f"color_{ts}.png")
            shutil.copy2(data["label"][ts], rgb_labels / f"color_{ts}.txt")
            shutil.copy2(data["label"][ts], rgbd_labels / f"color_{ts}.txt")
    print("[复制] rgb / rgbd × train/val/test 完成")

    # ---- 6. yaml + splits.json + dataset_report.json ----
    module.build_yaml(dataset / "rgb" / "data_rgb.yaml", args.class_name,
                      keypoint_count, 3, False, dataset_path=output / "rgb")
    module.build_yaml(dataset / "rgbd" / "data_rgbd.yaml", args.class_name,
                      keypoint_count, 4, True, dataset_path=output / "rgbd")

    splits = {
        "class": args.class_name,
        "kpt_shape": [keypoint_count, 3],
        "n_total": fused,
        "n": {"train": len(split_train), "val": len(split_val), "test": len(split_test)},
        "source": {TRAIN_GROUP: "train+val", TEST_GROUP: "test"},
        "train": [f"color_{t}" for t in split_train],
        "val": [f"color_{t}" for t in split_val],
        "test": [f"color_{t}" for t in split_test],
        "depth_range_mm": [args.depth_low, args.depth_high],
    }
    (dataset / "splits.json").write_text(json.dumps(splits, indent=2, ensure_ascii=False),
                                        encoding="utf-8")

    def count(directory: Path) -> int:
        return len(list(directory.glob("*"))) if directory.exists() else 0

    per = {"train": len(split_train), "val": len(split_val), "test": len(split_test)}
    validation_passed = all(
        count(dataset / channel / kind / split) == per[split]
        for channel in ("rgb", "rgbd")
        for kind in ("images", "labels")
        for split in per
    )
    report = {
        "total_samples": fused,
        "train_samples": per["train"],
        "val_samples": per["val"],
        "test_samples": per["test"],
        "keypoint_count": keypoint_count,
        "rgb_channels": 3,
        "rgbd_channels": 4,
        "depth_range_mm": [args.depth_low, args.depth_high],
        "split_rule": f"{TRAIN_GROUP} 按时间戳切 train/val，{TEST_GROUP} 整体作为 test",
        "validation_passed": validation_passed,
    }
    (dataset / "dataset_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[report] validation_passed={validation_passed}")
    if not validation_passed:
        raise RuntimeError("最终校验未通过，未写入正式目录")

    # ---- 7. 发布 ----
    if output.exists():
        if not args.force:
            raise FileExistsError(f"{output} 已存在；要覆盖请加 --force")
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(dataset), str(output))
    shutil.rmtree(staging, ignore_errors=True)

    for yaml_path in (output / "rgb" / "data_rgb.yaml", output / "rgbd" / "data_rgbd.yaml"):
        module.verify_published_yaml(yaml_path)

    print(f"[READY] {output}")
    print(f"[READY] total={fused} train={per['train']} val={per['val']} test={per['test']} "
          f"kpt={keypoint_count} rgb_ch=3 rgbd_ch=4")


if __name__ == "__main__":
    main()
