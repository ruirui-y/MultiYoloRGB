r"""所有数据路径的唯一来源。

以前 20 多个脚本各自写死 ``H:\YJJ\Yolo_RGBD\Resource\session1_200146\dataset\hand``，
搬家一次要改十几处。现在统一从这里取：脚本只 ``import paths``，
数据目录真搬了也只改这一个文件（或用环境变量 ``RGBD_DATA_ROOT`` 覆盖）。

约定：
    REPO         本仓库（MultiYoloRGB）
    DATA_ROOT    Resource  —— 数据不进 git，约 7 GB
    BACKUP_ROOT  ResourceBack —— 只读快照，出事了从这里还原

按部位分：hand_back（手背）/ hand_palm（手心）/ abdomen（腹部），
每个部位下 ``raw/`` 是原始采集，``datasets/`` 是生成的 3 通道 / 4 通道数据集。

命令行自检：
    .venv\Scripts\python.exe paths.py
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- 根目录

REPO = Path(__file__).resolve().parent

#: 数据根。默认与本仓库同级；换机器或数据搬家时设环境变量 RGBD_DATA_ROOT。
DATA_ROOT = Path(os.environ.get("RGBD_DATA_ROOT", REPO.parent / "Resource")).resolve()

#: 只读快照根（不参与训练，只在还原时用）。
BACKUP_ROOT = Path(os.environ.get("RGBD_BACKUP_ROOT", REPO.parent / "ResourceBack")).resolve()

WEIGHTS_DIR = REPO / "train_weight"
RUNS_DIR = REPO / "runs"
DOCS_DIR = REPO / "docs"


# ---------------------------------------------------------------- 部位根目录

HAND_BACK_ROOT = DATA_ROOT / "hand_back"          # 手背（第二批搬迁后才有，现在还在 DATA_ROOT 下）
HAND_PALM_ROOT = DATA_ROOT / "hand_palm"          # 手心
ABDOMEN_ROOT = DATA_ROOT / "abdomen"              # 腹部


# ---------------------------------------------------------------- 手背：原始采集
#
# 已搬迁（2026-11-23 第二批）：hand_back/raw/sessionXX_xxxxxx
# 搬迁前是 DATA_ROOT/sessionXX_xxxxxx，改这一行即可回到旧结构。

SESSION_ROOT = HAND_BACK_ROOT / "raw"
SESSION_DIRS = {
    "01": SESSION_ROOT / "session1_200146",
    "02": SESSION_ROOT / "session2_200247",
    "03": SESSION_ROOT / "session3_200348",
    "04": SESSION_ROOT / "session4_200434",
    "05": SESSION_ROOT / "session5_200946",
    "06": SESSION_ROOT / "session6_201035",
    "07": SESSION_ROOT / "session7_201408",
    "08": SESSION_ROOT / "session8_201506",
    "09": SESSION_ROOT / "session9_201943",
    "10": SESSION_ROOT / "session10_202022",
    "11": SESSION_ROOT / "session11_202425",
    "12": SESSION_ROOT / "session12_202512",
}


# ---------------------------------------------------------------- 手背：10 套数据集
#
# 已搬迁（2026-11-23 第二批）：hand_back/datasets/<name>
# 搬迁前是 session1_200146/dataset/<name>

_DATASET_ROOT = HAND_BACK_ROOT / "datasets"

#: 存放"多套手部数据集"的父目录（hand/ hand_common/ hand_inv/ ... 都在它下面）。
#: 生成数据集之类的脚本要往里写新目录，所以单独暴露一个名字。
DATASET_PARENT = _DATASET_ROOT

#: 名称 -> 数据集目录（目录下含 rgb/ 与/或 rgbd/）
DATASETS: dict[str, Path] = {
    "hand": _DATASET_ROOT / "hand",                              # 主训练集 52/11/12
    "hand_test": _DATASET_ROOT / "hand_test",                    # 同人测试副本
    "hand_cross_subject": _DATASET_ROOT / "hand_cross_subject",  # 52/11/101
    "hand_common": _DATASET_ROOT / "hand_common",                # 公共测试集 100（正式口径）
    "hand_inv": _DATASET_ROOT / "hand_inv",                      # 深度反转训练集
    "hand_common_inv": _DATASET_ROOT / "hand_common_inv",        # 反转公共测试集
    "hand_center": _DATASET_ROOT / "hand_center",                # 中心化训练集
    "hand_common_center": _DATASET_ROOT / "hand_common_center",  # 中心化公共测试集
    "hand_gen": _DATASET_ROOT / "hand_gen",                      # 生成深度训练集
    "hand_common_gen": _DATASET_ROOT / "hand_common_gen",        # 生成深度公共测试集
}

#: 实验默认评估用的数据集（公共测试集），大量脚本此前用 ac._DEFAULT_DATASET
DEFAULT_DATASET = DATASETS["hand_common"]

#: 训练时用的数据集
TRAIN_DATASET = DATASETS["hand"]


# ---------------------------------------------------------------- 腹部

ABDOMEN_RAW = ABDOMEN_ROOT / "raw" / "abdomen_133"                 # image/ depth/ npy/ labels/
ABDOMEN_DATASET = ABDOMEN_ROOT / "datasets" / "abdomen_dataset"    # rgb/ rgbd/ splits.json

# ---------------------------------------------------------------- 手心

PALM_RAW = HAND_PALM_ROOT / "raw" / "palm_0916"    # g01~g05，每组 image/ depth/ npy/
PALM_GROUPS = {f"g{i:02d}": PALM_RAW / f"g{i:02d}" for i in range(1, 6)}


# ---------------------------------------------------------------- yaml 与权重

def yaml_of(dataset: str | Path, channels: int = 4) -> Path:
    """取某个数据集的 yaml：channels=3 找 rgb/data_rgb.yaml，4 找 rgbd/data_rgbd.yaml。

    数据集目录里若存在带后缀的专用配置（data_rgbd_inv / _center / _gen），优先用它们。
    """
    if isinstance(dataset, Path):
        directory = dataset
    else:
        directory = DATASETS[dataset]
    sub = "rgb" if channels == 3 else "rgbd"
    base = directory / sub
    candidates = [base / f"data_{sub}.yaml"]
    if channels == 4:
        candidates = [base / "data_rgbd_inv.yaml", base / "data_rgbd_center.yaml",
                      base / "data_rgbd_gen.yaml", base / "data_rgbd.yaml"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[-1]


def weight(name: str) -> Path:
    """按名字取权重：weight("yolo11n_rgbd_4ch") -> train_weight/yolo11n_rgbd_4ch.pt"""
    return WEIGHTS_DIR / f"{name}.pt"


def run_dir(*parts: str) -> Path:
    """runs 下的子目录，自动创建。"""
    directory = RUNS_DIR.joinpath(*parts)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


# ---------------------------------------------------------------- 自检

def _status(path: Path) -> str:
    if not path.exists():
        return "缺失"
    if path.is_file():
        return f"{path.stat().st_size / 1024:.0f} KB"
    return "目录"


def report() -> int:
    """打印所有路径及是否存在；返回缺失项数量。"""
    groups: list[tuple[str, list[tuple[str, Path]]]] = [
        ("根目录", [("REPO", REPO), ("DATA_ROOT", DATA_ROOT), ("BACKUP_ROOT", BACKUP_ROOT)]),
        ("手背原始", [(f"session{k}", v) for k, v in SESSION_DIRS.items()]),
        ("手背数据集", [(f"datasets[{k}]", v) for k, v in DATASETS.items()]),
        ("腹部", [("abdomen_raw", ABDOMEN_RAW), ("abdomen_dataset", ABDOMEN_DATASET)]),
        ("手心", [("palm_raw", PALM_RAW)] + [(f"palm[{k}]", v) for k, v in PALM_GROUPS.items()]),
    ]
    missing = 0
    for title, items in groups:
        print(f"\n[{title}]")
        for label, path in items:
            state = _status(path)
            if state == "缺失":
                missing += 1
            print(f"  {label:<22} {state:<8} {path}")
    print(f"\n缺失 {missing} 项")
    return missing


if __name__ == "__main__":
    raise SystemExit(0 if report() == 0 else 1)
