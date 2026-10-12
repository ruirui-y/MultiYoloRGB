# MultiYoloRGB

多 YOLO 版本（**YOLOv8 / YOLO11 / YOLO26**）在手部关键点检测上的对比实验：
同一份数据、同一套超参，比较 **RGB（3 通道）** 与 **RGBD（4 通道，含深度）** 两种输入。

> 本仓为独立实验仓。评估口径与 4 通道权重改造脚本复用主仓
> `yolov8_rgbd_detection/YJJ_Pose_Scripts/`，**不在此复制**，避免同一算法出现两份实现。

---

## 一、实验设计

| 项 | 设定 |
|---|---|
| 权重 | 官方 pose 预训练：`yolov8n-pose` / `yolo11n-pose` / `yolo26n-pose` |
| 数据 | 同一手部数据集；train/val = 受试者 A，test = 受试者 B（cross-subject） |
| 控制变量 | 同数据、同超参、同 seed，**仅替换起始权重** |
| 关键点 | K0~K6 共 7 点（COCO 17 点权重在训练时按 data yaml 自动重建） |
| 训练 | 120 epochs / batch 4 / imgsz 640 / patience 20 / seed 0，增强全关 |

**为什么只选这三代**：官方 assets（v8.4.0）中**仅这三代发布了 pose 预训练权重**。
YOLOv12 无 pose 权重；v5 / v9 / v10 / v13 / RT-DETR 无 pose 变体。

---

## 二、环境

```
Python      3.13.5
ultralytics 8.4.175      （必须 >= 8.4，否则无法加载 YOLO26）
torch       2.10.0+cu130
```

搭建（**必须带 `--system-site-packages`**，直接复用已有 CUDA torch，省 2.5 GB 下载）：

```cmd
H:\YJJ\Conda\python.exe -m venv --system-site-packages .venv
.venv\Scripts\python.exe -m pip install -U ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> ⚠️ 本机 `pypi.org` / `github.com` 直连不通，`pip` 必须带清华镜像；
> 权重下载需走镜像（如 `https://gh-proxy.com/https://github.com/...`）。
>
> ⚠️ **不要**在主仓根目录下跑本仓脚本，也**不要**继承 GUI 启动的子进程环境 ——
> 主仓内有 vendored `ultralytics 8.3.176`，会通过 `PYTHONPATH` 劫持 import，
> 导致 YOLO26 权重加载失败。本仓脚本已在 import 前主动剥离该路径。

---

## 三、目录

```
MultiYoloRGB/
├── weights/            官方原始权重（3 通道）
├── yolo11_4ch/         4 通道改造版权重
├── train_weight/       全部训练产物（统一入口，供测试/消融/探针共用）
├── runs/               运行产物（已被 .gitignore 忽略，约 80MB）
└── *.py                实验脚本
```

**训练产物命名规则**：`模型_模态_通道.pt`，例：

```
train_weight/
  yolov8n_rgb_3ch.pt     yolo11n_rgbd_4ch.pt
  yolo11n_rgb_3ch.pt     yolov8n_rgbd_4ch.pt
  yolo26n_rgb_3ch.pt
```

---

## 四、脚本

所有命令均在本仓根目录执行，前缀统一为 `.venv\Scripts\python.exe`。

| 脚本 | 作用 |
|---|---|
| `check_models.py` | 三代权重加载 + 推理输出结构一致性校验（boxes / keypoints 形状） |
| `smoke_train.py` | 冒烟训练（1 轮 × 3 版本），验证 7 点 head 重建与训练链路 |
| `train_all.py` | 正式训练，`--only <权重名>` 指定要跑的版本 |
| `eval_versions.py` | 三版本 cross-subject NME 评估 |
| `test_all.py` | 全部权重统一测试，按第一层通道数自动选 `rgb` / `rgbd` 目录 |
| `eval_4ch.py` | 单模型 4 通道评估 |
| `ablate_depth_nme.py` | 深度消融（NME 口径），5 个变体见下 |
| `box_kpt_relation.py` | 逐框统计：每图输出多少框、框的 IoU 与点的 NME 有何关系 |
| `conf_sweep.py` | conf 阈值 × 选框规则（取 IoU 最大 / 取 conf 最大）二维扫描 |
| `layer_probe.py` | 层特征探针：hook 层 9（SPPF）、层 10（C2PSA）、head 入口三层 |

**深度消融的 5 个变体**（`--variants`，默认全跑）：

```
true_depth             原始 4 通道
zero_depth             第 4 通道置 0
spatial_shuffle_depth  打乱深度像素位置（保留统计量）
flat_valid_depth       有效区填均值（抹掉内部几何）
blurred_depth          高斯模糊 31
```

**4 通道权重改造**（复用主仓脚本，不重写）：

```cmd
.venv\Scripts\python.exe H:\YJJ\Yolo_RGBD\yolov8_rgbd_detection\YJJ_Pose_Scripts\02_model\build_rgbd_pose_weights.py ^
    --input weights\yolo11n-pose.pt --output yolo11_4ch\yolo11n-pose_4ch.pt
```

规则：第 4 通道 = 前 3 通道均值初始化，其余权重原样保留。

---

## 五、评估口径

**NME**（Normalized Mean Error）

```
单点误差  d_k = sqrt((px-gx)^2 + (py-gy)^2)
归一化    nme_k = d_k / GT框对角线        ← 分母用真实框，不用预测框
参与范围  仅 GT visibility > 0 的关键点
单图 NME  = mean(有效点的 nme_k)
最终 NME  = 对测试集全部图片取 mean       ← 分母 = 公共集张数
```

**推理与命中**

```
conf = 0.05       ← 不要用 0.001（会放出大量低分噪音框，干扰选框规则）
imgsz = 640
命中判定：预测框与 GT 框 IoU >= 0.50
选框规则：取 IoU 最大的预测实例
```

> ⚠️ `conf=0.001` 是历史值。实测 `yolo11n_rgbd_4ch` 在该阈值下每图输出 176 个框
> （0.05 下只剩 1 个），"取 IoU 最大"会被噪音框带偏，使 NME 读数偏高约 16%。
> 详见 `conf_sweep.py` 的扫描结果。

**数据路径**（在仓库外，脚本内为绝对路径）：

```
H:\YJJ\Yolo_RGBD\Resource\session1_200146\dataset\
  hand\                训练用：train 52 / val 11 / test 12
  hand_cross_subject\  评估用：test 101（另一受试者）
  hand_common\         公共测试集：100 张（全部权重都命中的子集）
```

---

## 六、实验进展

- [x] 权重下载 + 独立环境搭建
- [x] 推理输出结构一致性校验（三代 `keypoints` / `boxes` 结构一致）
- [x] 冒烟训练 + 三版本正式训练（RGB 3ch）
- [x] 公共测试集筛选（多权重命中交集 → `hand_common`，100 张）
- [x] cross-subject 评估 + 选框规则校正（conf=0.05）
- [x] 深度消融（第一轮：用没用；第二轮：怎么用）
- [x] 逐框关系分析 + conf 阈值扫描
- [x] 层特征探针（定位到 v11 独有的 C2PSA 层）
- [ ] 4 通道改造扩展到 v8 / v26
- [ ] C2PSA 内部拆解（注意力权重对比）

---

## 七、已知坑（都踩过）

1. **4 通道图必须用 `cv2.IMREAD_UNCHANGED` 读** —— 默认读法会静默丢弃第 4 通道，
   程序不报错，但结果全错。
2. **RGB 存 `.jpg`，RGBD 存 `.png`** —— 同一帧扩展名不同，跨表对比必须按
   去掉扩展名的 stem 配对，否则一张都对不上。
3. **主仓的 vendored ultralytics 会劫持 import** —— 新脚本必须在 import 前剥离项目根路径。
4. **`optimizer=auto` 会忽略 `lr0`** —— ultralytics 会自动选 AdamW(lr=0.002)，
   日志里会打印 `ignoring 'lr0=...'`。
