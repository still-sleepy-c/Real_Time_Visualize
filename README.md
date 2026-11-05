# 逐像素 Willert 三维速度场 · 可视化 (real_time)

基于「RAFT 光流 + RAFTStereo 立体」的逐像素 Willert 三维速度重建，
用 PySide6 + OpenCV 做的实时可视化程序。

## 目录结构

```
real_time/
├── version1-0.py                 # 主脚本（桌面可视化）
├── stereo_willert_per_pixel.py   # 极线校正 / 逐像素 Willert / 模型加载
├── core/                         # RAFT 光流代码（CBAM 版编码器）
├── RAFT-Stereo/
│   └── core_stereo/              # RAFTStereo 立体匹配代码
├── weights/
│   ├── 55000_raft_from_ck121.pth # RAFT 光流权重
│   └── raftstereo-realtime.pth   # RAFTStereo 立体权重
├── calib/
│   └── Cam_12.npz                # 相机参数 (K1,K2,R1,t1,R2,t2,...)
├── rect_maps_v2.npz              # 极线校正映射缓存（缺失会自动重建）
├── requirements.txt
└── README.md
```

## 环境要求

- Python 3.11
- NVIDIA GPU + CUDA（模型推理在 CUDA 上运行，`DEVICE = 'cuda'`）

### 安装依赖

```bash

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121


pip install -r requirements.txt
```

## 运行

```bash
cd real_time
python version1-0.py \
    --left_path  "D:/StereoData/test/left_paddle" \
    --right_path "D:/StereoData/test/right_paddle"
```

`--left_path` / `--right_path` 指向左右相机**图像序列**所在目录（同序号的左右帧需一一对应，
支持 `.png` / `.jpg`）。权重、相机参数、校正缓存默认使用 `./weights`、`./calib`、
`./rect_maps_v2.npz` 内的相对路径，一般无需指定。

## 主要命令行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--left_path` / `--right_path` | — | 左右图像序列目录 |
| `--calib` | `./calib/Cam_12.npz` | 相机参数文件 |
| `--flow_model` | `./weights/55000_raft_from_ck121.pth` | RAFT 光流权重 |
| `--stereo_model` | `./weights/raftstereo-realtime.pth` | RAFTStereo 权重 |
| `--rect_cache` | `./rect_maps_v2.npz` | 校正映射缓存 |
| `--frame` | `1` | 起始帧索引 (0-based) |
| `--num_frames` | `150` | 重建帧对数 |
| `--lam` | `0.2` | 正则化系数 λ / ε |
| `--global_lam` | 关 | 使用全局正则化（默认逐像素） |
| `--overlay_alpha` | `0.9` | 伪彩叠加透明度 |
| `--edge_filter` | 关 | Canny 边缘过滤（去边缘飞点） |
| `--iters_flow` / `--iters_stereo` | `20` | 光流 / 立体迭代次数 |
| `--save_dir` | `./saved_velocity` | 保存 NPY 的默认目录 |

## 界面功能

- **播放 / 暂停**：算出一帧回放一帧，按计算速度自动前进。
- **速度矢量箭头**：勾选「显示速度矢量箭头」并拖动密度滑块，可叠加 (U,V) 箭头。
- **正则化系数**：右侧面板实时调节 λ（或 ε），并切换「全局 / 逐像素」正则化。
- **背景透明度**：拖动 `overlay_alpha` 滑块，在纯伪彩与校正左图背景之间切换。
- **点击查询坐标**：点击画面任意点，右侧栏底部显示该点世界坐标 (Xw,Yw,Zw)。
- **保存速度 NPY**：暂停后，点击顶部「保存 NPY」，把当前帧三维速度
  `(U,V,W)` 存成 `(3,H,W)` 的 float32 `.npy` 文件（第 0 维 0=U, 1=V, 2=W，单位 m/frame）。
  保存路径在顶部「选择路径」里指定。

## 说明

- 画面显示的是**校正后左图**视角，速度场与显示图像处于同一（校正）坐标系。
- 速度单位为 **m/frame**（每帧位移），如需 m/s 请除以帧间隔时间。
