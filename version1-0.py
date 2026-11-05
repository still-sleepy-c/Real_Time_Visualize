
from __future__ import annotations

import sys
sys.path.append('core')
sys.path.append('RAFT-Stereo')
sys.path.append('.')

import argparse
import glob
import os

import cv2
import numpy as np
import torch

from PySide6 import QtCore, QtWidgets
from PySide6.QtGui import QImage, QPixmap

from stereo_willert_per_pixel import (
    StereoRectifier, PerPixelWillert,
    load_flow_model, load_stereo_model,
    run_raft, run_raft_stereo, load_image_numpy, DEVICE,
)


def compute_one_frame(frame_idx, left_files, right_files,
                      rectifier, willert, model_flow, model_stereo, args):
    """单帧全部计算: 加载 → 校正 → 推理 → Willert → 二维伪彩图。"""
    rawL_t = load_image_numpy(left_files[frame_idx])
    rawL_t1 = load_image_numpy(left_files[frame_idx + 1])
    rawR_t = load_image_numpy(right_files[frame_idx])
    rawR_t1 = load_image_numpy(right_files[frame_idx + 1])

    rectL_t, rectR_t = rectifier.rectify_pair(rawL_t, rawR_t)
    rectL_t1 = rectifier.rectify_single(rawL_t1, 'L')
    rectR_t1 = rectifier.rectify_single(rawR_t1, 'R')

    def np2t(img):
        t = torch.from_numpy(img).contiguous().permute(2, 0, 1).float()
        return t[None].to(DEVICE)

    with torch.no_grad():
        flow_L, pL = run_raft(model_flow, np2t(rectL_t), np2t(rectL_t1),
                              iters=args.iters_flow)
        flow_R, pR = run_raft(model_flow, np2t(rectR_t), np2t(rectR_t1),
                              iters=args.iters_flow)
        flow_L = pL.unpad(flow_L)
        flow_R = pR.unpad(flow_R)

        disp, pS = run_raft_stereo(model_stereo, np2t(rectL_t), np2t(rectR_t),
                                   iters=args.iters_stereo)
        disp = pS.unpad(disp.squeeze(1))
        disp_np = -(disp.squeeze().cpu().numpy())

        U, V, W, mask = willert.reconstruct(flow_L, flow_R, disp_np)

    n_raw = int(mask.sum())
    mag = np.sqrt(U ** 2 + V ** 2 + W ** 2)

    if args.edge_filter:
        edge = cv2.Canny(rectL_t, 50, 150)
        edge_dilated = cv2.dilate(edge, np.ones((7, 7), np.uint8))
        mask = mask & (edge_dilated == 0)

    if mask.any():
        vmax = float(np.percentile(mag[mask], 95))
    else:
        vmax = 1e-6
    vmax = max(vmax, 1e-6)

    mag_masked = np.where(mask, mag, np.nan)
    mag_norm = np.clip(np.nan_to_num(mag_masked) / vmax, 0.0, 1.0)
    mag_u8 = (mag_norm * 255).astype(np.uint8)
    color = cv2.applyColorMap(mag_u8, cv2.COLORMAP_TURBO)

    background = rectL_t.copy()
    a = float(args.overlay_alpha)
    if a >= 1.0:
        overlay = color.copy()
        overlay[~mask] = background[~mask]
    elif a <= 0.0:
        overlay = background
    else:
        overlay = background.copy()
        overlay[mask] = (color[mask] * a + background[mask] * (1.0 - a)).astype(np.uint8)

    # 世界坐标点云 (供点击查询)
    Xw, Yw, Zw = willert.world_points(disp_np)

    # 把 U, V, W, mask 也返回给主线程，用于在主线程快速画箭头 / 保存
    return {"frame": frame_idx, "color": overlay, "vmax": vmax,
            "n_valid": int(mask.sum()), "n_raw": n_raw,
            "U": U, "V": V, "W": W, "mask": mask,
            "Xw": Xw, "Yw": Yw, "Zw": Zw}


class ClickableLabel(QtWidgets.QLabel):
    """可点击的 QLabel，用于捕获画布上的点击坐标 (label 局部坐标)。"""
    clicked = QtCore.Signal(int, int)

    def mousePressEvent(self, event):
        pos = event.position()
        self.clicked.emit(int(pos.x()), int(pos.y()))
        super().mousePressEvent(event)


class TaskSignals(QtCore.QObject):
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)


class FrameTask(QtCore.QRunnable):
    def __init__(self, frame_idx, left_files, right_files,
                 rectifier, willert, model_flow, model_stereo, args):
        super().__init__()
        self.frame_idx = frame_idx
        self.left_files = left_files
        self.right_files = right_files
        self.rectifier = rectifier
        self.willert = willert
        self.model_flow = model_flow
        self.model_stereo = model_stereo
        self.args = args
        self.signals = TaskSignals()

    @QtCore.Slot()
    def run(self):
        try:
            payload = compute_one_frame(
                self.frame_idx, self.left_files, self.right_files,
                self.rectifier, self.willert,
                self.model_flow, self.model_stereo, self.args)
            self.signals.finished.emit(payload)
        except Exception as exc:
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")


class Viewer(QtWidgets.QMainWindow):
    def __init__(self, args, left_files, right_files,
                 rectifier, willert, model_flow, model_stereo):
        super().__init__()
        self.args = args
        self.left_files = left_files
        self.right_files = right_files
        self.rectifier = rectifier
        self.willert = willert
        self.model_flow = model_flow
        self.model_stereo = model_stereo

        self.frames = list(range(args.frame, min(args.frame + args.num_frames,
                                                 len(left_files) - 1)))
        if not self.frames:
            raise RuntimeError("没有可重建的帧对 (检查 --frame / --num_frames)")

        self.loading = False
        self.pending_index = None
        self.current_index = 0
        self._playing = False
        self._base_pixmap = None

        # 用于缓存当前帧的数据，以便快速重绘箭头
        self.cache_overlay = None
        self.cache_U = None
        self.cache_V = None
        self.cache_W = None
        self.cache_mask = None
        # 世界坐标点云 (点击查询用)
        self.cache_Xw = None
        self.cache_Yw = None
        self.cache_Zw = None
        # 画布缩放反算参数 (在 _update_display 中更新)
        self._scale = 1.0
        self._offset_x = 0.0
        self._offset_y = 0.0

        self.refresh_timer = QtCore.QTimer(self)
        self.refresh_timer.setSingleShot(True)
        self.refresh_timer.setInterval(200)
        self.refresh_timer.timeout.connect(self._recompute_current)

        # 用于箭头调节防抖
        self.arrow_timer = QtCore.QTimer(self)
        self.arrow_timer.setSingleShot(True)
        self.arrow_timer.setInterval(100)  # 100ms内如果还在拖滑块，就不重绘
        self.arrow_timer.timeout.connect(self._redraw_arrows)

        self._build_ui()
        self._request_frame(0)

    def _build_ui(self):
        self.setWindowTitle("逐像素 Willert 三维速度场 — 二维桌面可视化")
        self.resize(1200, 800)
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)

        # ===== 主布局: 上下 结构 =====
        main_layout = QtWidgets.QVBoxLayout(root)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # ===== 顶部: 保存栏 (选择路径 + 保存 NPY) =====
        save_bar = QtWidgets.QHBoxLayout()
        save_bar.addWidget(QtWidgets.QLabel("保存路径:"))
        self.save_path_edit = QtWidgets.QLineEdit(self.args.save_dir)
        save_bar.addWidget(self.save_path_edit, 1)
        self.browse_button = QtWidgets.QPushButton("选择路径")
        self.browse_button.clicked.connect(self._choose_save_path)
        save_bar.addWidget(self.browse_button)
        self.save_button = QtWidgets.QPushButton("保存 NPY")
        self.save_button.setEnabled(True)  # 初始未播放, 可保存
        self.save_button.clicked.connect(self._save_current_npy)
        save_bar.addWidget(self.save_button)
        main_layout.addLayout(save_bar)

        # ===== 上半部分: 左右 分割 (画布 + 参数面板) =====
        top_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(top_layout, 1)  # 权重1, 占满剩余空间

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        top_layout.addWidget(splitter)

        # --- 左侧: 主画布 ---
        self.image_label = ClickableLabel("正在计算…")
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setStyleSheet("background-color: #0a0a0a;")
        self.image_label.clicked.connect(self._on_canvas_clicked)
        splitter.addWidget(self.image_label)

        # --- 右侧: 参数面板 ---
        panel = QtWidgets.QWidget()
        panel.setMinimumWidth(260)
        panel.setMaximumWidth(340)
        form = QtWidgets.QVBoxLayout(panel)
        form.setContentsMargins(4, 4, 4, 4)
        splitter.addWidget(panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        self.status = QtWidgets.QLabel("正在加载…")
        self.status.setWordWrap(True)
        form.addWidget(self.status)

        form.addWidget(QtWidgets.QLabel("-" * 40))

        # ---- 箭头控制区 ----
        self.arrow_check = QtWidgets.QCheckBox("显示速度矢量箭头 (U,V)")
        self.arrow_check.setChecked(False)
        self.arrow_check.toggled.connect(self._on_arrow_settings_changed)
        form.addWidget(self.arrow_check)

        form.addWidget(QtWidgets.QLabel("箭头密度 (间距 px，越小越密):"))
        self.arrow_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.arrow_slider.setRange(10, 100)
        self.arrow_slider.setValue(30)
        self.arrow_slider.setEnabled(False) # 默认不开启，所以禁用滑块
        self.arrow_slider.valueChanged.connect(self._on_arrow_settings_changed)
        form.addWidget(self.arrow_slider)

        form.addWidget(QtWidgets.QLabel("-" * 40))

        # ---- 其他参数区 ----
        form.addWidget(QtWidgets.QLabel("Willert 正则化系数 λ / ε:"))
        self.lam_spin = QtWidgets.QDoubleSpinBox()
        self.lam_spin.setRange(0.0, 2.0)
        self.lam_spin.setDecimals(3)
        self.lam_spin.setSingleStep(0.01)
        self.lam_spin.setValue(float(self.willert.lam))
        self.lam_spin.valueChanged.connect(self._lam_changed)
        form.addWidget(self.lam_spin)

        self.global_check = QtWidgets.QCheckBox("全局正则化 (λ²I)")
        self.global_check.setChecked(not self.willert.per_pixel)
        self.global_check.toggled.connect(self._global_toggled)
        form.addWidget(self.global_check)

        form.addWidget(QtWidgets.QLabel("背景透明度 overlay_alpha:"))
        self.overlay_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.overlay_slider.setRange(0, 100)
        self.overlay_slider.setValue(int(round(self.args.overlay_alpha * 100)))
        self.overlay_slider.valueChanged.connect(self._overlay_changed)
        form.addWidget(self.overlay_slider)

        form.addWidget(QtWidgets.QLabel("背景=校正左图；|V| 伪彩: 蓝(慢)→青→黄→红(快)"))
        form.addStretch(1)

        # # 点击坐标显示区 (固定在右侧栏底部)
        # self.coord_label = QtWidgets.QLabel("点击画面查询该点世界坐标")
        # self.coord_label.setWordWrap(True)
        # self.coord_label.setStyleSheet("color: #ffd54f;")
        # form.addWidget(self.coord_label)

        # 点击坐标显示区 (固定在右侧栏底部)
        self.coord_label = QtWidgets.QLabel("点击画面查询该点世界坐标")
        self.coord_label.setWordWrap(True)
        # ✅ 将颜色修改为黑色
        self.coord_label.setStyleSheet("color: #000000; font-weight: bold;")
        form.addWidget(self.coord_label)

        # ===== 下半部分: 时间轴 =====
        bottom_widget = QtWidgets.QWidget()
        bottom_widget.setStyleSheet("QWidget { background-color: #262626; border-radius: 4px; } "
                                    "QLabel, QPushButton { color: #dddddd; }")
        bottom_layout = QtWidgets.QHBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(8, 6, 8, 6)

        self.play_button = QtWidgets.QPushButton("▶ 播放")
        self.play_button.setFixedWidth(90)
        self.play_button.clicked.connect(self._toggle_play)
        bottom_layout.addWidget(self.play_button)

        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setRange(0, len(self.frames) - 1)
        self.frame_slider.valueChanged.connect(self._frame_slider_changed)
        bottom_layout.addWidget(self.frame_slider, 1)

        self.frame_label = QtWidgets.QLabel("frame —")
        self.frame_label.setMinimumWidth(160)
        self.frame_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        bottom_layout.addWidget(self.frame_label)

        main_layout.addWidget(bottom_widget)

    def _request_frame(self, index: int):
        index = max(0, min(len(self.frames) - 1, int(index)))
        if self.loading:
            self.pending_index = index
            return
        self.loading = True
        self.frame_label.setText(f"计算 frame {self.frames[index]}…")
        task = FrameTask(self.frames[index], self.left_files, self.right_files,
                         self.rectifier, self.willert,
                         self.model_flow, self.model_stereo, self.args)
        self.active_task = task
        task.signals.finished.connect(self._frame_loaded, QtCore.Qt.QueuedConnection)
        task.signals.failed.connect(self._frame_failed, QtCore.Qt.QueuedConnection)
        QtCore.QThreadPool.globalInstance().start(task)

    def _frame_slider_changed(self, index: int):
        self.current_index = int(index)
        self._request_frame(self.current_index)

    @QtCore.Slot(object)
    def _frame_loaded(self, payload):
        self.loading = False
        self.active_task = None
        if payload is None:
            self.status.setText(f"frame {self.frames[self.current_index]}: 无有效点")
            self._schedule_pending()
            return

        frame = payload["frame"]
        self.current_index = self.frames.index(frame)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.current_index)
        self.frame_slider.blockSignals(False)

        # 1. 缓存结果，用于主线程快速画箭头
        self.cache_overlay = payload["color"]
        self.cache_U = payload["U"]
        self.cache_V = payload["V"]
        self.cache_W = payload["W"]
        self.cache_mask = payload["mask"]
        self.cache_Xw = payload["Xw"]
        self.cache_Yw = payload["Yw"]
        self.cache_Zw = payload["Zw"]

        # 2. 触发重绘 (带上箭头)
        self._redraw_arrows()

        self.frame_label.setText(f"frame {frame}  ({self.current_index + 1}/{len(self.frames)})")
        ratio = 100 * (1 - payload["n_valid"] / max(payload["n_raw"], 1))
        self.status.setText(f"有效点 {payload['n_valid']:,}  "
                            f"(边缘过滤 -{ratio:.1f}%)\n"
                            f"|V| 上限 {payload['vmax']:.5f} m/frame")
        self._schedule_pending()

    def _redraw_arrows(self):
        """主线程快速重绘：基于缓存的底图，按需画上箭头"""
        if self.cache_overlay is None:
            return

        img = self.cache_overlay.copy()

        if self.arrow_check.isChecked():
            step = self.arrow_slider.value()
            U = self.cache_U
            V = self.cache_V
            mask = self.cache_mask

            # 放大倍数：速度物理单位通常很小，为了画面上能看清需要放大
            arrow_scale = 5000.0

            for y in range(step // 2, img.shape[0], step):
                for x in range(step // 2, img.shape[1], step):
                    if mask[y, x]:
                        # 注意：V 取负，因为图像坐标系 Y 轴向下，而物理坐标系 Y 轴向上
                        dx = int(U[y, x] * arrow_scale)
                        dy = int(-V[y, x] * arrow_scale)
                        cv2.arrowedLine(img, (x, y), (x + dx, y + dy), (255, 255, 255), 1, tipLength=0.3)

        # 更新画布
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self._base_pixmap = QPixmap.fromImage(qimg)
        self._update_display()

    def _on_arrow_settings_changed(self):
        """当开关或滑块变化时触发"""
        self.arrow_slider.setEnabled(self.arrow_check.isChecked())
        self.arrow_timer.start() # 防抖，延迟 100ms 重绘

    def _on_canvas_clicked(self, x, y):
        """点击画布: 把 label 坐标反算回原图像素, 查询该点世界坐标。"""
        if self._base_pixmap is None or self.cache_mask is None:
            return
        u = int(round((x - self._offset_x) / self._scale))
        v = int(round((y - self._offset_y) / self._scale))
        h, w = self.cache_mask.shape
        if not (0 <= u < w and 0 <= v < h):
            self.coord_label.setText(f"点击 ({x},{y}) 超出图像范围")
            return
        if not self.cache_mask[v, u]:
            self.coord_label.setText(f"像素({u},{v}) 无有效重建")
            return
        Xw = self.cache_Xw[v, u]
        Yw = self.cache_Yw[v, u]
        Zw = self.cache_Zw[v, u]
        self.coord_label.setText(
            f"像素({u},{v})\n世界坐标: ({Xw:.4f}, {Yw:.4f}, {Zw:.4f}) m")

    def _update_display(self):
        if self._base_pixmap is None:
            return
        scaled = self._base_pixmap.scaled(
            self.image_label.size(), QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)

        # 记录反算参数: label 坐标 → 原图像素坐标
        orig_w = self._base_pixmap.width()
        if orig_w > 0:
            self._scale = scaled.width() / orig_w
            self._offset_x = (self.image_label.width() - scaled.width()) / 2.0
            self._offset_y = (self.image_label.height() - scaled.height()) / 2.0

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()

    @QtCore.Slot(str)
    def _frame_failed(self, message):
        self.loading = False
        self.active_task = None
        self.status.setText("计算失败：" + message)
        self._schedule_pending()

    def _schedule_pending(self):
        if self.pending_index is not None and self.pending_index != self.current_index:
            pending = self.pending_index
            self.pending_index = None
            self._request_frame(pending)
        elif self._playing:
            self._advance()
        else:
            self.pending_index = None

    def _toggle_play(self):
        self._playing = not self._playing
        self.play_button.setText("⏸ 暂停" if self._playing else "▶ 播放")
        self.save_button.setEnabled(not self._playing)  # 播放中禁用保存
        if self._playing and not self.loading:
            self._advance()

    def _advance(self):
        nxt = (self.current_index + 1) % len(self.frames)
        if self.args.frame_duration_ms > 0:
            QtCore.QTimer.singleShot(self.args.frame_duration_ms,
                                     lambda: self._request_frame(nxt))
        else:
            self._request_frame(nxt)

    def _choose_save_path(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "选择保存目录", self.save_path_edit.text())
        if d:
            self.save_path_edit.setText(d)

    def _save_current_npy(self):
        if self.cache_U is None:
            self.status.setText("当前无已计算帧，无法保存")
            return
        save_dir = self.save_path_edit.text().strip()
        if not save_dir:
            self.status.setText("请先选择保存路径")
            return
        os.makedirs(save_dir, exist_ok=True)
        frame = self.frames[self.current_index]
        # 三维速度 (世界系, m/frame): 第0维 0=U, 1=V, 2=W
        vel = np.stack([self.cache_U, self.cache_V, self.cache_W], axis=0)
        vel = vel.astype(np.float32)
        path = os.path.join(save_dir, f"velocity_frame_{frame:04d}.npy")
        np.save(path, vel)
        self.status.setText(
            f"已保存: {path}\n形状 {vel.shape} (0=U, 1=V, 2=W) 单位 m/frame")

    def _lam_changed(self, value):
        self.willert.lam = float(value)
        self.refresh_timer.start()

    def _global_toggled(self, checked):
        self.willert.per_pixel = not checked
        self.refresh_timer.start()

    def _overlay_changed(self, value):
        self.args.overlay_alpha = value / 100.0
        self.refresh_timer.start()

    def _recompute_current(self):
        self._request_frame(self.current_index)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calib', default='./calib/Cam_12.npz')
    parser.add_argument('--left_path', default=r"D:\StereoData\test\left_paddle")
    parser.add_argument('--right_path', default=r"D:\StereoData\test\right_paddle")
    parser.add_argument('--flow_model', default='./weights/tuned_optical_flow.pth')
    parser.add_argument('--stereo_model', default='./weights/raftstereo-realtime.pth')
    parser.add_argument('--rect_cache', default='./rect_maps_v2.npz')
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--iters_flow', type=int, default=20)
    parser.add_argument('--iters_stereo', type=int, default=20)

    # --- 核心改动区域 ---
    parser.add_argument('--lam', type=float, default=0.2, help='正则化系数 λ (默认 0.2)')
    parser.add_argument('--global_lam', action='store_true', help='使用全局正则化 (默认逐像素)')
    # --------------------

    parser.add_argument('--frame', type=int, default=1, help='起始帧索引 (0-based)')
    parser.add_argument('--num_frames', type=int, default=150, help='重建的帧对数')
    parser.add_argument('--edge_filter', action='store_true', default=False,
                        help='Canny 边缘过滤 (去掉边缘飞点)')
    parser.add_argument('--overlay_alpha', type=float, default=0.9,
                        help='伪彩叠加透明度: 0=只显示校正左图背景, 1=完全覆盖背景')
    parser.add_argument('--frame_duration_ms', type=int, default=0,
                        help='播放时帧间最小间隔(ms); 0=全速回放')
    parser.add_argument('--save_dir', default='./saved_velocity',
                        help='保存 NPY 的默认目录 (界面顶部可改)')
    return parser.parse_args()


def main():
    args = parse_args()

    calib = np.load(args.calib)
    K1, K2 = calib['K1'], calib['K2']
    R1_cam = calib['R1']
    t1_cam = calib['t1'].reshape(3, 1) if calib['t1'].ndim == 1 else calib['t1']
    R2_cam = calib['R2']
    t2_cam = calib['t2'].reshape(3, 1) if calib['t2'].ndim == 1 else calib['t2']

    R_stereo = R2_cam @ R1_cam.T
    T_stereo = t2_cam - R_stereo @ t1_cam

    left_files = sorted(glob.glob(os.path.join(args.left_path, '*.png')) +
                        glob.glob(os.path.join(args.left_path, '*.jpg')))
    right_files = sorted(glob.glob(os.path.join(args.right_path, '*.png')) +
                         glob.glob(os.path.join(args.right_path, '*.jpg')))
    assert len(left_files) == len(right_files), "左右帧数不一致"

    sample = load_image_numpy(left_files[0])
    img_h, img_w = sample.shape[:2]
    print(f"[Init] 图像 {img_w}×{img_h}")

    if os.path.exists(args.rect_cache):
        rectifier = StereoRectifier.load(args.rect_cache)
        if rectifier.img_w != img_w or rectifier.img_h != img_h:
            rectifier = StereoRectifier(K1, K2, R_stereo, T_stereo,
                                        img_w, img_h, alpha=args.alpha)
            rectifier.save(args.rect_cache)
    else:
        rectifier = StereoRectifier(K1, K2, R_stereo, T_stereo,
                                    img_w, img_h, alpha=args.alpha)
        rectifier.save(args.rect_cache)

    print("[Init] 加载 RAFT 光流 + RAFTStereo ...")
    model_flow = load_flow_model(args.flow_model)
    model_stereo = load_stereo_model(args.stereo_model)

    # 默认是逐像素正则化 (not args.global_lam 为 True)，且 lam=0.2
    # 传 t1_cam 让 world_points 输出真正的世界坐标 (而非校正坐标系坐标)
    willert = PerPixelWillert(rectifier, R1_cam, lam=args.lam,
                              per_pixel=not args.global_lam, t1_cam=t1_cam)

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Willert 2D Desktop Viewer")
    try:
        window = Viewer(args, left_files, right_files,
                        rectifier, willert, model_flow, model_stereo)
    except Exception as exc:
        QtWidgets.QMessageBox.critical(None, "无法启动可视化", str(exc))
        return 1
    window.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())