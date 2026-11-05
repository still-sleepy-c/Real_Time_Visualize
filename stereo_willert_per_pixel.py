
import sys
sys.path.append("core")
sys.path.append("RAFT-Stereo")
"""
print(sys.path)之后是
['', 'D:\\py3.11\\python311.zip', 'D:\\py3.11\\DLLs', 'D:\\py3.11\\Lib', 'D:\\py3.11', 
'C:\\Users\\Lenovo\\Desktop\\PythonProject1\\.venv', 'C:\\Users\\Lenovo\\Desktop\\PythonProject1\\.venv\\Lib\\site-packages', 
'core', 'RAFT-Stereo']
"""

import argparse
import os
import glob
import time
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

from raft import RAFT
from core_stereo.raft_stereo import RAFTStereo
from utils.utils import InputPadder

DEVICE = 'cuda'


# ===================================================================
# Part A: 极线校正器
# ===================================================================

class StereoRectifier:
    """
    用 K1, K2, R, t 做 Bouguet 极线校正
    校正后: 两台相机平行, 光轴对齐, 极线水平, 新内参 K_rect 相同
    """

    def __init__(self, K1, K2, R, t, img_w, img_h, alpha=0.0):
        """
        Args:
            K1, K2: 原始内参 (3×3)
            R, t: cv2.stereoCalibrate 的相机1→相机2 变换
            img_w, img_h: 图像尺寸
            alpha: 0=无黑边裁剪, 1=保留全部像素
        """
        self.img_w, self.img_h = img_w, img_h
        self.img_size = (img_w, img_h)

        dist1 = np.zeros(5)
        dist2 = np.zeros(5)

        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            K1, dist1, K2, dist2, self.img_size, R, t,
            alpha=alpha
        )

        


        self.R1_rect = R1   # 相机1 校正旋转
        self.R2_rect = R2   # 相机2 校正旋转
        self.P1 = P1        # 相机1 校正投影矩阵
        self.P2 = P2        # 相机2 校正投影矩阵
        self.Q = Q           # 视差→深度矩阵

        # 校正后内参 (两相机相同)
        self.K_rect = P1[:3, :3]

        # 校正映射表
        self.mapL_x, self.mapL_y = cv2.initUndistortRectifyMap(
            K1, dist1, R1, P1, self.img_size, cv2.CV_32FC1)
        self.mapR_x, self.mapR_y = cv2.initUndistortRectifyMap(
            K2, dist2, R2, P2, self.img_size, cv2.CV_32FC1)

        fx = self.K_rect[0, 0]
        cx = self.K_rect[0, 2]
        cy = self.K_rect[1, 2]
        print(f"[Rectifier] K_rect: fx={fx:.1f}, cx={cx:.1f}, cy={cy:.1f}")
        print(f"[Rectifier] P2[0,3] = {P2[0,3]:.1f} → baseline* = {abs(P2[0,3]/fx):.3f} m")
        print(f"[Rectifier] Maps ready: {self.img_size}")

    def rectify_pair(self, imgL, imgR):
        """两路一起校正"""
        L = cv2.remap(imgL, self.mapL_x, self.mapL_y, cv2.INTER_LINEAR)
        R = cv2.remap(imgR, self.mapR_x, self.mapR_y, cv2.INTER_LINEAR)
        return L, R

    def rectify_single(self, img, side='L'):
        """单路校正"""
        m_x, m_y = (self.mapL_x, self.mapL_y) if side == 'L' else (self.mapR_x, self.mapR_y)
        return cv2.remap(img, m_x, m_y, cv2.INTER_LINEAR)

    def save(self, filepath):
        np.savez(filepath,
                 mapL_x=self.mapL_x, mapL_y=self.mapL_y,
                 mapR_x=self.mapR_x, mapR_y=self.mapR_y,
                 K_rect=self.K_rect, Q=self.Q,
                 R1_rect=self.R1_rect, R2_rect=self.R2_rect,
                 img_w=self.img_w, img_h=self.img_h)
        print(f"[Rectifier] Saved: {filepath}")

    @classmethod
    def load(cls, filepath):
        """从保存的 npz 重建 (无需 K1,K2,R,t)"""
        data = np.load(filepath, allow_pickle=False)
        self = cls.__new__(cls)
        self.mapL_x = data['mapL_x']
        self.mapL_y = data['mapL_y']
        self.mapR_x = data['mapR_x']
        self.mapR_y = data['mapR_y']
        self.K_rect = data['K_rect']
        self.Q = data['Q']
        self.R1_rect = data['R1_rect']
        self.R2_rect = data['R2_rect']
        self.img_w = int(data['img_w'])
        self.img_h = int(data['img_h'])
        self.img_size = (self.img_w, self.img_h)
        self.P1 = None  # 离线保存时不需要
        self.P2 = None
        print(f"[Rectifier] Loaded: {filepath}")
        return self


# ===================================================================
# Part B: 逐像素 Willert 重建器
# ===================================================================

class PerPixelWillert:
    """
    校正坐标系下逐像素 3C 速度重建

    校正后:
      左相机 = 世界原点 (校正坐标系)
      右相机 = (+B, 0, 0)
      两相机内参相同 (K_rect)
    """

    def __init__(self, rectifier, R1_cam, lam=0.0, per_pixel=False, t1_cam=None):
        """
        Args:
            rectifier: StereoRectifier, 含 K_rect, Q
            R1_cam: (3,3) 原始相机1 的世界→相机 旋转
                    用于将结果转回世界坐标系
            lam: (float) Tikhonov 正则系数
                 - per_pixel=False: 全局 λ, 求解 V = (JᵀJ + λ²I)⁻¹ Jᵀ d
                 - per_pixel=True:  λ 作为相对系数 ε, 逐像素 λ_i = ε·‖J_i‖_F
            per_pixel: (bool) True 时逐像素正则 (lam 解释为相对系数 ε)
            t1_cam: (3,1) 相机1 平移 (世界系), 用于 world_points 输出世界坐标;
                    传 None 则 world_points 返回校正坐标系坐标
        """
        self.K = rectifier.K_rect.astype(np.float32)
        self.Q = rectifier.Q.astype(np.float32)
        self.R1_cam = R1_cam.astype(np.float32)  # 世界→Cam1
        self.R1_rect = rectifier.R1_rect.astype(np.float32)  # Cam1→校正 旋转
        self.lam = float(lam)
        self.per_pixel = bool(per_pixel)
        self.fx = float(self.K[0, 0])
        self.fy = float(self.K[1, 1])
        self.cx = float(self.K[0, 2])
        self.cy = float(self.K[1, 2])
        self.H = rectifier.img_h
        self.W = rectifier.img_w
        self.t1_cam = None if t1_cam is None else np.asarray(t1_cam, dtype=np.float32).reshape(3, 1)

    # ---- 三角测量 ----
    def triangulate(self, disp):
        """
        disp: (H, W) float32, 视差图 (左→右)

        Returns:
            X, Y, Z: (H, W) 校正坐标系下的左相机坐标
        """
        H, W = disp.shape

        u, v = np.meshgrid(np.arange(W), np.arange(H))  # pixel coords
        u = u.astype(np.float32)
        v = v.astype(np.float32)

        # (X, Y, Z, W) = Q @ [u, v, d, 1]
        pts = np.stack([u, v, disp, np.ones_like(disp)], axis=-1)  # (H, W, 4)
        pts = pts.reshape(-1, 4).T  # (4, H*W)
        XYZW = self.Q @ pts  # (4, H*W)
        X = (XYZW[0] / XYZW[3]).reshape(H, W)
        Y = (XYZW[1] / XYZW[3]).reshape(H, W)
        Z = (XYZW[2] / XYZW[3]).reshape(H, W)

        return X, Y, Z

    def world_points(self, disp):
        """由视差得到世界坐标点云 (X_w, Y_w, Z_w) → (H,W) 三个数组

        校正坐标系原点在左相机光心; 转回世界系:
            P_world = R1_camᵀ·R1_rectᵀ·P_rect - R1_camᵀ·t1_cam
        若 self.t1_cam 为 None, 返回校正坐标系坐标 (跳过平移)。
        """
        X, Y, Z = self.triangulate(disp)               # 校正坐标系 (H,W)
        H, W = disp.shape
        R_rect2world = self.R1_cam.T @ self.R1_rect.T   # (3,3)
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)  # (N,3)
        pts_w = (R_rect2world @ pts.T).T                # (N,3)
        if self.t1_cam is not None:
            pts_w = pts_w - (self.R1_cam.T @ self.t1_cam).ravel()  # 平移 (广播减每列)
        Xw = pts_w[:, 0].reshape(H, W)
        Yw = pts_w[:, 1].reshape(H, W)
        Zw = pts_w[:, 2].reshape(H, W)
        return Xw, Yw, Zw

    # ---- 采样右图时间流 ----
    def sample_flow_R(self, flow_R, disp):
        """
        flow_R: (1, 2, H, W) tensor, 右图时间流
        disp:   (H, W) float32, 左→右视差

        flow_R 在右图坐标系; 对于左图像素 (i,j),
        需要在右图 (i, j-d) 处采样 flow_R

        Returns:
            flow_R_sampled: (1, 2, H, W) tensor
        """
        B, C, H, W = flow_R.shape
        disp_t = torch.from_numpy(disp).float().to(flow_R.device)

        # grid_sample 坐标: (B, H, W, 2), 归一化到 [-1,1]
        gy, gx = torch.meshgrid(
            torch.arange(H, device=flow_R.device, dtype=torch.float32),
            torch.arange(W, device=flow_R.device, dtype=torch.float32),
            indexing='ij')

        gx_shifted = gx - disp_t  # 右图对应列
        gy_shifted = gy           # 极线校正后同行

        gx_norm = 2.0 * gx_shifted / (W - 1) - 1.0
        gy_norm = 2.0 * gy_shifted / (H - 1) - 1.0

        grid = torch.stack([gx_norm, gy_norm], dim=-1).unsqueeze(0)  # (1, H, W, 2)
        sampled = F.grid_sample(flow_R, grid, mode='bilinear',
                                padding_mode='border', align_corners=True)


        return sampled

    # ---- Jacobian ----
    def compute_jacobian(self, X, Y, Z):
        """
        校正坐标系下解析 Jacobian

        uL = uR + fx*B/Z,  vL = vR

        J = [ fx/Z,    0,   -(uL-cx)/Z ]   ← x1 (CamL u)
            [   0,  fy/Z,   -(vL-cy)/Z ]   ← y1 (CamL v)
            [ fx/Z,    0,   -(uR-cx)/Z ]   ← x2 (CamR u)
            [   0,  fy/Z,   -(vR-cy)/Z ]   ← y2 (CamR v)

        uR = uL - fx*B/Z (由视差关系)

        Returns:
            J: (H, W, 4, 3) numpy
        """
        H, W = X.shape
        Z_clip = np.maximum(Z, 1e-6)

        uL = self.fx * X / Z_clip + self.cx
        vL = self.fy * Y / Z_clip + self.cy

        # 基线 B (米): 从 Q[3,2] = 1/B 取
        # 注意: Q[0,3] 是 -cx (主点, 像素), 不是基线; 之前误用了 Q[0,3]
        baseline = 1.0 / float(self.Q[3, 2]) if self.Q[3, 2] != 0 else 1.0
        uR = uL - self.fx * baseline / Z_clip

        J = np.zeros((H, W, 4, 3), dtype=np.float32)
        fxZ = self.fx / Z_clip
        fyZ = self.fy / Z_clip

        J[:, :, 0, 0] = fxZ          # ∂uL/∂X
        J[:, :, 0, 1] = 0.0          # ∂uL/∂Y
        J[:, :, 0, 2] = -(uL - self.cx) / Z_clip  # ∂uL/∂Z
        J[:, :, 1, 0] = 0.0          # ∂vL/∂X
        J[:, :, 1, 1] = fyZ          # ∂vL/∂Y
        J[:, :, 1, 2] = -(vL - self.cy) / Z_clip  # ∂vL/∂Z
        J[:, :, 2, 0] = fxZ          # ∂uR/∂X
        J[:, :, 2, 1] = 0.0          # ∂uR/∂Y
        J[:, :, 2, 2] = -(uR - self.cx) / Z_clip  # ∂uR/∂Z
        J[:, :, 3, 0] = 0.0          # ∂vR/∂X
        J[:, :, 3, 1] = fyZ          # ∂vR/∂Y
        J[:, :, 3, 2] = -(vL - self.cy) / Z_clip  # ∂vR/∂Z (= ∂vL/∂Z)

        return J

    # ---- 逐像素相对正则系数 ----
    @staticmethod
    def relative_lam(J, eps=1e-3):
        """
        逐像素 λ_i = eps * ‖J_i‖_F

        ‖J_i‖_F = sqrt(Σ σ²(J_i)) 是 J_i 局部尺度的一个上界,
        用它做基准的 λ_i 与深度/焦距无关 (尺度不变):
        - 良态像素: λ_i² 相对 σ² 很小 → 几乎无偏
        - 病态像素: λ_i² 仍能把增益上限压到 1/(eps·σ_max), 防止噪声爆炸

        Args:
            J:   (H, W, 4, 3) Jacobian
            eps: 相对系数

        Returns:
            lam_map: (H, W) float32
        """
        Jfro2 = np.sum(J * J, axis=(2, 3))  # (H, W) = ‖J‖_F²
        return (eps * np.sqrt(Jfro2)).astype(np.float32)

    # ---- 批量 3×3 线性求解 (闭式逆, 全向量化) ----
    @staticmethod
    def _solve3x3(A, b):
        """批量求解 A x = b, A:(N,3,3), b:(N,3,1) → x:(N,3,1)

        用伴随矩阵/行列式闭式求逆, 一次向量化算完 N 个 3×3 系统,
        避免 np.linalg.solve 对 (N,3,3) 堆叠在 Python 里逐矩阵循环。
        A 是正定(含正则), det>0, 无需退化分支。
        """
        a00 = A[:, 0, 0]; a01 = A[:, 0, 1]; a02 = A[:, 0, 2]
        a10 = A[:, 1, 0]; a11 = A[:, 1, 1]; a12 = A[:, 1, 2]
        a20 = A[:, 2, 0]; a21 = A[:, 2, 1]; a22 = A[:, 2, 2]

        # 余子式
        c00 = a11 * a22 - a12 * a21
        c01 = a12 * a20 - a10 * a22
        c02 = a10 * a21 - a11 * a20
        c10 = a02 * a21 - a01 * a22
        c11 = a00 * a22 - a02 * a20
        c12 = a01 * a20 - a00 * a21
        c20 = a01 * a12 - a02 * a11
        c21 = a02 * a10 - a00 * a12
        c22 = a00 * a11 - a01 * a10

        det = a00 * c00 + a01 * c01 + a02 * c02
        det = np.where(np.abs(det) < 1e-12, 1e-12, det)

        b0 = b[:, 0, 0]; b1 = b[:, 1, 0]; b2 = b[:, 2, 0]
        x0 = (c00 * b0 + c10 * b1 + c20 * b2) / det
        x1 = (c01 * b0 + c11 * b1 + c21 * b2) / det
        x2 = (c02 * b0 + c12 * b1 + c22 * b2) / det
        return np.stack([x0, x1, x2], axis=-1)[..., None]

    # ---- Willert 反演 (torch GPU 版) ----
    def reconstruct(self, flow_L, flow_R, disp, mask_flow_L=None):
        """
        Args:
            flow_L: (1, 2, H, W) tensor (GPU), 左图时间流
            flow_R: (1, 2, H, W) tensor (GPU), 右图时间流
            disp:   (H, W) numpy, 视差
            mask_flow_L: (H, W) numpy, 可选左流有效 mask

        Returns:
            U, V, W: (H, W) numpy, 校正坐标系下速度 (m/frame)
            mask:    (H, W) bool, 有效点
        """
        H, W = disp.shape
        dev = flow_L.device
        dt = torch.float32

        # 常量 → torch
        Q = torch.as_tensor(self.Q, dtype=dt, device=dev)            # (4,4)
        R_rect2world = torch.as_tensor(
            self.R1_cam.T @ self.R1_rect.T, dtype=dt, device=dev)   # (3,3)
        baseline = 1.0 / float(self.Q[3, 2]) if self.Q[3, 2] != 0 else 1.0
        disp_t = torch.as_tensor(disp, dtype=dt, device=dev)         # (H,W)

        # 1. 三角测量 (torch): XYZW = [u,v,d,1] @ Qᵀ
        gy, gx = torch.meshgrid(
            torch.arange(H, device=dev, dtype=dt),
            torch.arange(W, device=dev, dtype=dt), indexing='ij')
        pts = torch.stack([gx, gy, disp_t, torch.ones_like(disp_t)], dim=-1)  # (H,W,4)
        XYZW = pts.reshape(-1, 4) @ Q.T                                 # (N,4)
        XYZW = XYZW.reshape(H, W, 4)
        X = XYZW[..., 0] / XYZW[..., 3]
        Y = XYZW[..., 1] / XYZW[..., 3]
        Z = XYZW[..., 2] / XYZW[..., 3]

        # 2. 采样右图时间流 (torch, 已在 GPU)
        flow_R_s = self.sample_flow_R(flow_R, disp)[0]   # (2,H,W)
        flow_L_t = flow_L[0]                             # (2,H,W)

        # 3. Jacobian (torch)
        Z_clip = Z.clamp(min=1e-6)
        uL = self.fx * X / Z_clip + self.cx
        vL = self.fy * Y / Z_clip + self.cy
        uR = uL - self.fx * baseline / Z_clip
        fxZ = self.fx / Z_clip
        fyZ = self.fy / Z_clip
        J = torch.zeros(H, W, 4, 3, dtype=dt, device=dev)
        J[..., 0, 0] = fxZ;                    J[..., 0, 2] = -(uL - self.cx) / Z_clip
        J[..., 1, 1] = fyZ;                    J[..., 1, 2] = -(vL - self.cy) / Z_clip
        J[..., 2, 0] = fxZ;                    J[..., 2, 2] = -(uR - self.cx) / Z_clip
        J[..., 3, 1] = fyZ;                    J[..., 3, 2] = -(vL - self.cy) / Z_clip

        # 4. 位移向量 d = [uL, vL, uR, vR]
        d = torch.stack([flow_L_t[0], flow_L_t[1], flow_R_s[0], flow_R_s[1]], dim=-1)  # (H,W,4)

        # 5. 逐像素 Willert: V = (JᵀJ + λ²I)⁻¹ Jᵀ d
        Jf = J.reshape(-1, 4, 3)
        df = d.reshape(-1, 4, 1)
        JtJ = Jf.transpose(1, 2) @ Jf      # (N,3,3)
        Jtd = Jf.transpose(1, 2) @ df      # (N,3,1)
        eye3 = torch.eye(3, dtype=dt, device=dev)

        if self.per_pixel and self.lam > 0:
            lam_map = self.lam * (J * J).sum(dim=(2, 3)).sqrt()   # (H,W) = ε·‖J‖_F
            reg = (lam_map.reshape(-1) ** 2)[:, None, None] * eye3
            Vf = torch.linalg.solve(JtJ + reg, Jtd)
        elif self.lam > 0:
            reg = (self.lam ** 2) * eye3
            Vf = torch.linalg.solve(JtJ + reg, Jtd)
        else:
            reg = 1e-6 * eye3
            Vf = torch.linalg.solve(JtJ + reg, Jtd)

        V = Vf.reshape(H, W, 3)

        # 6. 旋转回世界坐标 (torch)
        vec_world = (R_rect2world @ V.reshape(-1, 3).T).T   # (N,3)
        U_world = vec_world[:, 0].reshape(H, W)
        V_world = vec_world[:, 1].reshape(H, W)
        W_world = vec_world[:, 2].reshape(H, W)

        # 7. mask: Z>0, disp有效, 数值非nan
        mask = (disp_t > 0) & torch.isfinite(Z) & (Z > 0) & (Z < 1e4)
        if mask_flow_L is not None:
            mask = mask & torch.as_tensor(mask_flow_L, dtype=torch.bool, device=dev)
        mask = mask & torch.isfinite(V).all(dim=-1)



        U_np = U_world.cpu().numpy()
        V_np = V_world.cpu().numpy()
        W_np = W_world.cpu().numpy()
        mask_np = mask.cpu().numpy()
        Z_np = Z.cpu().numpy()
        speed = np.sqrt(U_np**2 + V_np**2 + W_np**2)
        flow_L_np = flow_L[0].permute(1,2,0).cpu().numpy()   # 如果还没取

        return (U_world.cpu().numpy(), V_world.cpu().numpy(),
                W_world.cpu().numpy(), mask.cpu().numpy())

    def _rotate_to_world(self, U_rect, V_rect, W_rect):
        """
        校正坐标系 → 世界坐标系
        校正坐标系原点=左相机光心, 轴对齐
        V_world = R1^T @ V_rect

        旋转矩阵都是正交矩阵，逆矩阵等于转置矩阵.
        """
        H, W = U_rect.shape
        vec_rect = np.stack([U_rect.ravel(), V_rect.ravel(), W_rect.ravel()], axis=1)  # (N, 3)
        # 校正 → 相机1 → 世界: R1_rect.T 先回相机1, R1_cam.T 再转世界
        R_rect2world = self.R1_cam.T @ self.R1_rect.T
        vec_world = (R_rect2world @ vec_rect.T).T  # (N, 3)
        U_w = vec_world[:, 0].reshape(H, W)
        V_w = vec_world[:, 1].reshape(H, W)
        W_w = vec_world[:, 2].reshape(H, W)
        return U_w, V_w, W_w


# ===================================================================
# Part C: 图像加载 + RAFT 推理
# ===================================================================

def load_image(filepath):
    """加载单张图像 → (1, 3, H, W) tensor float [0,255]"""
    img = np.array(Image.open(filepath)).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=2)
    img = img[:, :, :3]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def load_image_numpy(filepath):
    """加载为 numpy uint8 (用于 cv2.remap)"""
    img = np.array(Image.open(filepath)).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=2)
    return img[:, :, :3]


def run_raft(model, img1, img2, iters=20):
    """RAFT 光流推理, 带 pad"""
    padder = InputPadder(img1.shape)
    img1_p, img2_p = padder.pad(img1, img2)
    _, flow_up = model(img1_p, img2_p, iters=iters, test_mode=True)
    return flow_up, padder


def run_raft_stereo(model, imgL, imgR, iters=20):
    """RAFTStereo 推理"""
    padder = InputPadder(imgL.shape)
    imgL_p, imgR_p = padder.pad(imgL, imgR)
    disp = model(imgL_p, imgR_p, iters=iters, test_mode=True)
    flow_up = disp[-1] if isinstance(disp, (list, tuple)) else disp
    return flow_up, padder


def load_flow_model(ckpt_path, small=False, mixed_precision=False, alternate_corr=False):
    """加载 RAFT 光流模型"""
    from argparse import Namespace
    args = Namespace(
        small=small, mixed_precision=mixed_precision, alternate_corr=alternate_corr
    )
    model = torch.nn.DataParallel(RAFT(args))
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    model = model.module
    model.to(DEVICE)
    model.eval()
    print(f"[Model] RAFT flow loaded: {ckpt_path}")
    return model


def load_stereo_model(ckpt_path, mixed_precision=False):
    """加载 RAFTStereo 模型"""
    from argparse import Namespace
    args = Namespace(
        hidden_dims=[128, 128, 128],
        corr_implementation="reg",
        shared_backbone=True,
        corr_levels=4,
        corr_radius=4,
        n_downsample=3,
        context_norm="batch",
        slow_fast_gru=True,
        n_gru_layers=2,
        mixed_precision=mixed_precision,
    )
    model = RAFTStereo(args)
    state = torch.load(ckpt_path, map_location=DEVICE)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()
    model.to(DEVICE)
    model.eval()
    print(f"[Model] RAFTStereo loaded: {ckpt_path}")
    return model


# ===================================================================
# Part D: 可视化
# ===================================================================

def visualize_3c(U, V, W, mask, save_path, title="Per-Pixel Willert 2D3C"):
    """三格 diverging colormap"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    data = [
        (U, 'U (X world)'),
        (V, 'V (Y world)'),
        (W, 'W (Z world)'),
    ]
    for i, (d, label) in enumerate(data):
        d_masked = np.where(mask, d, np.nan)
        vmax = np.nanpercentile(np.abs(d_masked), 98) if mask.any() else 1e-6
        vmax = max(vmax, 1e-6)
        im = axes[i].imshow(d_masked, cmap=plt.cm.RdBu_r, origin='upper',
                            vmin=-vmax, vmax=vmax)
        axes[i].set_title(f'{label} [m/frame]')
        plt.colorbar(im, ax=axes[i], shrink=0.82)
    fig.suptitle(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Viz] {save_path}")


# ===================================================================
# Part E: 主函数
# ===================================================================

def parse_args():
    p = argparse.ArgumentParser(description='Per-Pixel Stereo Willert 2D3C')
    # p.add_argument('--calib', default=r'D:\Stereo\Cam_v1.npz', help='标定 npz')
    p.add_argument('--calib',default=r"D:\StereoData\training\cam_par\Cam_12.npz")
    # p.add_argument('--left_path', default=r"D:\Stereo1\blender_img_v1\left", help='左相机图像文件夹')
    p.add_argument('--left_path', default=r"D:\StereoData\training\left")

    # p.add_argument('--right_path', default=r"D:\Stereo1\blender_img_v1\right", help='右相机图像文件夹')
    p.add_argument('--right_path', default=r"D:\StereoData\training\right", help='右相机图像文件夹')
    # p.add_argument('--flow_model', default=r'C:/Users\Lenovo\Desktop\RAFT_master/0_raft.pth', help='RAFT 光流权重')
    p.add_argument('--flow_model', default=r"D:\StereoData\training\1w5_pureraft.pth", help='RAFT 光流权重')
    p.add_argument('--stereo_model', default=r"D:\3D 重建\raftstereo-realtime.pth", help='RAFTStereo 权重')
    p.add_argument('--output', default=r'D:\RAFT_with DCN_and_CBAM\Results\results_per_pixel', help='输出目录')
    p.add_argument('--rect_cache', default='./rect_maps_v2.npz', help='校正映射缓存')
    p.add_argument('--start_frame', type=int, default=1)
    p.add_argument('--end_frame', type=int, default=50)
    p.add_argument('--iters_flow', type=int, default=20)
    p.add_argument('--iters_stereo', type=int, default=20)
    p.add_argument('--fps', type=float, default=30.0)
    p.add_argument('--lam', type=float, default=0.0, help='Tikhonov 正则系数 (0=普通伪逆)')
    p.add_argument('--lam_per_pixel', action='store_true',
                   help='逐像素正则: λ_i = lam * ‖J_i‖_F (lam 作为相对系数 ε)')
    p.add_argument('--no_display', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    # ---- 加载标定 ----
    calib = np.load(args.calib)
    K1, K2 = calib['K1'], calib['K2']
    R1_cam = calib['R1']   # 世界→Cam1
    t1_cam = calib['t1'].reshape(3, 1) if calib['t1'].ndim == 1 else calib['t1']
    R2_cam = calib['R2']
    t2_cam = calib['t2'].reshape(3, 1) if calib['t2'].ndim == 1 else calib['t2']

    # cv2.stereoCalibrate 格式: P_cam2 = R @ P_cam1 + T
    # P_cam1 = R1 @ P_w + t1 → P_w = R1^T @ (P_cam1 - t1)
    # P_cam2 = R2 @ R1^T @ (P_cam1 - t1) + t2
    #        = (R2 @ R1^T) @ P_cam1 + (t2 - R2 @ R1^T @ t1)
    R_stereo = R2_cam @ R1_cam.T
    T_stereo = t2_cam - R_stereo @ t1_cam

    # ---- 加载第一帧确定图像尺寸 ----
    left_files = sorted(glob.glob(os.path.join(args.left_path, '*.png')) +
                        glob.glob(os.path.join(args.left_path, '*.jpg')))
    if not left_files:
        raise FileNotFoundError(f"没找到左图: {args.left_path}")
    img_sample = np.array(Image.open(left_files[0]))
    img_h, img_w = img_sample.shape[:2]
    print(f"[Init] 图像: {img_w}×{img_h}")

    # ---- 极线校正 (缓存) ----
    if os.path.exists(args.rect_cache):
        rectifier = StereoRectifier.load(args.rect_cache)
    else:
        rectifier = StereoRectifier(K1, K2, R_stereo, T_stereo, img_w, img_h,1)
        # rectifier.Q[3,2]=-rectifier.Q[3,2]
        rectifier.save(args.rect_cache)
    # print(rectifier.Q)
    
    # print(rectifier.Q)
    # ---- 加载模型 ----
    model_flow = load_flow_model(args.flow_model)
    model_stereo = load_stereo_model(args.stereo_model)

    # ---- 逐像素 Willert ----
    willert = PerPixelWillert(rectifier, R1_cam, lam=args.lam,
                              per_pixel=args.lam_per_pixel)

    # ---- 图像列表 ----
    right_path = args.right_path if os.path.isdir(args.right_path) else args.left_path
    right_files = sorted(glob.glob(os.path.join(right_path, '*.png')) +
                         glob.glob(os.path.join(right_path, '*.jpg')))

    if right_path == args.left_path:
        left_files = [f for f in left_files if 'cam0' in os.path.basename(f).lower()
                      or '_L' in os.path.basename(f)]
        right_files = [f for f in right_files if 'cam1' in os.path.basename(f).lower()
                       or '_R' in os.path.basename(f)]

    assert len(left_files) == len(right_files), \
        f"左右文件数不一致: {len(left_files)} vs {len(right_files)}"

    end_frame = args.end_frame or (len(left_files) - 1)
    n_frames = min(end_frame - args.start_frame, len(left_files) - args.start_frame)

    print(f"[Init] {n_frames} 帧待处理")

    total_time = 0.0
    for idx in range(args.start_frame - 1, args.start_frame - 1 + n_frames):
        t0 = time.time()

        # -- 加载 + 校正 --
        rawL_t = load_image_numpy(left_files[idx])
        rawL_t1 = load_image_numpy(left_files[idx + 1])
        rawR_t = load_image_numpy(right_files[idx])
        rawR_t1 = load_image_numpy(right_files[idx + 1])

        rectL_t, rectR_t = rectifier.rectify_pair(rawL_t, rawR_t)
        rectL_t1 = rectifier.rectify_single(rawL_t1, 'L')
        rectR_t1 = rectifier.rectify_single(rawR_t1, 'R')

        # -- 转 tensor --
        def np2tensor(img):
            t = torch.from_numpy(img).permute(2, 0, 1).float()
            return t[None].to(DEVICE)

        with torch.no_grad():
            # 三路模型
            flow_L, padderL = run_raft(model_flow,
                                       np2tensor(rectL_t), np2tensor(rectL_t1),
                                       iters=args.iters_flow)

            flow_R, padderR = run_raft(model_flow,
                                       np2tensor(rectR_t), np2tensor(rectR_t1),
                                       iters=args.iters_flow)

            disp, padderS = run_raft_stereo(model_stereo,
                                            np2tensor(rectL_t), np2tensor(rectR_t),
                                            iters=args.iters_stereo)

            # Unpad
            flow_L = padderL.unpad(flow_L)  # (1, 2, H_rect, W_rect)
            flow_R = padderR.unpad(flow_R)
            disp = padderS.unpad(disp.squeeze(1))  # (H_rect, W_rect)

        disp_np = -(disp.squeeze().cpu().numpy())
        

        # -- 单位转换: pixel/frame → m/frame --
        # 像素位移 / fx * Z → 世界位移
        # 此处 flow 仍是 pixel/frame, Jacobian 自动转换单位
        # 输出 U,V,W = pinv(J) @ [pixel位移] → 世界米/frame (J 内含 pixel/m 转换)

        # -- Willert --
        U, V, W, mask = willert.reconstruct(flow_L, flow_R, disp_np)
        dt = time.time() - t0
        total_time += dt

        n_valid = mask.sum()
        n_total = mask.size
        print(f"[{idx+1:04d}] U={np.mean(U[mask]):.4f}±{np.std(U[mask]):.4f}, "
              f"V={np.mean(V[mask]):.4f}±{np.std(V[mask]):.4f}, "
              f"W={np.mean(W[mask]):.4f}±{np.std(W[mask]):.4f} m/frame, "
              f"valid={n_valid}/{n_total} ({100.*n_valid/n_total:.0f}%), "
              f"t={dt:.2f}s")

        # -- 保存 --
        out_npz = os.path.join(args.output, f'velocity_{idx+1:04d}.npz')
        np.savez_compressed(out_npz, U=U, V=V, W=W, mask=mask)

        # -- 可视化 --
        if idx % 10 == 0:
            viz_path = os.path.join(args.output, f'viz_{idx+1:04d}.png')
            visualize_3c(U, V, W, mask, viz_path)

    avg = total_time / max(n_frames, 1)
    print(f"\n[Done] {n_frames} 帧, 平均 {avg:.2f}s/帧 ({1/avg:.1f} FPS)")


if __name__ == '__main__':
    main()

