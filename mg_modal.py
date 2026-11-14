# -*- coding: utf-8 -*-
"""
模态波扩展 MG 模型 (四状态) —— P2-8/P2-9

动机: MG 一阶 Galerkin 模型的失稳前兆只有失速团幅值 J 本身 (t_pre 与失稳
几乎重合, 闭环避喘 0/6 归因之一)。文献共识 (Garnier/Camp/Day): 压缩系统
模态波 (rotating modal wave, ~0.5×f_rotor) 在特性线峰值到达**之前**就开始
增长, 是 spike/stall 成核的前兆。本扩展在 MG 三状态上叠加:

  dA/dxi = A * [ sigma_m*(x_m - x) - k_J*J ] - A^3/A_sat^2 + kappa*A^2*max(1-x,0)

其中 A 为模态波幅值 (sqrt 量纲), x = Phi/W - 1:
  - x_m = 1.10 > 1: 模态波失稳边界在特性线峰值 (x=1, Phi=2W) **之前**
    —— 节流尚在正斜率段时模态波已开始增长 (modal-inception-before-peak)
  - k_J*J: 失速团形成后模态波能量被吸收 (模态波 -> 失速团转变)
  - A^3/A_sat^2: 饱和 (A_sat = 0.05, sqrt 量纲)
  - kappa*A^2*max(1-x,0): 模态波对失速团的成核 seeding (仅在越过峰值后
    生效, 物理上模态波诱发 spike/stall nucleation)
  - A 地板 a_floor=1e-4 (湍流背景扰动, 同 j_floor 机制)

信号合成: p_modal = A_M * A * cos(2*pi*0.5*f_rotor*t + phase)
  0.5×f_rotor 落在检测器失速带 (0.25-0.625×f_rotor) 内 —— 归一化设计
  无需改动即可捕获模态波前兆 (这是跨模型泛化的又一验证)。
"""
import numpy as np

# 模态波参数 (TUDa 校准版特性线 CALIB 之下整定)
MODAL_PARAMS = dict(
    x_m=1.18,        # 模态波失稳边界 (x=1 为特性线峰值) —— 由 TUDa 实测
                     # 前兆提前量 (0.72-0.92s, 见 precursor_tf.py) 标定:
                     # x_m 敏感性扫描 {1.10,1.15,1.18,1.21} -> lead
                     # {0.26,0.57,0.93,1.27}s, 取与实测一致点
    sigma_m=1.2,     # 模态波增长率尺度 [1/xi]
    A_sat=0.05,      # 模态波饱和幅值 (sqrt 量纲)
    a_floor=1e-4,    # 湍流背景地板
    k_J=5.0,         # 失速团对模态波的吸收系数
    kappa=0.5,       # 模态波 -> 失速团成核 seeding 系数
    A_gt=0.01,       # 模态波 ground truth 阈值 (事件: modal inception)
)
A_M = 800.0        # 模态波压力幅值标度 [Pa] (A=0.0125 -> RMS≈7Pa > 噪声)
F_MODAL_FRAC = 0.5  # 模态波旋转速度 = 0.5 × f_rotor


def modal_deriv(y, B, gammaT, p, modal=None):
    """四状态导数 y = (Phi, Psi, J, A), 标量 (闭环逐步积分用)"""
    Phi, Psi, J, A = y
    x = Phi / p["W"] - 1.0
    psi_c = p["psi_c0"] + p["H"] * (1.0 + 1.5 * x - 0.5 * x ** 3)
    PhiT = gammaT * np.sqrt(max(Psi, 1e-9))
    rho = 3.0 * p["a"] * p["H"] / ((1.0 + p["m"] * p["a"]) * p["W"])

    dPhi = (psi_c - Psi - 0.75 * p["H"] * x * J) / p["lc"]
    dPsi = (Phi - PhiT) / (4.0 * B ** 2 * p["lc"])
    dJ = J * rho * (1.0 - x ** 2 - J / 4.0)
    mp = modal or MODAL_PARAMS
    dA = A * (mp["sigma_m"] * (mp["x_m"] - x) - mp["k_J"] * J) \
        - A ** 3 / mp["A_sat"] ** 2
    dJ += mp["kappa"] * A * A * max(1.0 - x, 0.0)   # 模态波成核 seeding
    return np.array([dPhi, dPsi, dJ, dA])
