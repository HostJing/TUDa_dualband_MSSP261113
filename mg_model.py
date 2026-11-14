# -*- coding: utf-8 -*-
"""
Moore-Greitzer 非线性喘振/旋转失速模型（三状态 Galerkin 降阶形式）

参考:
  Moore & Greitzer (1986), A theory of post-stall transients in axial compression systems
  Gravdahl & Egeland (1999), Compressor Surge and Stall  (eq. 2.1-2.4)
  AI-Based Detection of Surge and Rotating Stall in Axial Compressors (eq. 21)

状态:
  Phi : 环向平均流量系数 (无量纲)
  Psi : 集气室压力系数 (无量纲)
  J   : 旋转失速幅值平方 (一阶模态)

方程 (无量纲时间 xi):
  dPhi/dxi = (1/lc) * ( psi_c(Phi) - Psi - (3/4)*H*(Phi/W - 1)*J )
  dPsi/dxi = (Phi - PhiT(Psi)) / (4 * B^2 * lc)
  dJ/dxi   = J * rho * ( 1 - (Phi/W - 1)^2 - J/4 ),  rho = 3aH/((1+ma)W)

压气机立方特性线:
  psi_c(Phi) = psi_c0 + H * [ 1 + 1.5*x - 0.5*x^3 ],  x = Phi/W - 1

节气门特性:
  PhiT(Psi) = gammaT * sqrt(Psi)
"""

import numpy as np


# ---------------------------------------------------------------------------
# 默认参数 (无量纲, 参考上述文献典型取值)
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = dict(
    psi_c0=0.30,   # 特性线截距 (关机压比)
    H=0.18,        # 特性线半高
    W=0.25,        # 特性线半宽
    lc=4.0,        # 系统等效无量纲长度 lc = lI + 1/a + lE
    a=0.30,        # 叶排时间滞后参数倒数
    m=1.75,        # 叶排滞后修正参数
    B=1.0,         # Greitzer B 参数 (几何/转速决定喘振类型)
)


def compressor_char(phi, psi_c0=0.30, H=0.18, W=0.25):
    """立方压气机特性线 psi_c(phi)"""
    x = phi / W - 1.0
    return psi_c0 + H * (1.0 + 1.5 * x - 0.5 * x ** 3)


def char_peak_flow(W=0.25):
    """特性线峰值点流量 (稳定边界): x=1 -> Phi = 2W"""
    return 2.0 * W


def rho_growth(a=0.30, H=0.18, m=1.75, W=0.25):
    """J 方程增长率因子 rho = 3aH/((1+ma)W)"""
    return 3.0 * a * H / ((1.0 + m * a) * W)


# ---------------------------------------------------------------------------
# 向量化 RK4 积分器 (同时积分多个工况)
# ---------------------------------------------------------------------------
def mg_deriv(y, B, gammaT, p, dist=None, seed=None):
    """y = (Phi, Psi, J), 参数均为数组 (按工况广播)

    dist: 进口总压畸变亏缺 (无量纲, 数组, 可为 None) —— 从特性线扣除,
          模拟进气畸变/压力 deficit, 降低有效稳定裕度
    seed : J 方程的扰动注入 (数组, 可为 None) —— 模拟湍流/畸变诱发的
          失速团随机成核
    """
    Phi, Psi, J = y[0], y[1], y[2]
    H, W, lc, a, m, psi_c0 = p["H"], p["W"], p["lc"], p["a"], p["m"], p["psi_c0"]
    x = Phi / W - 1.0
    psi_c = psi_c0 + H * (1.0 + 1.5 * x - 0.5 * x ** 3)
    if dist is not None:
        psi_c = psi_c - dist
    PhiT = gammaT * np.sqrt(np.maximum(Psi, 1e-9))

    dPhi = (psi_c - Psi - 0.75 * H * x * J) / lc
    dPsi = (Phi - PhiT) / (4.0 * B ** 2 * lc)
    rho = 3.0 * a * H / ((1.0 + m * a) * W)
    dJ = J * rho * (1.0 - x ** 2 - J / 4.0)
    if seed is not None:
        dJ = dJ + seed
    dJ = np.where(J <= 0.0, np.maximum(dJ, 0.0), dJ)  # J 非负 (有 seed 时允许从 0 生长)
    return np.stack([dPhi, dPsi, dJ])


def integrate(params_list, gammaT_sched, xi_end, dt=2e-3, J0=1e-8,
              dist_fn=None, seed_fn=None, phi0=None, j_floor=0.0):
    """
    向量化积分多工况 MG 模型.

    params_list : list[dict], 每个工况的参数 (B 可不同)
    gammaT_sched: callable(xi, idx) -> array(n_cases) 节气门开度时间表
    dist_fn     : callable(xi) -> array(n_cases) 进口畸变亏缺 (可选)
    seed_fn     : callable(xi, rng_state?) -> array(n_cases) J 扰动注入 (可选;
                  为保持 RK4 确定性, 由调用方预先离散化, 此处按 xi 查表)
    phi0        : 初始平衡流量系数 (默认 None -> char_peak_flow(W0)*1.08,
                  向后兼容; 校准版特性线 W 不同时应显式给出)
    j_floor     : J 数值地板 (默认 0.0 = 旧行为). 稳定段 J 指数衰减可能深达
                  1e-200+ (双精度未下溢但恢复需上千 xi), 物理上湍流维持着
                  背景扰动底噪; 设 j_floor=1e-12 保证节流越过峰值后失速
                  模态能在合理时间内成核. 校准版 (H/W 大, 衰减快) 必须开启.
    返回:
      xi   : (T,) 时间网格
      Y    : (T, 3, n_cases) 状态轨迹
      meta : 参数快照
    """
    n = len(params_list)
    meta = {k: np.array([pl[k] for pl in params_list]) for k in DEFAULT_PARAMS}
    gammaT0 = np.array([pl.get("gammaT", params_list[0].get("gammaT", 0.45))
                        for pl in params_list])

    y = np.zeros((3, n))
    # 初始状态: 由初始节气门求稳态 (J=0 特性线上取点)
    Phi_eq = phi0 if phi0 is not None else char_peak_flow(meta["W"][0]) * 1.08
    Psi_eq = compressor_char(Phi_eq, psi_c0=meta["psi_c0"][0], H=meta["H"][0], W=meta["W"][0])
    y[0, :] = Phi_eq
    y[1, :] = Psi_eq
    y[2, :] = J0

    steps = int(xi_end / dt) + 1
    xi_grid = np.linspace(0.0, xi_end, steps)
    Y = np.zeros((steps, 3, n))
    Y[0] = y

    p = meta
    for i in range(steps - 1):
        xi = xi_grid[i]
        gT = gammaT_sched(xi, gammaT0)
        dist = dist_fn(xi) if dist_fn is not None else None
        sd = seed_fn[i] if seed_fn is not None else None
        k1 = mg_deriv(y, p["B"], gT, p, dist=dist, seed=sd)
        k2 = mg_deriv(y + 0.5 * dt * k1, p["B"], gT, p, dist=dist, seed=sd)
        k3 = mg_deriv(y + 0.5 * dt * k2, p["B"], gT, p, dist=dist, seed=sd)
        k4 = mg_deriv(y + dt * k3, p["B"], gT, p, dist=dist, seed=sd)
        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if j_floor > 0.0:
            y[2] = np.clip(y[2], j_floor, None)   # 湍流背景扰动地板
        else:
            y[2] = np.maximum(y[2], 0.0)          # J >= 0 (旧行为)
        y[0] = np.maximum(y[0], -0.35) # 深喘振允许适度倒流, 数值限幅
        Y[i + 1] = y
    return xi_grid, Y, meta


# ---------------------------------------------------------------------------
# 信号综合: 模拟机匣壁面动态压力传感器信号
# ---------------------------------------------------------------------------
def synthesize_signal(xi_grid, Y, meta, rng, fs_phys=None, xi_per_sec=50.0,
                      f_rotor=400.0, stall_frac=0.45, K_p=10000.0,
                      A_rs=350.0, noise_std=5.0, seed_burst=True):
    """
    由 (Phi, Psi, J) 轨迹合成机匣壁面动态压力信号 p(t).

    物理映射: t = xi / xi_per_sec (秒)
      - 喘振低频大幅振荡 -> Psi 直接缩放 (K_p Pa/单位)
      - 失速团/模态波     -> sqrt(J) * cos(2*pi*f_rs*t), f_rs = stall_frac * f_rotor
      - 噪声 + 偶发干扰脉冲
    返回 dict(t, p, Psi, Phi, J)
    """
    T = len(xi_grid)
    t = xi_grid / xi_per_sec
    Psi, Phi, J = Y[:, 1, :], Y[:, 0, :], Y[:, 2, :]

    # 压力基准: 稳态特性线值作为均值漂移
    psi_ref = compressor_char(np.full_like(Psi, 0.56),
                              psi_c0=meta["psi_c0"][0], H=meta["H"][0], W=meta["W"][0])
    p_mean = (Psi - 0.0) * K_p  # 直接取集气室压力系数映射

    # 失速团旋转扰动 (各工况随机相位)
    f_rs = stall_frac * f_rotor
    n_cases = J.shape[1]
    phase = rng.uniform(0, 2 * np.pi, size=n_cases)
    p_rs = A_rs * np.sqrt(np.maximum(J, 0.0)) * np.cos(
        2 * np.pi * f_rs * t[:, None] + phase[None, :])

    p = p_mean + p_rs + rng.normal(0.0, noise_std, size=(T, n_cases))

    # 偶发电磁干扰脉冲 (模拟真实试验环境)
    if seed_burst:
        for c in range(n_cases):
            n_burst = rng.integers(0, 4)
            for _ in range(n_burst):
                b0 = rng.integers(0, max(T - 400, 1))
                blen = rng.integers(20, 200)
                amp = rng.uniform(30, 120) * rng.choice([-1, 1])
                decay = np.exp(-np.linspace(0, 6, blen))
                p[b0:b0 + blen, c] += amp * decay * np.sin(
                    2 * np.pi * 900 * t[b0:b0 + blen])
    return dict(t=t, p=p, Psi=Psi, Phi=Phi, J=J)


# ---------------------------------------------------------------------------
# 工况时间表: 缓慢关小节气门 (瞬态节流过程, 类似真实喘振试验)
# ---------------------------------------------------------------------------
def make_throttle_schedule(gammaT_start, gammaT_end, xi_ramp_start=10.0,
                           xi_ramp_end=90.0, hold_after=None, smooth=True):
    """
    返回 callable(xi, gammaT0) -> gammaT(xi):
    在 [xi_ramp_start, xi_ramp_end] 间由 start 线性关小到 end, 其余保持.
    smooth=True 时用 cosine smoothstep (二阶导连续), 避免合成信号在
    全段 FFT 带通滤波时因节流趋势阶跃产生时域泄漏振铃。
    """
    def sched(xi, gammaT0):
        g = np.full_like(gammaT0, gammaT_start, dtype=float)
        if hold_after is not None and xi >= xi_ramp_end:
            return np.full_like(gammaT0, gammaT_end, dtype=float)
        if xi >= xi_ramp_start:
            frac = (xi - xi_ramp_start) / max(xi_ramp_end - xi_ramp_start, 1e-9)
            frac = min(frac, 1.0)
            if smooth:
                frac = 0.5 * (1.0 - np.cos(np.pi * frac))
            return gammaT_start + frac * (gammaT_end - gammaT_start)
        return g
    return sched


# ---------------------------------------------------------------------------
# 事件真值 (ground truth) 计算
# ---------------------------------------------------------------------------
def event_ground_truth(xi, Y, meta, params_list,
                       J_precursor=1e-3, J_stall=0.10, psi_osc_frac=0.15):
    """
    按工况逐个计算事件时刻 (xi 域):
      pre_stall : J 首次超过 J_precursor (失速先兆出现)
      stall     : J 首次超过 J_stall (旋转失速充分发展)
      surge     : Phi 首次倒流 (<0, 深喘振判据) 或 Psi 滑动 std
                  (1 个无量纲时间窗) 超过 psi_osc_frac*H 且持续
    返回 list[dict(t_pre, t_stall, t_surge, label)]
    """
    events = []
    n = Y.shape[2]
    H = meta["H"][0]
    thr = psi_osc_frac * H
    win = 2000  # 滑动窗口步数 (~1 个无量纲时间单位)
    for c in range(n):
        Phi = Y[:, 0, c]
        Psi = Y[:, 1, c]
        J = Y[:, 2, c]
        # 向量化滑动 std (累积和法)
        k = np.arange(1, len(Psi) + 1)
        lo = np.maximum(0, k - win)
        c1 = np.concatenate([[0.0], np.cumsum(Psi)])
        c2 = np.concatenate([[0.0], np.cumsum(Psi ** 2)])
        m1 = (c1[k] - c1[lo]) / (k - lo)
        m2 = (c2[k] - c2[lo]) / (k - lo)
        std_w = np.sqrt(np.maximum(m2 - m1 ** 2, 0.0))
        idx_pre = np.argmax(J > J_precursor) if np.any(J > J_precursor) else -1
        idx_stall = np.argmax(J > J_stall) if np.any(J > J_stall) else -1
        surge_mask = (std_w > thr) | (Phi < 0.0)
        # 要求连续 3 个窗口超阈值
        sm = surge_mask.astype(int)
        run = np.convolve(sm, np.ones(3, dtype=int), mode="valid")
        idx_surge = np.argmax(run >= 3) + 1 if np.any(run >= 3) else -1

        B = params_list[c]["B"]
        gammaT_end = params_list[c].get("gammaT_end", None)
        if idx_surge >= 0:
            label = "surge"
        elif idx_stall >= 0:
            label = "rotating_stall"
        else:
            label = "stable"
        events.append(dict(
            case=c, B=B,
            t_pre=xi[idx_pre] if idx_pre >= 0 else np.nan,
            t_stall=xi[idx_stall] if idx_stall >= 0 else np.nan,
            t_surge=xi[idx_surge] if idx_surge >= 0 else np.nan,
            label=label,
        ))
    return events
