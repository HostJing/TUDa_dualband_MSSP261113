# -*- coding: utf-8 -*-
"""
TUDa-GLR-OpenStage 数据接入 adapter

数据结构 (官方 Open_Stage_TUDa_Data_Info.pdf p.19 + 03_Kulite/readme.txt):
  - kudatabase (T, 21): 500 kHz 机匣 Kulite 壁压, 无量纲 ps_kulite/ptref
  - 21 通道 = 叶尖机匣轴向阵列 (同一周向位置 phicor=260°,
    axcor 180.7–252.7 mm rig 坐标, 相对叶片前缘 ±27% 弦长)
  - kulite_revolutions (1478, 21, 2714) uint8: 锁相相位 bin × 转数 (stall run only)
  - speed: 百分数 (100.163 = 100.163% × N_nominal 20000 rpm)
  - massflow: ISA 校正流量 [kg/s]

用法:
  ep = read_raw_mat("Stall_Speed100.16_Kulite_data.mat")   # -> episode dict
  sanity_check(ep); res = run_detection(ep)
"""
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# TUDa 官方参数 (Shared operating points + SetupInfo)
N_NOMINAL_RPM = 20000.0     # 设计转速
N_BLADES = 16               # Rotor 1 叶片数 (BPF = 16 * f_rotor)
FS_RAW = 500000.0           # Kulite 记录采样率 (readme.txt)


# --------------------------------------------------------------------------
# 原始数据读取
# --------------------------------------------------------------------------
def read_raw_mat(path, channels=None):
    """读取 TUDa Kulite .mat -> episode dict(t, P, p, fs, f_rotor, meta).

    Args:
        path: .mat 文件路径 (稳态或 stall run)
        channels: 选择通道子集 (默认全部 21 通道)
    返回:
        episode dict; P 为 (T, K) 无量纲 ps/ptref; f_rotor 由 speed 字段换算。
    """
    import scipy.io as sio
    m = sio.loadmat(path)
    s = m["kulite_data"][0, 0]
    kd = np.asarray(s["kudatabase"], dtype=np.float64)   # (T, 21)
    if channels is not None:
        kd = kd[:, list(channels)]
    axcor = np.asarray(s["axcor"]).ravel()
    if channels is not None:
        axcor = axcor[list(channels)]
    speed_pct = float(np.asarray(s["speed"]).ravel()[0])
    massflow = float(np.asarray(s["massflow"]).ravel()[0])
    ptref = float(np.asarray(s["ptref"]).ravel()[0])
    ttref = float(np.asarray(s["ttref"]).ravel()[0])
    f_rotor = speed_pct / 100.0 * N_NOMINAL_RPM / 60.0   # Hz
    T = kd.shape[0]
    t = np.arange(T) / FS_RAW
    ep = dict(
        t=t, P=kd, p=kd[:, 0],
        fs=FS_RAW, f_rotor=f_rotor,
        meta=dict(source=os.path.basename(path),
                  speed_pct=speed_pct, massflow=massflow,
                  ptref=ptref, ttref=ttref,
                  n_sensors=kd.shape[1], axcor=axcor.tolist(),
                  is_stall_run="Stall" in os.path.basename(path),
                  note="ps/ptref dimensionless; axial array at phi=260deg"))
    return ep


def decimate(P, fs, fs_new):
    """FFT 低通抗混叠 + 整数抽取 (检测频带最高 0.625*f_rotor, 2 kHz 足够).

    返回 (P_ds, fs_new_actual)
    """
    from detector import fft_bandpass
    factor = max(int(round(fs / fs_new)), 1)
    # 抗混叠: 截到 fs_new/2 的 80%
    fc = 0.4 * fs / factor
    if P.ndim == 1:
        P = fft_bandpass(P - P.mean(), fs, 0.0, fc) + P.mean()
    else:
        P = np.column_stack([
            fft_bandpass(P[:, k] - P[:, k].mean(), fs, 0.0, fc) + P[:, k].mean()
            for k in range(P.shape[1])])
    return P[::factor], fs / factor


def estimate_f_rotor(p, fs, n_bodies=N_BLADES, f_search=(200.0, 500.0)):
    """由 BPF 谱峰独立估计转频 (与 speed 字段换算交叉验证).

    BPF = n_bodies * f_rotor 应为谱中最强叶片通过频率。
    返回 (f_rotor_est, bpf_freq, bpf_peak_ratio)
    """
    x = p - p.mean()
    n = min(len(x), 1 << 21)          # ≤2M 点 FFT
    x = x[:n]
    spec = np.abs(np.fft.rfft(x * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    bmask = (freqs >= n_bodies * f_search[0]) & (freqs <= n_bodies * f_search[1])
    fb = freqs[bmask]
    sb = spec[bmask]
    i = int(np.argmax(sb))
    bpf = fb[i]
    # 峰显著性: BPF 邻域峰 vs 全频段中位谱
    sig = sb[i] / (np.median(spec) + 1e-12)
    return bpf / n_bodies, bpf, float(sig)


class WPTLayout:
    """机匣壁压传感器布置.

    TUDa 实际布局为轴向阵列 (同一周向位置), angles_deg 各通道相同,
    周向相干性/传播速度分析不适用 (见 array_processing 能力边界);
    axial_pos 为 rig 坐标 mm, 用于轴向外推/多传感器融合分组。

    Args:
        n_sensors: 通道数
        angles_deg: 各通道周向安装角 (deg); 轴向阵列传入同一角度
        axial_pos:  各通道轴向位置 [mm]
    """

    def __init__(self, n_sensors, angles_deg=None, axial_pos=None):
        self.n_sensors = n_sensors
        self.angles_deg = np.asarray(
            angles_deg if angles_deg is not None
            else np.linspace(0, 360, n_sensors, endpoint=False))
        self.axial_pos = axial_pos

    def spacing_matrix(self):
        """相邻(周向)通道间隔角矩阵 (rad); 轴向阵列时全为 0"""
        a = np.deg2rad(np.sort(self.angles_deg))
        return np.abs(np.diff(np.concatenate([a, a[:1] + 2 * np.pi])))

    @classmethod
    def tuda_axial(cls, axcor_mm):
        """由 read_raw_mat 的 axcor 构建 TUDa 轴向布局"""
        return cls(n_sensors=len(axcor_mm),
                   angles_deg=[260.0] * len(axcor_mm),
                   axial_pos=list(map(float, axcor_mm)))


# --------------------------------------------------------------------------
# episode 构建
# --------------------------------------------------------------------------
def build_episodes(t, P, fs, f_rotor, seg_s=20.0, hop_s=5.0, t_start=0.0):
    """把长记录切片为重叠 episode, 返回 list[dict(t, P, fs, f_rotor, meta)]"""
    t = np.asarray(t)
    P = np.asarray(P)
    seg_n = int(seg_s * fs)
    hop_n = int(hop_s * fs)
    i0 = int(t_start * fs)
    out = []
    while i0 + seg_n <= len(t):
        out.append(dict(
            t=t[i0:i0 + seg_n] - t[i0],
            P=P[i0:i0 + seg_n],
            p=P[i0:i0 + seg_n, 0] if P.ndim > 1 else P[i0:i0 + seg_n],
            fs=float(fs), f_rotor=float(f_rotor),
            meta=dict(t_abs_start=float(t[i0]), seg_s=seg_s)))
        i0 += hop_n
    return out


def resample_uniform(t, P, fs_new):
    """非均匀采样 -> 均匀采样 (线性插值; 高精度需求再换 spline)"""
    t = np.asarray(t)
    n_new = int((t[-1] - t[0]) * fs_new) + 1
    t_u = t[0] + np.arange(n_new) / fs_new
    if P.ndim == 1:
        P_u = np.interp(t_u, t, P)
    else:
        P_u = np.column_stack([np.interp(t_u, t, P[:, k])
                               for k in range(P.shape[1])])
    return t_u, P_u


# --------------------------------------------------------------------------
# ground truth 估计 (datasheet 未标注失速时刻时的后备)
# --------------------------------------------------------------------------
def pressure_fluctuation_criterion(episode, f_rotor=None,
                                   k_sigma=6.0, win_s=0.5):
    """基于失速带 RMS 相对基线跳变的失速起始时刻估计 (k_sigma 倍标准差)"""
    from detector import SurgeDetector, fft_bandpass, sliding_rms
    f_rotor = f_rotor or episode["f_rotor"]
    band = tuple(s * f_rotor for s in SurgeDetector.STALL_NORM)
    p = episode["p"] if "p" in episode else episode["P"][:, 0]
    pb = fft_bandpass(p - p.mean(), episode["fs"], *band)
    tw, rms = sliding_rms(pb, episode["fs"], win_s, 0.05)
    med = np.median(rms)
    mad = np.median(np.abs(rms - med)) + 1e-9
    idx = np.argmax(rms > med + k_sigma * 1.4826 * mad)
    return float(tw[idx]) if rms[idx] > med + k_sigma * 1.4826 * mad else None


# --------------------------------------------------------------------------
# 管线对接 (数据到手后打通这三行即可跑全套验证)
# --------------------------------------------------------------------------
def run_detection(episode, online=False, **kwargs):
    """episode -> 检测器 -> 报警结果 (离线/在线可选)"""
    if online:
        from detector_online import OnlineSurgeDetector
        det = OnlineSurgeDetector.from_rotor(
            fs=episode["fs"], f_rotor=episode["f_rotor"], **kwargs)
        if episode["P"].ndim > 1:
            return det.run_episode(episode["t"], episode["P"][:, 0])
        return det.run_episode(episode["t"], episode["p"])
    from detector import SurgeDetector
    det = SurgeDetector.from_rotor(
        fs=episode["fs"], f_rotor=episode["f_rotor"], **kwargs)
    p = episode["p"] if "p" in episode else episode["P"][:, 0]
    return det.run(episode["t"], p)


def sanity_check(episode):
    """数据到手后的第一件事: 检查采样率/转速/幅值量级是否合理"""
    t, p = episode["t"], (episode["p"] if "p" in episode else episode["P"][:, 0])
    fs = episode["fs"]
    f_rotor = episode["f_rotor"]
    dt = np.diff(t)
    checks = [
        ("时间戳均匀 (max dt抖动<1%)", bool(dt.max() - dt.min() < 0.01 * dt.mean())),
        ("Nyquist > 0.8*f_rotor", bool(fs / 2 > 0.8 * f_rotor)),
        ("失速带 0.625*f_rotor < Nyquist", bool(0.625 * f_rotor < fs / 2)),
        ("信号幅值非零且非饱和 (std在(1e-3,1e4)Pa)", bool(1e-3 < p.std() < 1e4)),
        ("无 NaN", bool(np.isfinite(p).all())),
    ]
    for name, ok in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    return all(ok for _, ok in checks)
