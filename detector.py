# -*- coding: utf-8 -*-
"""
预失速/喘振检测器 —— 只使用机匣壁面动态压力信号 p(t)

思路参考清华大学多特征融合自适应喘振检测方法 (Aerospace 2026, doi:10.3390/aerospace13080734):
  1. 带通滤波提取失稳相关频带能量 (失速带 ~ 0.4*N 转频; 喘振带低频)
  2. 滑动窗口特征: 频带 RMS + 斜率特征
  3. 自适应阈值 (以起始稳定段基线为参考) + 持续性判据 (抗瞬态干扰)

不依赖 scipy: 用 FFT 带通 + numpy 滑动统计实现。
"""
import numpy as np


def fft_bandpass(x, fs, f_lo, f_hi):
    """FFT 零相位带通滤波 (整段信号, 适合离线/准在线分块处理)"""
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    X[~mask] = 0.0
    return np.fft.irfft(X, n=len(x))


def bandpass_bands(x, fs, bands, seg_s=4.0, overlap=0.5, detrend=True):
    """分块 WOLA 多带带通 (一次分块, 块内单次 FFT 提取多个频带).

    bands: list[(f_lo, f_hi)] -> 返回 list[np.ndarray] 对应各带信号.

    架构要点: 去趋势与分带必须在**同一块内**完成 —— 若先分块高通再串联
    第二次分块带通, 第一级的块边界去趋势跳变会产生宽带阶跃, 被第二级
    带通提取为周期性尖峰 (校准数据实测暴露: 每 2s 一个假报警)。
    单块内 FFT 后用不同 mask 提取各带, 块间输出由 Hann 权重自然归零。

    seg_s=4.0: 喘振带下限 0.33 Hz 在 4s 块内有 ~1.3 个周期; 块泄漏
    支撑 ±2s, 距事件 >4s 的基线段不受污染。块内线性去趋势干净移除
    缓变节流趋势 (块内近似线性), 其谱泄漏不再进入喘振带下段。
    """
    x = np.asarray(x, float)
    n = len(x)
    seg = int(seg_s * fs)
    outs = [np.zeros(n) for _ in bands]
    if n <= seg:
        x0 = x - np.linspace(x[0], x[-1], n) if detrend else x
        X = np.fft.rfft(x0)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        for k, (lo, hi) in enumerate(bands):
            Xk = X.copy()
            Xk[(freqs < lo) | (freqs > hi)] = 0.0
            outs[k] = np.fft.irfft(Xk, n=n)
        return outs
    hop = max(int(seg * (1.0 - overlap)), 1)
    w = np.hanning(seg)
    pad = seg - hop
    # 边缘值填充 (非零填充): 整段去均值后头部仍可能有直流偏置,
    # 零填充在拼接处产生阶跃, 经块内去趋势残留并污染头部基线
    xp = np.concatenate([np.full(pad, x[0]), x, np.full(pad + seg, x[-1])])
    bufs = [np.zeros(len(xp)) for _ in bands]
    wsum = np.zeros(len(xp))
    freqs = np.fft.rfftfreq(seg, d=1.0 / fs)
    masks = [((freqs >= lo) & (freqs <= hi)) for lo, hi in bands]
    for i0 in range(0, len(xp) - seg + 1, hop):
        xb = xp[i0:i0 + seg]
        if detrend:
            xb = xb - np.linspace(xb[0], xb[-1], seg)
        X = np.fft.rfft(xb * w)
        for k in range(len(bands)):
            Xk = X.copy()
            Xk[~masks[k]] = 0.0
            bufs[k][i0:i0 + seg] += np.fft.irfft(Xk, n=seg)
        wsum[i0:i0 + seg] += w
    for k in range(len(bands)):
        bufs[k] /= np.maximum(wsum, 0.5)
        outs[k] = bufs[k][pad:pad + n]
    return outs


def bandpass_blocked(x, fs, f_lo, f_hi, seg_s=4.0, overlap=0.5, detrend=True):
    """单带分块 WOLA 带通 (bandpass_bands 的单带便捷封装)."""
    return bandpass_bands(x, fs, [(f_lo, f_hi)], seg_s=seg_s,
                          overlap=overlap, detrend=detrend)[0]


def fft_highpass(x, fs, f_lo):
    """FFT 零相位高通 (去除节气门漂移等慢趋势, 避免污染喘振带基线)"""
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    X[freqs < f_lo] = 0.0
    return np.fft.irfft(X, n=len(x))


def sliding_rms(x, fs, win_s=0.05, hop_s=0.01):
    """滑动 RMS (窗口 win_s, 步长 hop_s), 返回窗口中心时间与 RMS"""
    win = max(int(win_s * fs), 2)
    hop = max(int(hop_s * fs), 1)
    c2 = np.concatenate([[0.0], np.cumsum(x.astype(float) ** 2)])
    k = np.arange(win, len(x) + 1, hop)
    rms = np.sqrt((c2[k] - c2[k - win]) / win)
    t = (k - win / 2) / fs
    return t, rms


def sliding_slope(x, fs, win_s=0.05, hop_s=0.01):
    """滑动窗口内最大压降斜率 |dp/dt| (区分喘振与阶跃干扰的特征之一)"""
    win = max(int(win_s * fs), 2)
    hop = max(int(hop_s * fs), 1)
    k = np.arange(win, len(x) + 1, hop)
    slope = np.abs(np.diff(x))[...]
    cs = np.concatenate([[0.0], np.cumsum(slope)])
    return (k - win / 2) / fs, (cs[k] - cs[k - win]) / win


class SurgeDetector:
    """
    双频带 + 自适应阈值检测器.

    失速带 (stall band): 0.25~0.625 倍转频 -> 捕捉模态波/失速团
    喘振带 (surge band): 0.001~0.03 倍转频 低频大幅压力振荡
    报警条件: 频带 RMS 连续 n_sustain 个窗口超过 基线*k 倍

    推荐用 from_rotor() 工厂构造: 所有频带按转频归一化 (倍数),
    窗口按转数归一化, 更换台架/转速/采样率无需重新整定。
    """

    # ---- 归一化设计常数 (× f_rotor 或 × 转子转数), 依据本仿真数据集整定 ----
    STALL_NORM = (0.25, 0.625)     # 失速带, × f_rotor
    SURGE_NORM = (0.001, 0.03)     # 喘振带, × f_rotor (400Hz -> 0.4~12 Hz)
    DETREND_NORM = 0.001           # 去趋势高通, × f_rotor
    WIN_REVS = 20                  # 失速带 RMS 窗口 (转子转数)
    HOP_REVS = 4                   # 失速带步长
    WIN_SURGE_REVS = 200           # 喘振带 RMS 窗口
    HOP_SURGE_REVS = 40            # 喘振带步长
    BASELINE_REVS = 220            # 基线学习窗长 (起始稳定段)
    BASELINE_SKIP_REVS = 60        # 基线跳过 (滤波边缘瞬态)

    def __init__(self, fs, f_rotor=400.0,
                 stall_band=(100.0, 250.0), surge_band=(0.4, 12.0),
                 detrend_hz=0.4, baseline_s=0.55, baseline_skip_s=0.15,
                 k_stall=5.0, k_surge=6.0,
                 n_sustain=4, win_s=0.05, hop_s=0.01,
                 win_surge_s=0.5, hop_surge_s=0.1):
        self.fs = fs
        self.f_rotor = f_rotor
        self.stall_band = stall_band
        self.surge_band = surge_band
        self.detrend_hz = detrend_hz
        self.baseline_s = baseline_s
        self.baseline_skip_s = baseline_skip_s
        self.k_stall = k_stall
        self.k_surge = k_surge
        self.n_sustain = n_sustain
        self.win_s = win_s
        self.hop_s = hop_s
        self.win_surge_s = win_surge_s
        self.hop_surge_s = hop_surge_s

    @classmethod
    def from_rotor(cls, fs, f_rotor, **overrides):
        """由归一化常数计算物理参数构造检测器 (换台架只改 fs/f_rotor)"""
        phys = dict(
            stall_band=tuple(s * f_rotor for s in cls.STALL_NORM),
            surge_band=tuple(s * f_rotor for s in cls.SURGE_NORM),
            detrend_hz=cls.DETREND_NORM * f_rotor,
            win_s=cls.WIN_REVS / f_rotor,
            hop_s=cls.HOP_REVS / f_rotor,
            win_surge_s=cls.WIN_SURGE_REVS / f_rotor,
            hop_surge_s=cls.HOP_SURGE_REVS / f_rotor,
            baseline_s=cls.BASELINE_REVS / f_rotor,
            baseline_skip_s=cls.BASELINE_SKIP_REVS / f_rotor,
        )
        phys.update(overrides)
        return cls(fs=fs, f_rotor=f_rotor, **phys)

    def run(self, t, p):
        """返回 dict(t_win, r_stall, r_surge, alarm_time, alarm_type)"""
        fs = self.fs
        # 1. 去趋势 (去除节气门慢漂移) + 分带
        #    分块 WOLA 滤波: 非平稳大事件 (末端喘振环游) 不向基线段泄漏
        p_hp, p_stall, p_surge = bandpass_bands(
            p - np.mean(p), fs,
            [(self.detrend_hz, fs / 2), self.stall_band, self.surge_band])
        # 2. 滑动 RMS (失速带短窗, 喘振带长窗)
        tw, r_stall = sliding_rms(p_stall, fs, self.win_s, self.hop_s)
        tg, r_surge = sliding_rms(p_surge, fs, self.win_surge_s, self.hop_surge_s)
        # 3. 基线 (起始稳定段, 跳过滤波边缘瞬态)
        bmask_s = (tw >= self.baseline_skip_s) & (tw <= self.baseline_s)
        bmask_g = (tg >= self.baseline_skip_s) & (tg <= self.baseline_s)
        base_stall = np.median(r_stall[bmask_s]) + 1e-9 if bmask_s.any() else r_stall[0] + 1e-9
        base_surge = np.median(r_surge[bmask_g]) + 1e-9 if bmask_g.any() else r_surge[0] + 1e-9
        # 4. 持续性判据: 连续 n_sustain 窗口超阈值 -> 报警 (滚动计数)
        #    报警仅在基线学习窗建立之后有效 (窗内滤波瞬态/泄漏不作报警)
        exc_stall = r_stall > self.k_stall * base_stall
        exc_surge = r_surge > self.k_surge * base_surge
        t_armed = max(self.baseline_s, self.win_surge_s)
        cnt_s = cnt_g = 0
        alarm_time, alarm_type = None, None
        # 两带各自滑动计数, 取先触发者 (喘振带时间网格)
        cnt_s_arr = np.zeros(len(tw), dtype=int)
        for i in range(len(tw)):
            cnt_s = cnt_s + 1 if exc_stall[i] else 0
            cnt_s_arr[i] = cnt_s
        for i in range(len(tg)):
            if tg[i] < t_armed:
                continue
            cnt_g = cnt_g + 1 if exc_surge[i] else 0
            if alarm_time is None and cnt_g >= self.n_sustain:
                alarm_time, alarm_type = tg[i], "surge_band"
        for i in range(len(tw)):
            if tw[i] < t_armed:
                continue
            if cnt_s_arr[i] >= self.n_sustain:
                if alarm_time is None or tw[i] < alarm_time:
                    alarm_time, alarm_type = tw[i], "stall_band"
                break
        return dict(t_win=tw, t_surge_win=tg,
                    r_stall=r_stall, r_surge=r_surge,
                    base_stall=base_stall, base_surge=base_surge,
                    alarm_time=alarm_time, alarm_type=alarm_type,
                    exc_stall=exc_stall, exc_surge=exc_surge)
