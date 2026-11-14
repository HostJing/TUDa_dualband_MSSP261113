# -*- coding: utf-8 -*-
"""
周向阵列多传感器处理模块

背景: TUDa-GLR-OpenStage 等 WPT 测量沿机匣周向+轴向布置多个动态压力传感器。
旋转失速团在相邻传感器间产生确定的相位超前/滞后, 而噪声/电磁干扰在通道间
不相干 —— 这两个物理事实可以:
  1) 用周向互相关估计失速团传播速度 (失速团转速占转子转速的比例 beta);
  2) 用空间相干性 (inter-sensor coherence) 区分真实旋转失稳与不相干干扰,
     压制单通道检测器的误报。

多通道信号合成 (synthesize_array):
  在单通道 episode (npz) 基础上重建 K 通道周向阵列信号:
    p_k(t) = p_mean(t)                       共模 (喘振为平面波, 通道间零相差)
           + A_rs*sqrt(J)*cos(2*pi*f_rs*t + phi0 + m*2*pi*k/K)   旋转失速团 (m=1)
           + 各通道独立噪声 + 各通道独立 EMI 脉冲                 不相干分量
  相邻传感器间隔角 2*pi/K, 失速团扫过该角度用时
    dt = 1/(K * beta * f_rotor),  beta = f_rs/f_rotor = stall_frac
"""
import numpy as np

# 与 generate_dataset.py 一致的信号合成常数
A_RS = 350.0
NOISE_STD = 5.0
K_P = 10000.0
EMI_F = 900.0


# ---------------------------------------------------------------------------
# 多通道信号合成
# ---------------------------------------------------------------------------
def synthesize_array(d_npz, n_sensors=8, seed=7, angles_deg=None,
                     mode_order=1, noise_std=NOISE_STD, gains=None):
    """由 episode npz (含 t, Psi, J, fs, f_rotor, stall_frac) 生成 K 通道阵列信号

    angles_deg: 各通道周向安装角 (deg); None 按等间隔处理.
                非均匀布置时旋转模态相位按实际角度计算.
    mode_order: 周向模态阶数 (默认 m=1 单失速团)
    noise_std : 每通道独立噪声标准差 [Pa]
    gains     : 各通道静态增益 (len K, 可为 None); 模拟传感器灵敏度失配
    返回 dict(t, P (T,K), fs, f_rotor, stall_frac)
    """
    t = d_npz["t"]
    Psi, J = d_npz["Psi"], d_npz["J"]
    fs = float(d_npz["fs"])
    f_rotor = float(d_npz["f_rotor"])
    stall_frac = float(d_npz["stall_frac"])
    T = len(t)
    rng = np.random.default_rng(seed)

    if angles_deg is None:
        angles_deg = np.linspace(0, 360, n_sensors, endpoint=False)
    angles_deg = np.asarray(angles_deg, dtype=float)
    theta = np.deg2rad(angles_deg)

    # 共模: 喘振/漂移是集气室平面压力波 -> 各通道相同 (零相差)
    p_mean = Psi * K_P
    # 旋转失速团: 周向模态 m, 通道 k 相位滞后 m*theta_k
    f_rs = stall_frac * f_rotor
    phase0 = rng.uniform(0, 2 * np.pi)
    ang = 2 * np.pi * f_rs * t[:, None] + phase0 - mode_order * theta[None, :]
    p_rs = A_RS * np.sqrt(np.maximum(J, 0.0))[:, None] * np.cos(ang)

    P = p_mean[:, None] + p_rs \
        + rng.normal(0.0, noise_std, size=(T, n_sensors))
    if gains is not None:
        P = P * np.asarray(gains, dtype=float)[None, :]

    # 各通道独立 EMI 脉冲 (不相干)
    for k in range(n_sensors):
        n_burst = rng.integers(0, 3)
        for _ in range(n_burst):
            b0 = rng.integers(0, max(T - 400, 1))
            blen = rng.integers(20, 200)
            amp = rng.uniform(30, 120) * rng.choice([-1, 1])
            decay = np.exp(-np.linspace(0, 6, blen))
            P[b0:b0 + blen, k] += amp * decay * np.sin(2 * np.pi * EMI_F * t[b0:b0 + blen])
    return dict(t=t, P=P, fs=fs, f_rotor=f_rotor, stall_frac=stall_frac)


# ---------------------------------------------------------------------------
# 频带工具 (FFT 零相位, 离线分析用)
# ---------------------------------------------------------------------------
def fft_bandpass(x, fs, f_lo, f_hi):
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    X[(freqs < f_lo) | (freqs > f_hi)] = 0.0
    return np.fft.irfft(X, n=len(x))


def xcorr_lag(x, y, fs, max_lag_s):
    """FFT 互相关, 返回 (lag_s, peak_corr).
    lag>0 表示 x 滞后 y (x 的波形在时间上出现得更晚).
    峰值做抛物线亚样本插值, 消除滞后量化误差 (1/fs)."""
    n = len(x) + len(y)
    nfft = 1 << int(np.ceil(np.log2(n)))
    X = np.fft.rfft(x, nfft)
    Y = np.fft.rfft(y, nfft)
    c = np.fft.irfft(np.conj(X) * Y, nfft)
    # c[k] 对应 y 滞后 x k 个样本 -> 互换解释: x 相对 y 的滞后为 -k
    # 直接在圆周内搜索: 负 lag 用 nfft-k 表示
    max_lag = max(int(round(max_lag_s * fs)), 1)
    pos = c[1:max_lag + 1]
    neg = c[nfft - max_lag:nfft]
    i_p = int(np.argmax(pos)) + 1
    i_n = int(np.argmax(neg))
    if pos[i_p - 1] >= neg[i_n]:
        k0, c0 = i_p, pos[i_p - 1]
        # 抛物线插值 (邻点越界时用 0)
        cm = c[k0 - 1] if k0 - 1 >= 0 else 0.0
        cp = c[k0 + 1] if k0 + 1 < nfft else 0.0
        den = cm - 2 * c0 + cp
        delta = 0.5 * (cm - cp) / den if abs(den) > 1e-12 else 0.0
        delta = float(np.clip(delta, -0.5, 0.5))
        return (k0 + delta) / fs, c0
    k0, c0 = max_lag - i_n, neg[i_n]
    gidx = nfft - max_lag + i_n
    cm = c[gidx - 1] if gidx - 1 >= 0 else 0.0
    cp = c[gidx + 1] if gidx + 1 < nfft else 0.0
    den = cm - 2 * c0 + cp
    delta = 0.5 * (cm - cp) / den if abs(den) > 1e-12 else 0.0
    delta = float(np.clip(delta, -0.5, 0.5))
    return -(k0 - delta) / fs, c0


def windowed_speed(P_band, fs, f_rotor, angles_deg=None,
                   win_s=0.2, hop_s=0.05, max_lag_s=0.01):
    """时间分辨的失速团传播速度估计 (相邻通道互相关滞后)

    P_band: (T, K) 已带通的阵列信号
    angles_deg: 各通道周向安装角 (deg); None 时按等间隔 K 通道处理.
                支持任意非均匀布置 (TUDa WPT 为扩散安装) —— 按角度排序后
                对每个相邻对用其实际间隔角换算速度。
    返回 dict(t, beta, corr)  beta = 失速团转速 / 转子转速
    """
    K = P_band.shape[1]
    if angles_deg is None:
        angles_deg = np.linspace(0, 360, K, endpoint=False)
    angles_deg = np.asarray(angles_deg, dtype=float) % 360.0
    order = np.argsort(angles_deg)
    ang_sorted = angles_deg[order]
    # 相邻对 (含首尾闭合) 的间隔角 [deg]
    dtheta = np.abs(np.diff(np.concatenate([ang_sorted, ang_sorted[:1] + 360.0])))

    win = int(win_s * fs)
    hop = int(hop_s * fs)
    ts, betas, corrs = [], [], []
    for i0 in range(0, len(P_band) - win + 1, hop):
        seg = P_band[i0:i0 + win]
        beta_pairs, corr_pairs = [], []
        for k in range(K):
            x = seg[:, order[k]]
            y = seg[:, order[(k + 1) % K]]
            lag, c = xcorr_lag(x, y, fs, max_lag_s)
            # 失速团扫过 dtheta 度用时 |lag| -> 转一圈 360 度用时 360/(f_rotor*beta)
            if lag != 0:
                beta_pairs.append(dtheta[k] / (360.0 * f_rotor * abs(lag)))
            corr_pairs.append(abs(c))
        ts.append((i0 + win / 2) / fs)
        betas.append(np.median(beta_pairs) if beta_pairs else np.nan)
        corrs.append(np.median(corr_pairs) / max(np.sqrt(np.mean(seg ** 2)), 1e-9))
    return dict(t=np.array(ts), beta=np.array(betas), corr=np.array(corrs))


def spatial_coherence(P, fs, band, n_sub=8):
    """失速频带内相邻通道平均空间相干性 (Welch 跨段平均, 返回时间序列)

    coherence(f) = |<X_k conj(X_k+1)>|^2 / (<|X_k|^2> <|X_k+1|^2>)
    <·> 为跨子段平均; 相干旋转失稳 -> 高, 独立噪声/EMI -> 低
    返回 (t_center, coherence_series)
    """
    T = P.shape[0]
    freqs = np.fft.rfftfreq(min(2048, T // 4), d=1.0 / fs)
    bmask = (freqs >= band[0]) & (freqs <= band[1])
    K = P.shape[1]
    out, tc = [], []
    sub_len = min(2048, T // 4)
    win_len = int(sub_len + (n_sub - 1) * sub_len / 2)   # 50% 重叠
    hop = win_len // 2
    hann = np.hanning(sub_len)[:, None]
    tt = np.arange(sub_len, dtype=float)
    for i0 in range(0, T - win_len + 1, hop):
        blk = P[i0:i0 + win_len, :]
        Sxy = np.zeros((sub_len // 2 + 1, K), dtype=complex)
        Sxx = np.zeros((sub_len // 2 + 1, K))
        n_avg = 0
        for s in range(n_sub):
            a = int(s * sub_len / 2)
            sub = blk[a:a + sub_len, :].astype(float)
            # 逐通道去线性趋势 + Hann 窗: 消除共模慢漂移的旁瓣泄漏
            # (否则 ~kPa 级直流/斜坡的 sinc 旁瓣在带内频点完全相干, 污染估计)
            t1 = tt - tt.mean()
            xc = sub - sub.mean(0)
            coef = (xc * t1[:, None]).sum(0) / (t1 @ t1)
            sub = xc - np.outer(t1, coef)
            X = np.fft.rfft(sub * hann, axis=0)          # (F, K)
            Sxy += X * np.conj(X[:, np.roll(np.arange(K), -1)])
            Sxx += np.abs(X) ** 2
            n_avg += 1
        Sxy, Sxx = Sxy / n_avg, Sxx / n_avg
        coh_pairs = []
        for k in range(K):
            coh_f = np.abs(Sxy[:, k]) ** 2 / (Sxx[:, k] * Sxx[:, (k + 1) % K] + 1e-12)
            coh_pairs.append(np.mean(coh_f[bmask]))
        out.append(np.mean(coh_pairs))
        tc.append((i0 + win_len / 2) / fs)
    return np.array(tc), np.array(out)


def _sustained_runs(exc, t_win, n_sustain):
    """把超阈布尔序列切分为持续 >= n_sustain 的游程 [(start, end)]"""
    runs, i, n = [], 0, len(exc)
    while i < n:
        if exc[i]:
            j = i
            while j < n and exc[j]:
                j += 1
            if j - i >= n_sustain:
                runs.append((float(t_win[i + n_sustain - 1]),
                             float(t_win[j - 1])))
            i = j
        else:
            i += 1
    return runs


def coherence_gated_alarm(P, fs, f_rotor, k_stall=4, k_surge=4,
                          coh_thr=0.4, coh_surge_thr=0.5, confirm_s=5.0):
    """单通道检测器 + 空间相干性闸门 (游程分段确认):

    1) 通道 0 单通道检测 -> 失速带/喘振带各自的持续超阈游程 (候选事件);
    2) 失速带候选: 游程内按 confirm_s 分段检查失速带空间相干性, 越过
       coh_thr 即确认 (旋转失稳团相干; 噪声/EMI/漂移泄漏不相干);
       全程不相干 -> 拒绝 (干扰), 不影响后续游程评估;
    3) 喘振带候选: 喘振/真实压力瞬变是共模平面波 (跨通道零相差, 相干),
       通道级噪声/EMI 不相干 -> 用喘振带相干性 > coh_surge_thr 把关。
    """
    from detector import SurgeDetector
    stall_band = tuple(s * f_rotor for s in SurgeDetector.STALL_NORM)
    surge_band = tuple(s * f_rotor for s in SurgeDetector.SURGE_NORM)
    det = SurgeDetector.from_rotor(fs=fs, f_rotor=f_rotor,
                                   k_stall=k_stall, k_surge=k_surge)
    t0 = np.arange(P.shape[0]) / fs
    res = det.run(t0, P[:, 0])

    runs_s = _sustained_runs(res["exc_stall"], res["t_win"], det.n_sustain)
    runs_g = _sustained_runs(res["exc_surge"], res["t_surge_win"], det.n_sustain)
    tc_s, coh_s = spatial_coherence(P, fs, stall_band)
    tc_g, coh_g = spatial_coherence(P, fs, surge_band)

    def _cross(t_lo, t_hi, tc, coh, thr):
        """相干性在 [t_lo, t_hi] 内首次越过 thr 的时刻, 无则 None"""
        m = (tc >= t_lo) & (tc <= t_hi)
        if not m.any():
            return None
        sub_t, sub_c = tc[m], coh[m]
        idx = np.argmax(sub_c > thr) if np.any(sub_c > thr) else -1
        return float(sub_t[idx]) if idx >= 0 else None

    # 失速带游程: 逐段 (confirm_s) 检查相干性
    alarm_time = alarm_type = confirm_time = None
    for t_lo, t_hi in runs_s:
        k = 0
        while alarm_time is None and t_lo + k * confirm_s <= t_hi:
            c = _cross(t_lo + k * confirm_s,
                       min(t_lo + (k + 1) * confirm_s, t_hi + confirm_s / 2),
                       tc_s, coh_s, coh_thr)
            if c is not None:
                alarm_time, alarm_type, confirm_time = t_lo, "stall_band", c
            k += 1
        if alarm_time is not None:
            break
    # 喘振带游程 (若失速带未确认)
    if alarm_time is None:
        for t_lo, t_hi in runs_g:
            c = _cross(t_lo, min(t_hi + confirm_s / 2, t_hi + confirm_s),
                       tc_g, coh_g, coh_surge_thr)
            if c is not None:
                alarm_time, alarm_type, confirm_time = t_lo, "surge_band", c
                break
    return dict(alarm_time=alarm_time, alarm_type=alarm_type,
                confirm_time=confirm_time, confirmed=alarm_time is not None,
                n_runs_stall=len(runs_s), n_runs_surge=len(runs_g),
                t_coh=tc_s, coherence=coh_s,
                stall_band=stall_band, surge_band=surge_band,
                coh_thr=coh_thr, coh_surge_thr=coh_surge_thr)
