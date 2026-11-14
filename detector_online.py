# -*- coding: utf-8 -*-
"""
在线流式版预失速/喘振检测器 (causal, block-wise)

与离线版 (detector.py, FFT 全段零相位) 的区别:
  - 因果性: 只用当前时刻之前的样本, 可部署到实时监控 (NI cRIO / 嵌入式)
  - 滤波: RBJ cookbook 双二阶 IIR (手写系数, 无 scipy 依赖)
      * 高通 0.4 Hz 去趋势 (1 阶级联)
      * 失速带 100-250 Hz 带通 (2 级级联提高选择性)
      * 喘振带 0.4-12 Hz 带通
  - 包络: 指数滑动 RMS (EMA, 时间常数对应离线版窗口)
  - 报警逻辑: 与离线版一致 —— 相对起始稳定段基线超 k 倍且持续 n 个判定块

用法:
    det = OnlineSurgeDetector(fs=20000)
    for block in blocks:          # 每块 5 ms
        status = det.push(block)  # dict(alarm, r_stall, r_surge, ...)
"""
import numpy as np


# ---------------------------------------------------------------------------
# RBJ audio EQ cookbook 双二阶系数
# ---------------------------------------------------------------------------
def biquad_bandpass(fs, f0, Q):
    """常峰值带通 (constant 0 dB peak gain), RBJ cookbook BPF"""
    A = 10 ** (0 / 40.0)  # 增益 0 dB
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * Q)
    cw, sw = np.cos(w0), np.sin(w0)
    b0 = alpha
    b1, b2 = 0.0, -alpha
    a0, a1, a2 = 1 + alpha, -2 * cw, 1 - alpha
    return np.array([b0, b1, b2]) / a0, np.array([a1, a2]) / a0


def biquad_highpass(fs, f0, Q=0.707):
    """二阶高通 (也用作去趋势)"""
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * Q)
    cw, sw = np.cos(w0), np.sin(w0)
    b0 = (1 + cw) / 2
    b1 = -(1 + cw)
    b2 = (1 + cw) / 2
    a0, a1, a2 = 1 + alpha, -2 * cw, 1 - alpha
    return np.array([b0, b1, b2]) / a0, np.array([a1, a2]) / a0


class Biquad:
    """单级双二阶滤波器 (保持状态, 支持逐块推入)"""

    def __init__(self, b, a):
        self.b, self.a = b, a
        self.x1 = self.x2 = self.y1 = self.y2 = 0.0

    def prime(self, x0):
        """用首样本值预置状态: 恒定输入下稳态输出为 0 (BPF) / 消除直流阶跃振铃"""
        self.x1 = self.x2 = float(x0)
        self.y1 = self.y2 = 0.0

    def process(self, x):
        b0, b1, b2 = self.b
        a1, a2 = self.a
        y = np.empty_like(x)
        x1, x2, y1, y2 = self.x1, self.x2, self.y1, self.y2
        for i in range(len(x)):
            xi = x[i]
            yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            y[i] = yi
            x2, x1 = x1, xi
            y2, y1 = y1, yi
        self.x1, self.x2, self.y1, self.y2 = x1, x2, y1, y2
        return y

    def process_fast(self, x):
        """lfilter 向量化 (一次性处理整段时使用; 分块时退回 process)"""
        b0, b1, b2 = self.b
        a1, a2 = self.a
        y = np.zeros(len(x))
        x1, x2, y1, y2 = self.x1, self.x2, self.y1, self.y2
        for i in range(len(x)):
            xi = x[i]
            yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            y[i] = yi
            x2, x1 = x1, xi
            y2, y1 = y1, yi
        self.x1, self.x2, self.y1, self.y2 = x1, x2, y1, y2
        return y


def bandpass_chain(fs, f_lo, f_hi, n_stages=2):
    """带通级联: 用两级宽带带通近似矩形选择性 (Q 由带宽决定)"""
    fc = np.sqrt(f_lo * f_hi)
    Q = fc / (f_hi - f_lo)
    stages = []
    for _ in range(n_stages):
        b, a = biquad_bandpass(fs, fc, Q / np.sqrt(n_stages))
        stages.append(Biquad(b, a))
    return stages


class OnlineSurgeDetector:
    """
    在线双频带自适应阈值检测器.

    参数与离线版对齐:
      stall_band (0.25-0.625 倍转频), surge_band (0.0025-0.03 倍转频)
      EMA 时间常数: tau_stall = win_s/2, tau_surge = win_surge_s/2
      判定块 block_s (默认 10 ms), 持续 n_sustain 个块超阈值 -> 报警
      基线: 前 baseline_s 秒 (起始稳定段) 的块 RMS 中位数, 之后慢速自适应

    推荐用 from_rotor() 工厂构造: 频带按转频归一化 (倍数),
    窗口/时间常数按转子转数归一化, 更换台架/转速/采样率无需重新整定。
    """

    # ---- 归一化设计常数 (× f_rotor 或 × 转子转数), 依据本仿真数据集整定 ----
    STALL_NORM = (0.25, 0.625)     # 失速带, × f_rotor
    SURGE_NORM = (0.0025, 0.03)    # 喘振带, × f_rotor (400Hz -> 1~12 Hz)
    DETREND_NORM = 0.0025          # 去趋势高通, × f_rotor
    BLOCK_REVS = 4                 # 判定块长度
    WARMUP_REVS = 600              # 滤波链热身时长 (需 >5x hp 时间常数, 否则
                                   # 直流暂态尾污染基线学习窗)
    BASELINE_REVS = 220            # 基线学习窗长
    TAU_STALL_REVS = 10            # 失速带 EMA 时间常数
    TAU_SURGE_REVS = 120           # 喘振带 EMA 时间常数
    TAU_BASE_REVS = 4000           # 喘振带慢速自适应基线 (需远长于失稳事件时长,
                                   # 否则持续振荡被 '学' 成基线 -> 阈值追赶漏报)

    def __init__(self, fs, f_rotor=400.0,
                 stall_band=(100.0, 250.0), surge_band=(1.0, 12.0),
                 detrend_hz=1.0, baseline_s=0.55, baseline_skip_s=0.15,
                 k_stall=4.0, k_surge=4.0,
                 n_sustain=4, block_s=0.01,
                 tau_stall_s=0.025, tau_surge_s=0.3, tau_base_s=2.0,
                 vote_n=5, vote_m=3, warmup_s=0.3):
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
        self.vote_n = vote_n
        self.vote_m = vote_m
        self.warmup_s = warmup_s
        self.block_len = max(int(round(block_s * fs)), 1)
        # 滤波链
        self.hp = Biquad(*biquad_highpass(fs, detrend_hz))
        self.stall_stages = bandpass_chain(fs, *stall_band, n_stages=2)
        # 喘振带双子带并联: 宽带 (0.83-10Hz) 单级 BPF 对 1Hz 边缘衰减 -22dB,
        # 拆低/高子带各自级联 BPF 后能量合成, 恢复对近临界低频前兆的灵敏度
        f_lo, f_hi = surge_band
        g = np.sqrt(f_hi / f_lo)
        self.surge_sub = [(f_lo, f_lo * g), (f_lo * g, f_hi)]
        self.surge_stages = [bandpass_chain(fs, *sb, n_stages=2)
                             for sb in self.surge_sub]
        # EMA 系数
        self.a_stall = 1.0 - np.exp(-self.block_len / fs / tau_stall_s)
        self.a_surge = 1.0 - np.exp(-self.block_len / fs / tau_surge_s)
        self.a_base = 1.0 - np.exp(-self.block_len / fs / tau_base_s)
        self.reset()

    @classmethod
    def from_rotor(cls, fs, f_rotor, **overrides):
        """由归一化常数计算物理参数构造检测器 (换台架只改 fs/f_rotor)"""
        phys = dict(
            stall_band=tuple(s * f_rotor for s in cls.STALL_NORM),
            surge_band=tuple(s * f_rotor for s in cls.SURGE_NORM),
            detrend_hz=cls.DETREND_NORM * f_rotor,
            block_s=cls.BLOCK_REVS / f_rotor,
            warmup_s=cls.WARMUP_REVS / f_rotor,
            baseline_s=cls.BASELINE_REVS / f_rotor,
            tau_stall_s=cls.TAU_STALL_REVS / f_rotor,
            tau_surge_s=cls.TAU_SURGE_REVS / f_rotor,
            tau_base_s=cls.TAU_BASE_REVS / f_rotor,
        )
        phys.update(overrides)
        return cls(fs=fs, f_rotor=f_rotor, **phys)

    def reset(self):
        self.hp = Biquad(*biquad_highpass(self.fs, self.detrend_hz))
        self.stall_stages = bandpass_chain(self.fs, *self.stall_band, n_stages=2)
        f_lo, f_hi = self.surge_band
        g = np.sqrt(f_hi / f_lo)
        self.surge_sub = [(f_lo, f_lo * g), (f_lo * g, f_hi)]
        self.surge_stages = [bandpass_chain(self.fs, *sb, n_stages=2)
                             for sb in self.surge_sub]
        self.t = 0.0
        self.s2_stall = 0.0   # EMA of x^2
        self.s2_surge = 0.0
        self._ema_init = False
        self.baseline_blocks = []   # 起始稳定段的块 RMS
        self.base_stall = None
        self.base_surge = None
        self.cnt_s = 0
        self.hist_g = []      # 喘振带表决窗口
        self.s2_base_surge = 0.0   # 慢速自适应基线 (EMA of r^2)
        self.alarm_time = None
        self.alarm_type = None
        self.trace = []       # (t, r_stall, r_surge)

    def _band_rms(self, x_detrended):
        xs = x_detrended
        for st in self.stall_stages:
            xs = st.process(xs)
        # 喘振带: 双子带并联, 能量合成 RMS
        s2_g = 0.0
        for stages in self.surge_stages:
            xg = x_detrended
            for st in stages:
                xg = st.process(xg)
            s2_g += float(np.mean(xg ** 2))
        return xs, np.sqrt(s2_g + 1e-12)

    def push(self, block, warmup=False):
        """推入一个原始压力数据块 (np.ndarray), 返回当前状态 dict.
        warmup=True 时仅热身滤波器/EMA, 不记轨迹/基线/报警 (t 不前进)"""
        n = len(block)
        if not self._ema_init:
            all_stages = [self.hp, *self.stall_stages]
            for stages in self.surge_stages:
                all_stages += list(stages)
            for f in all_stages:
                f.prime(block[0])  # 消除直流阶跃引起的滤波器振铃
        xd = self.hp.process(block.astype(float))
        xs, xg = self._band_rms(xd)

        # 块 RMS -> EMA 包络 (首块直接初始化, 避免冷启动低估)
        r_stall_blk = np.sqrt(np.mean(xs ** 2) + 1e-12)
        r_surge_blk = np.sqrt(np.mean(xg ** 2) + 1e-12)
        if not self._ema_init:
            self.s2_stall = r_stall_blk ** 2
            self.s2_surge = r_surge_blk ** 2
            self._ema_init = True
        else:
            self.s2_stall += self.a_stall * (r_stall_blk ** 2 - self.s2_stall)
            self.s2_surge += self.a_surge * (r_surge_blk ** 2 - self.s2_surge)
        r_stall = np.sqrt(self.s2_stall)
        r_surge = np.sqrt(self.s2_surge)

        if warmup:
            return dict(alarm=None, t=self.t, r_stall=r_stall, r_surge=r_surge,
                        warmup=True)
        self.t += n / self.fs
        self.trace.append((self.t, r_stall, r_surge))

        # 基线学习期
        if self.base_stall is None:
            self.baseline_blocks.append((r_stall, r_surge))
            if self.t >= self.baseline_s:
                arr = np.array(self.baseline_blocks)
                self.base_stall = float(np.median(arr[:, 0])) + 1e-9
                self.base_surge = float(np.median(arr[:, 1])) + 1e-9
                self.s2_base_surge = self.base_surge ** 2  # 自适应基线初值
            return dict(alarm=None, t=self.t, r_stall=r_stall, r_surge=r_surge,
                        warmup=True)

        # 持续性判据: 失速带连续 n_sustain 块; 喘振带 vote_m/vote_n 多数表决
        # (喘振 EMA 在低频振荡周期内有起伏, 连续判据会被波谷清零)
        self.cnt_s = self.cnt_s + 1 if r_stall > self.k_stall * self.base_stall else 0

        # 喘振带自适应基线: 慢速跟踪漂移 (下限锁定初始学习值, 只升不降过快),
        # 阈值 = k_surge * baseline_slow —— 这是因果在线系统处理
        # 非平稳漂移的标准做法 (离线版用 FFT 清零漂移频点, 在线无法复制)
        self.s2_base_surge += self.a_base * (r_surge ** 2 - self.s2_base_surge)
        base_g = max(np.sqrt(self.s2_base_surge), self.base_surge)
        self.hist_g.append(r_surge > self.k_surge * base_g)
        if len(self.hist_g) > self.vote_n:
            self.hist_g.pop(0)
        if self.alarm_time is None:
            if sum(self.hist_g) >= self.vote_m:
                self.alarm_time, self.alarm_type = self.t, "surge_band"
            elif self.cnt_s >= self.n_sustain:
                self.alarm_time, self.alarm_type = self.t, "stall_band"
        return dict(alarm=self.alarm_time, t=self.t,
                    r_stall=r_stall, r_surge=r_surge,
                    exc_stall=self.cnt_s >= self.n_sustain,
                    exc_surge=sum(self.hist_g) >= self.vote_m,
                    warmup=False)

    def run_episode(self, t, p):
        """整段推入 (分块), 返回与离线版兼容的结果 dict"""
        self.reset()
        n_blk = self.block_len
        # 滤波器热身: 用首样本恒定值喂 warmup_s 秒, 让链路状态进入稳态 (丢弃输出)
        if self.warmup_s > 0:
            n_warm = int(self.warmup_s * self.fs) // n_blk * n_blk
            for i in range(0, n_warm, n_blk):
                self.push(np.full(n_blk, p[0]), warmup=True)
        N = (len(p) // n_blk) * n_blk
        for i in range(0, N, n_blk):
            self.push(p[i:i + n_blk])
        tr = np.array(self.trace)
        return dict(t_win=tr[:, 0], r_stall=tr[:, 1], r_surge=tr[:, 2],
                    base_stall=self.base_stall, base_surge=self.base_surge,
                    alarm_time=self.alarm_time, alarm_type=self.alarm_type)
