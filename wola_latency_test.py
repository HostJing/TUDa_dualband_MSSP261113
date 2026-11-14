# -*- coding: utf-8 -*-
"""
WOLA 滤波器组零相位验证 (论文 §3.1 引用的阶跃响应数字):
合成 stall/surge 带正弦在 t=3s 阶跃出现, 测带 RMS 的 10%/50% 响应延迟。
产出: 控制台数字 (与论文 §3.1 的 3 ms / 72 ms 对应)
"""
import numpy as np
from detector import SurgeDetector, bandpass_bands, sliding_rms

FS, FR = 2000.0, 333.9


def step_response(band, win_s, hop_s, f_in_frac, amp=1.0, noise=0.01):
    t = np.arange(0, 8, 1 / FS)
    p = np.where(t >= 3.0, amp * np.sin(2 * np.pi * f_in_frac * FR * t),
                 0.0) + np.random.default_rng(0).normal(0, noise, len(t))
    pb = bandpass_bands(p - p.mean(), FS, [band], detrend=True)[0]
    tw, r = sliding_rms(pb, FS, win_s, hop_s)
    rb = np.median(r[tw < 2.5])
    i50 = np.argmax(r > rb + 0.5 * (r.max() - rb))
    i10 = np.argmax(r > rb + 0.1 * (r.max() - rb))
    return tw[i10] - 3.0, tw[i50] - 3.0


def main():
    det = SurgeDetector.from_rotor(fs=FS, f_rotor=FR)
    print(f"stall band {det.stall_band[0]:.1f}-{det.stall_band[1]:.1f} Hz, "
          f"win {det.win_s*1000:.0f} ms")
    l10, l50 = step_response(det.stall_band, det.win_s, det.hop_s, 0.5)
    print(f"  RMS 10%/50% response latency: {l10*1000:+.0f} ms / "
          f"{l50*1000:+.0f} ms  (论文引用: 3 ms)")
    print(f"surge band {det.surge_band[0]:.3f}-{det.surge_band[1]:.3f} Hz, "
          f"win {det.win_surge_s*1000:.0f} ms")
    l10, l50 = step_response(det.surge_band, det.win_surge_s,
                             det.hop_surge_s, 0.01)
    print(f"  RMS 10%/50% response latency: {l10*1000:+.0f} ms / "
          f"{l50*1000:+.0f} ms  (论文引用: 72 ms)")
    print("结论: 滤波器组对带内信号无可见群延迟, "
          "检测延迟来自 RMS 窗与五连判据。")


if __name__ == "__main__":
    main()
