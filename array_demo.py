# -*- coding: utf-8 -*-
"""
周向阵列处理演示: 相干性闸门区分失速/干扰 + 失速团传播速度估计

生成 results/array_coherence_demo.png (README 中引用)。
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from array_processing import (synthesize_array, coherence_gated_alarm,
                              fft_bandpass, windowed_speed)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")

CASES = [
    ("episode_026_rotating_stall.npz", "tab:red",
     "rotating stall"),
    ("episode_049_stable.npz", "tab:blue",
     "stable + EMI (single-channel false alarm)"),
]
BETA_TRUE = 0.45
COH_THR = 0.4


def main():
    idx = pd.read_csv(os.path.join(DATA, "index.csv"))
    fig, ax = plt.subplots(2, 1, figsize=(11, 6.5))
    print(f"{'episode':<34} {'alarm':>8} {'confirmed':>10} {'beta_med':>9} {'coh_med':>8}")
    for fn, color, label in CASES:
        r = idx[idx.file == fn].iloc[0]
        d = np.load(os.path.join(DATA, fn), allow_pickle=True)
        arr = synthesize_array(d, n_sensors=8, seed=7)
        fs, fr = arr["fs"], arr["f_rotor"]

        g = coherence_gated_alarm(arr["P"], fs, fr)
        sb = tuple(s * fr for s in (0.25, 0.625))
        Pb = np.column_stack([fft_bandpass(arr["P"][:, k] - arr["P"][:, k].mean(),
                                           fs, *sb) for k in range(8)])
        sp = windowed_speed(Pb, fs, fr, win_s=0.2, hop_s=0.1)
        m = sp["corr"] > 0.5
        beta_med = np.nanmedian(sp["beta"][m])

        ax[0].plot(g["t_coh"], g["coherence"], color=color, lw=1.2, label=label)
        ax[1].plot(sp["t"], sp["beta"], color=color, lw=1, label=label)
        if g["alarm_time"] is not None:
            ax[0].axvline(g["alarm_time"], color=color, ls="--", lw=1)
        print(f"{fn:<34} {str(g['alarm_time']):>8} "
              f"{str(g['confirmed']):>10} {beta_med:>9.3f} "
              f"{np.median(g['coherence']):>8.2f}")

    ax[0].axhline(COH_THR, color="k", ls=":", lw=1)
    ax[0].text(0.05, COH_THR + 0.03, "confirm threshold", fontsize=8)
    ax[0].set_ylabel("stall-band spatial coherence")
    ax[0].set_ylim(0, 1.05)
    ax[0].legend(fontsize=9, loc="center right")
    ax[0].set_title("Coherence-gated alarm: coherent rotating stall "
                    "vs incoherent interference")
    ax[1].axhline(BETA_TRUE, color="k", ls=":", lw=1, label="true $\\beta$=0.45")
    ax[1].set_ylabel("est. pattern speed $\\beta$")
    ax[1].set_ylim(0, 1)
    ax[1].set_xlabel("t [s]")
    ax[1].legend(fontsize=9)
    fig.tight_layout()
    fp = os.path.join(RESULTS, "array_coherence_demo.png")
    fig.savefig(fp, dpi=120)
    print("saved:", fp)


if __name__ == "__main__":
    main()
