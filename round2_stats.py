# -*- coding: utf-8 -*-
"""第二轮审稿 R4-2/N4-1 补强:
A. 报警时刻残差块自助的块长敏感性 (blk = 100/200/400, N100 线)
B. critical lead 的快阀速率 (rate=3.0) 扫描 (x_m = 1.10/1.15, 5 seeds x 2 B)
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tuda_adapter import read_raw_mat, decimate, pressure_fluctuation_criterion
from test_tuda_stall import channel_rms_norm, FS_DS
from modal_closed_loop import run_scenario
import mg_modal as mm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPEN = os.path.join(ROOT, "tuda_data", "open_source")
N_NOISE = 50


def alarm_ci_block(blk, line, fname, n_rep=N_NOISE):
    ep = read_raw_mat(os.path.join(OPEN, fname))
    P_ds, fs_ds = decimate(ep["P"], ep["fs"], FS_DS)
    gt = pressure_fluctuation_criterion(
        dict(t=np.arange(P_ds.shape[0]) / fs_ds, P=P_ds, p=P_ds[:, 0],
             fs=fs_ds, f_rotor=ep["f_rotor"]), k_sigma=6.0, win_s=0.05)
    from scipy.ndimage import uniform_filter1d
    smooth = uniform_filter1d(P_ds, size=51, axis=0, mode="nearest")
    resid = P_ds - smooth
    rng = np.random.default_rng(2026)
    alarms = []
    i_bg0, i_bg1 = int(0.3 * fs_ds), int(1.5 * fs_ds)
    bg_len = i_bg1 - i_bg0
    T = P_ds.shape[0]
    n_blk = int(np.ceil(T / blk))
    for rep in range(n_rep):
        starts = i_bg0 + rng.integers(0, bg_len - blk + 1, n_blk)
        idx = np.concatenate([(starts[b] + np.arange(blk)) % bg_len + i_bg0
                              for b in range(n_blk)])[:T]
        P_p = P_ds + resid[idx]
        _, _, fus, alarm = channel_rms_norm(P_p, fs_ds, ep["f_rotor"])
        if alarm is not None:
            alarms.append(alarm)
    alarms = np.array([a for a in alarms if a is not None], dtype=float)
    lo, hi = np.percentile(alarms, [2.5, 97.5])
    return len(alarms), float(alarms.mean()), float(lo), float(hi), gt


def main():
    print("=" * 76)
    print("A. 残差块自助块长敏感性 (N100, K_auto 管线, rep={})".format(N_NOISE))
    fname = "Stall_Speed100.16_Kulite_data.mat"
    for blk in (100, 200, 400):
        n, mean, lo, hi, gt = alarm_ci_block(blk, "N100", fname)
        print(f"  blk={blk:3d} ({blk/2000.0*1000:.0f}ms): rep={n}/{N_NOISE}, "
              f"报警 {mean:.3f}s, 95% CI [{lo:.3f}, {hi:.3f}]s, "
              f"CI 宽度 {hi-lo:.3f}s (GT {gt:.3f}s)")

    print("\nB. critical lead 快阀速率 (rate=3.0, 5 seeds x 2 scenarios)")
    for x_m in (1.10, 1.15):
        mm.MODAL_PARAMS["x_m"] = x_m
        leads = []
        for label, B, fr in (("B=0.42", 0.42, 333.9), ("B=0.65", 0.65, 333.9)):
            for seed in (1, 2, 3):
                r = run_scenario(B, False, seed, fr, 3.0)
                if r["alarm_t"] and r["t_stall"]:
                    leads.append(r["t_stall"] - r["alarm_t"])
        lead_mean = float(np.mean(leads)) if leads else float("nan")
        ok, tot = 0, 0
        for label, B, fr in (("B=0.42", 0.42, 333.9), ("B=0.65", 0.65, 333.9)):
            for seed in range(1, 6):
                r = run_scenario(B, True, seed, fr, 3.0)
                tot += 1
                ok += int(r["avoided"])
        print(f"  x_m={x_m:.2f} lead={lead_mean:.2f}s (rate=3.0): "
              f"避喘 {ok}/{tot}")


if __name__ == "__main__":
    main()
