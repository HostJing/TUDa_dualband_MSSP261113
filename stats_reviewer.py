# -*- coding: utf-8 -*-
"""
审稿修订统计补强 (R2-2 / R2-3 / R1-3 / R1-4):
  A. 分层 GPD: N100 与 N65 稳态点分别拟合, 比较形状/尺度参数;
  B. pooled GPD 参数 bootstrap 95% CI (超阈样本重采样 x500);
  C. u0 分位敏感性: q80/q90/q95 -> K@p=1e-3 与经验分位对照;
  D. 前兆 z>3 流程在 26 稳态点上的经验虚警 episode 数 (9 带, 多重比较);
  E. GT 判据参数敏感性: k_sigma x win_s 扫描 -> GT 移动范围;
  F. 相干估计的自由度与 95% 显著线 (Bendat & Piersol)。

产出: results/reviewer_stats.csv + 控制台汇总
"""
import os
import sys
import glob
import csv
import numpy as np
from scipy.stats import genpareto

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tuda_adapter import read_raw_mat, decimate, pressure_fluctuation_criterion
from false_alarm_eval import eval_point, ROOT, STEADY_DIR
from auto_threshold import U0_Q
from precursor_tf import (BANDS_NORM, BG_T, WIN_ENS, HOP_ENS, Z_TH, N_SUSTAIN,
                          onset_time)
from detector import SurgeDetector, bandpass_bands, sliding_rms

FS_DS = 1250.0
STALL = {"N100": "Stall_Speed100.16_Kulite_data.mat",
         "N65": "Stall_Speed65.03_Kulite_data.mat"}
N_BOOT = 500


def gpd_fit(exc):
    c, loc, scale = genpareto.fit(exc, floc=0)
    return float(c), float(scale)


def main():
    print("=" * 76)
    print("A/B/C. POT/GPD 统计补强")
    files = sorted(glob.glob(os.path.join(STEADY_DIR, "*.mat")))
    recs = []
    lines = []
    for f in files:
        r = eval_point(f)
        recs.append(r)
        lines.append("N100" if "N100" in os.path.basename(f) else "N65")
    pooled = np.concatenate([r["fus"] for r in recs])

    # A. 分层拟合
    layer = {}
    for ln in ("N100", "N65"):
        xs = np.concatenate([r["fus"] for r, l in zip(recs, lines) if l == ln])
        u = np.quantile(xs, U0_Q)
        c, s = gpd_fit(xs[xs > u] - u)
        layer[ln] = (c, s, len(xs))
        print(f"  [{ln}] n={len(xs)}, u0={u:.3f}, c={c:.3f}, scale={s:.3f}")

    # B. pooled bootstrap CI
    u0 = np.quantile(pooled, U0_Q)
    exc = pooled[pooled > u0] - u0
    rng = np.random.default_rng(0)
    cb, sb = [], []
    for _ in range(N_BOOT):
        b = rng.choice(exc, size=len(exc), replace=True)
        c, s = gpd_fit(b)
        cb.append(c); sb.append(s)
    c0, s0 = gpd_fit(exc)
    print(f"  pooled: n_exc={len(exc)}, c={c0:.3f} "
          f"CI95=[{np.quantile(cb,0.025):.3f},{np.quantile(cb,0.975):.3f}], "
          f"scale={s0:.3f} CI95=[{np.quantile(sb,0.025):.3f},"
          f"{np.quantile(sb,0.975):.3f}]")

    # C. u0 敏感性 @ p=1e-3
    rate_u0 = np.mean(pooled > u0)
    print("  u0 敏感性 (p_target=1e-3):")
    for q in (0.80, 0.90, 0.95):
        uq = np.quantile(pooled, q)
        e_q = pooled[pooled > uq] - uq
        c_q, s_q = gpd_fit(e_q)
        p_cond = 1e-3 / np.mean(pooled > uq)
        K_gpd = uq + genpareto.ppf(1 - p_cond, c_q, loc=0, scale=s_q)
        K_emp = float(np.quantile(pooled, 1 - 1e-3))
        print(f"    u0=q{int(q*100)}: u0={uq:.3f}, K_gpd={K_gpd:.3f}, "
              f"K_emp={K_emp:.3f}")

    # ---- D. 前兆 z>3 流程在稳态点上的经验虚警 ----
    print("\nD. 前兆 z 流程稳态虚警 (9 带 x 26 稳态点)")
    bands_norm = BANDS_NORM
    n_ep_total, n_seq_total = 0, 0
    per_band = [0] * len(bands_norm)
    for f in files:
        e = read_raw_mat(f)
        P_ds, fs_ds = decimate(e["P"], e["fs"], FS_DS)
        f_rotor = e["f_rotor"]
        for bi, (lo, hi) in enumerate(bands_norm):
            blo, bhi = lo * f_rotor, hi * f_rotor
            Pb = bandpass_bands(P_ds[:, 0] - P_ds[:, 0].mean(), fs_ds,
                                [(blo, bhi)], detrend=True)[0]
            tw, R = sliding_rms(Pb, fs_ds, WIN_ENS, HOP_ENS)
            bgm = (tw >= BG_T[0]) & (tw <= BG_T[1])
            mu, sd = np.median(R[bgm]), np.std(R[bgm]) + 1e-12
            z = (R - mu) / sd
            z = np.where(tw >= BG_T[1], z, -np.inf)
            if onset_time(z, tw) is not None:
                n_ep_total += 1
                per_band[bi] += 1
            n_seq_total += 1
    print(f"  z>3 持续 0.3s 的 episode: {n_ep_total}/{n_seq_total} "
          f"(带-记录序列); 分带: {per_band}")

    # ---- E. GT 判据参数敏感性 (与论文 GT 管线一致: fs=2000 Hz) ----
    print("\nE. GT (envelope-jump) 判据参数敏感性")
    gt_sens = {}
    for line, fname in STALL.items():
        e = read_raw_mat(os.path.join(ROOT, "tuda_data", "open_source", fname))
        P_ds, fs_ds = decimate(e["P"], e["fs"], 2000.0)
        gts = {}
        for ks in (4.0, 5.0, 6.0, 8.0):
            for ws in (0.02, 0.05, 0.10):
                g = pressure_fluctuation_criterion(
                    dict(t=np.arange(P_ds.shape[0]) / fs_ds, P=P_ds,
                         p=P_ds[:, 0], fs=fs_ds, f_rotor=e["f_rotor"]),
                    k_sigma=ks, win_s=ws)
                gts[(ks, ws)] = g
        vals = [v for v in gts.values() if v is not None]
        # 排除判据在背景期误触发 (过早) 的参数组合
        plaus = [v for v in vals if v > 1.0]
        lo, hi = min(plaus), max(plaus)
        ref = gts[(6.0, 0.05)]
        gt_sens[line] = (lo, hi, ref)
        n_reject = len(vals) - len(plaus)
        print(f"  [{line}] GT(k6,w0.05)={ref:.3f}s, "
              f"参数扫描范围 [{lo:.3f}, {hi:.3f}] s "
              f"(span {hi-lo:.3f}s, 排除 {n_reject} 个背景期误触发组合)")
        # 采样率敏感性
        P2, fs2 = decimate(e["P"], e["fs"], 1250.0)
        g1250 = pressure_fluctuation_criterion(
            dict(t=np.arange(P2.shape[0]) / fs2, P=P2, p=P2[:, 0],
                 fs=fs2, f_rotor=e["f_rotor"]), k_sigma=6.0, win_s=0.05)
        print(f"  [{line}] 降采样率敏感性: fs=2000 -> {ref:.3f}s, "
              f"fs=1250 -> {g1250:.3f}s (差 {ref-g1250:+.3f}s)")
        gt_sens[line] = (lo, hi, ref, g1250)

    # ---- F. 相干自由度与 95% 线 (与 deep_analysis 管线一致: fs=2000) ----
    print("\nF. 相干估计自由度与 95% 显著线")
    for line, fname in STALL.items():
        e = read_raw_mat(os.path.join(ROOT, "tuda_data", "open_source", fname))
        P_ds, fs_ds = decimate(e["P"], e["fs"], 2000.0)
        t_ds = np.arange(P_ds.shape[0]) / fs_ds
        ref, _, _, _ = gt_sens[line]
        gt = ref
        m_pre = (t_ds > BG_T[1]) & (t_ds < gt)
        N = int(m_pre.sum())
        nper = min(N, int(fs_ds * 2))
        if N <= nper:
            nper = N // 2
        K_seg = int(np.floor((N - nper) / (nper / 2))) + 1
        K_seg = max(K_seg, 2)
        gam95 = float(np.sqrt(1 - 0.05 ** (1.0 / (K_seg - 1))))
        print(f"  [{line}] 前兆窗 N={N} 采样, nperseg={nper}, "
              f"平均段数 K={K_seg} (dof={2*K_seg}), "
              f"95% 显著线 gamma95={gam95:.3f}")

    # ---- CSV ----
    out = os.path.join(ROOT, "results", "reviewer_stats.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "key", "value"])
        for ln in ("N100", "N65"):
            c, s, n = layer[ln]
            w.writerow(["layered_gpd", ln, f"c={c:.3f},scale={s:.3f},n={n}"])
        w.writerow(["pooled_gpd", "c", f"{c0:.3f}"])
        w.writerow(["pooled_gpd", "c_CI95",
                    f"[{np.quantile(cb,0.025):.3f},{np.quantile(cb,0.975):.3f}]"])
        w.writerow(["pooled_gpd", "scale", f"{s0:.3f}"])
        w.writerow(["pooled_gpd", "scale_CI95",
                    f"[{np.quantile(sb,0.025):.3f},{np.quantile(sb,0.975):.3f}]"])
        for q in (0.80, 0.90, 0.95):
            uq = np.quantile(pooled, q)
            e_q = pooled[pooled > uq] - uq
            c_q, s_q = gpd_fit(e_q)
            p_cond = 1e-3 / np.mean(pooled > uq)
            K_gpd = uq + genpareto.ppf(1 - p_cond, c_q, loc=0, scale=s_q)
            w.writerow(["u0_sens", f"q{int(q*100)}",
                        f"u0={uq:.3f},K_gpd={K_gpd:.3f},"
                        f"K_emp={np.quantile(pooled,1-1e-3):.3f}"])
        w.writerow(["precursor_FA_steady", "episodes",
                    f"{n_ep_total}/{n_seq_total}"])
        w.writerow(["precursor_FA_steady", "per_band",
                    ",".join(str(x) for x in per_band)])
        for line in ("N100", "N65"):
            lo, hi, ref, g1250 = gt_sens[line]
            w.writerow(["gt_sens", line,
                        f"ref={ref:.3f},range=[{lo:.3f},{hi:.3f}],span={hi-lo:.3f},"
                        f"fs1250={g1250:.3f}"])
    print(f"\nCSV 已保存: {out}")


if __name__ == "__main__":
    main()
