# -*- coding: utf-8 -*-
"""
P1-5 自动阈值整定 (POT/GPD 极值理论) —— 消除人工阈值 K=1.6

方法 (Peaks-Over-Threshold):
  1. 26 个稳态点的融合统计量 pooled -> 越限样本 (u0 = 90 分位)
  2. GPD (广义 Pareto) MLE 拟合越限分布
  3. 按目标虚警率反解报警阈值 K_auto:
     稳态窗总越限概率 p_target = 1e-4 (期望 <2 个虚警窗 / 全部稳态数据)
  4. 验证: (a) 26 稳态点 sustained-5 虚警 episode 数 (期望 ~0)
           (b) 双转速 stall run 报警时刻 vs K=1.6 人工阈值

产出: results/auto_threshold.png / auto_threshold.csv
用法: python src/auto_threshold.py
"""
import os
import sys
import glob
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import genpareto

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tuda_adapter import read_raw_mat, decimate
import test_tuda_stall as tts
from false_alarm_eval import eval_point, ROOT, STEADY_DIR
from detector import fft_bandpass, sliding_rms

P_TARGET = 1e-4          # 单窗总越限概率目标
U0_Q = 0.90              # POT 初始阈值分位
N_SUSTAIN = tts.N_SUSTAIN_FUSION


def sustained_episode(fus, K, n=5):
    """返回 (有无 episode, 首个 episode 起始时刻索引)"""
    exc = fus > K
    cnt = 0
    for i, e in enumerate(exc):
        cnt = cnt + 1 if e else 0
        if cnt >= n:
            return True, i - n + 1
    return False, None


def main():
    files = sorted(glob.glob(os.path.join(STEADY_DIR, "*.mat")))
    print("=" * 76)
    print(f"P1-5 POT/GPD 自动阈值: {len(files)} 稳态点 pooled")
    recs = [eval_point(f) for f in files]
    pooled = np.concatenate([r["fus"] for r in recs])
    n_total = len(pooled)
    print(f"  融合统计量样本: {n_total} 窗 (每点 {len(recs[0]['fus'])} 窗)")

    u0 = np.quantile(pooled, U0_Q)
    exc = pooled[pooled > u0] - u0
    c, loc, scale = genpareto.fit(exc, floc=0)
    print(f"  u0 (90 分位) = {u0:.3f}, 越限样本 {len(exc)}, "
          f"GPD: c={c:.3f}, scale={scale:.3f}")

    # ---- 阈值族扫描 (POT/GPD 反解) + 操作点选择 ----
    GT = {"N100": 2.675, "N65": 5.075}
    stall_fus = {}
    for line, fname in (("N100", "Stall_Speed100.16_Kulite_data.mat"),
                        ("N65", "Stall_Speed65.03_Kulite_data.mat")):
        ep = read_raw_mat(os.path.join(ROOT, "tuda_data", "open_source", fname))
        P_ds, fs_ds = decimate(ep["P"], ep["fs"], tts.FS_DS)
        tw, Rn, fus, _ = tts.channel_rms_norm(P_ds, fs_ds, ep["f_rotor"])
        stall_fus[line] = (tw, fus)
    print(f"\n阈值族扫描 (GPD 反解, sustained-{N_SUSTAIN}):")
    rate_u0 = np.mean(pooled > u0)
    print(f"  {'p_target':>9} {'K':>7} {'稳态FA':>6} "
          f"{'N100报警':>9} {'N65报警':>9}  可用")
    rows = []
    for p_t in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4):
        p_cond = p_t / rate_u0
        K = float(u0 + genpareto.ppf(1 - p_cond, c, loc=0, scale=scale))
        n_ep = sum(sustained_episode(r["fus"], K, N_SUSTAIN)[0] for r in recs)
        alarms = {}
        for line in ("N100", "N65"):
            tw, fus = stall_fus[line]
            has, i0 = sustained_episode(fus, K, N_SUSTAIN)
            alarms[line] = tw[i0] if has else None
        ok = (n_ep == 0)   # 操作点判据: 零稳态虚警 (报警相对 GT 的早晚如实报告)
        rows.append(dict(p=p_t, K=K, fa=n_ep, **alarms, ok=ok))
        fmt = lambda t: f"{t:.2f}" if t else "None"
        print(f"  {p_t:>9} {K:>7.3f} {n_ep:>6} {fmt(alarms['N100']):>9} "
              f"{fmt(alarms['N65']):>9}  {'YES' if ok else ''}")
    # 选择: 零稳态虚警前提下最早报警 (即满足 0-FA 的最小 K)
    ok_rows = [r for r in rows if r["ok"]]
    sel = min(ok_rows, key=lambda r: min(r["N100"] or 99, r["N65"] or 99)) \
        if ok_rows else None
    K_auto = sel["K"] if sel else rows[-1]["K"]
    P_TARGET = sel["p"] if sel else 1e-4
    print(f"  -> 选择操作点: p_target={P_TARGET}, K_auto={K_auto:.3f}")
    K_emp = float(np.quantile(pooled, 1 - P_TARGET))
    print(f"  经验分位交叉验证: K_emp = {K_emp:.3f}")

    # ---- 对比: 人工阈值 vs 自动阈值 ----
    fa_manual = sum(sustained_episode(r["fus"], 1.6, N_SUSTAIN)[0] for r in recs)
    fa_auto = sel["fa"] if sel else rows[-1]["fa"]
    print(f"\n对比: 稳态虚警 episode 人工K=1.6: {fa_manual} -> "
          f"K_auto={K_auto:.2f}: {fa_auto} (208s 监测)")
    stall_res = {}
    for line in ("N100", "N65"):
        tw, fus = stall_fus[line]
        has, i0 = sustained_episode(fus, 1.6, N_SUSTAIN)
        stall_res[line] = {"K_manual(1.6)": tw[i0] if has else None,
                           "K_auto": sel[line] if sel else None}
        print(f"  [{line}] GT={GT[line]:.3f}s  K_manual 报警="
              f"{stall_res[line]['K_manual(1.6)'] and round(stall_res[line]['K_manual(1.6)'],3)}"
              f"  K_auto 报警="
              f"{stall_res[line]['K_auto'] and round(stall_res[line]['K_auto'],3)}")

    # ---- 保存 ----
    out_csv = os.path.join(ROOT, "results", "auto_threshold.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "value"])
        w.writerow(["u0_q90", f"{u0:.4f}"])
        w.writerow(["gpd_c", f"{c:.4f}"])
        w.writerow(["gpd_scale", f"{scale:.4f}"])
        w.writerow(["n_windows_pooled", n_total])
        w.writerow(["K_manual", "1.6"])
        w.writerow(["steady_FA_episodes_manual", fa_manual])
        w.writerow(["selected_p_target", P_TARGET])
        w.writerow(["K_auto", f"{K_auto:.4f}"])
        w.writerow(["steady_FA_episodes_auto", fa_auto])
        for r in rows:
            w.writerow([f"sweep_p={r['p']}", f"K={r['K']:.3f}, "
                        f"FA={r['fa']}, N100={r['N100']}, N65={r['N65']}, "
                        f"ok={r['ok']}"])
        for line in ("N100", "N65"):
            for k, v in stall_res[line].items():
                w.writerow([f"stall_{line}_{k}", f"{v:.3f}" if v else "None"])
    print(f"\nCSV 已保存: {out_csv}")

    # ---- 图 ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    xs = np.linspace(0, exc.max(), 200)
    ax.hist(exc, bins=40, density=True, alpha=0.6, label="exceedances")
    ax.plot(xs, genpareto.pdf(xs, c, loc=0, scale=scale), "r-",
            label=f"GPD fit (c={c:.2f})")
    ax.axvline(K_auto - u0, color="k", ls="--",
               label=f"K_auto-u0={K_auto-u0:.2f}")
    ax.set_xlabel("exceedance over u0 (fusion statistic)")
    ax.set_ylabel("pdf")
    ax.set_title("POT/GPD fit on pooled steady data (26 pts)")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    x = np.arange(2)
    wbar = 0.35
    ax.bar(x - wbar / 2, [stall_res[l]["K_manual(1.6)"] or 0 for l in ("N100", "N65")],
           wbar, color="#888", label="K_manual=1.6")
    ax.bar(x + wbar / 2, [stall_res[l]["K_auto"] or 0 for l in ("N100", "N65")],
           wbar, color="#c43b3b", label=f"K_auto={K_auto:.2f}")
    for i, l in enumerate(("N100", "N65")):
        ax.axhline(GT[l], xmin=0, xmax=1, color="green", ls=":", lw=0.8)
        ax.text(i + 0.42, GT[l] + 0.05, f"GT {GT[l]:.2f}s", color="green",
                fontsize=8, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(["N100", "N65"])
    ax.set_ylabel("alarm time [s]")
    ax.set_title("Stall-run alarm: manual vs auto threshold")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(ROOT, "results", "auto_threshold.png")
    fig.savefig(out_png, dpi=150)
    print(f"图已保存: {out_png}")


def read_gt(line):
    fname = ("Stall_Speed100.16_Kulite_data.mat" if line == "N100"
             else "Stall_Speed65.03_Kulite_data.mat")
    ep = read_raw_mat(os.path.join(ROOT, "tuda_data", "open_source", fname))
    P_ds, fs_ds = decimate(ep["P"], ep["fs"], 2000.0)
    return pressure_gt(P_ds, fs_ds, ep["f_rotor"])


def pressure_gt(P_ds, fs_ds, f_rotor):
    from tuda_adapter import pressure_fluctuation_criterion
    return pressure_fluctuation_criterion(
        dict(t=np.arange(P_ds.shape[0]) / fs_ds, P=P_ds, p=P_ds[:, 0],
             fs=fs_ds, f_rotor=f_rotor), k_sigma=6.0, win_s=0.05)


if __name__ == "__main__":
    main()
