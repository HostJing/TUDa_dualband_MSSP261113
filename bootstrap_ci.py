# -*- coding: utf-8 -*-
"""
R4 统计稳健性 —— 关键数字的 bootstrap/扰动置信区间

1. 稳态虚警: 26 点层面 bootstrap (B=2000) + 零事件上界 (rule of three)
2. 报警时刻: 噪声扰动重采样 (残差 bootstrap, 100 次) -> 报警时刻 CI
3. 避喘成功率: 模态波闭环扩展到 10 seeds/场景 (rate=1.0), 4 分钟级
产出: results/bootstrap_ci.csv
用法: python src/bootstrap_ci.py
"""
import os
import sys
import csv
import glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tuda_adapter import read_raw_mat, decimate, pressure_fluctuation_criterion
from test_tuda_stall import channel_rms_norm, FS_DS
from detector import SurgeDetector
import mg_modal as mm
from modal_closed_loop import run_scenario

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEADY = os.path.join(ROOT, "tuda_data", "steady")
OPEN = os.path.join(ROOT, "tuda_data", "open_source")
STALL_FILES = {"N100": "Stall_Speed100.16_Kulite_data.mat",
               "N65": "Stall_Speed65.03_Kulite_data.mat"}
K_MANUAL, K_AUTO = 1.6, 2.195
B_BOOT = 2000
N_NOISE = 100


def sustained(fus, K, n=5):
    exc = fus > K
    cnt = 0
    for i, e in enumerate(exc):
        cnt = cnt + 1 if e else 0
        if cnt >= n:
            return True, i - n + 1
    return False, None


def part1_fa_bootstrap(recs):
    """26 点层面 bootstrap -> 每 K 的 FA episode 点数分布"""
    out = {}
    for K_name, K in (("K_manual(1.6)", K_MANUAL), ("K_auto(2.195)", K_AUTO)):
        flags = np.array([sustained(r["fus"], K)[0] for r in recs])
        n_pts, n_win = len(flags), sum(len(r["fus"]) for r in recs)
        rng = np.random.default_rng(2026)
        counts = np.empty(B_BOOT)
        for b in range(B_BOOT):
            idx = rng.integers(0, n_pts, n_pts)
            counts[b] = flags[idx].sum()
        lo, hi = np.percentile(counts, [2.5, 97.5])
        # 零事件上界 (rule of three): 3/n
        ub_pts = 3.0 / n_pts if flags.sum() == 0 else hi / n_pts
        ub_win = 3.0 / n_win if (K_name.startswith("K_auto")) else None
        out[K_name] = dict(obs=int(flags.sum()), n_pts=n_pts,
                           ci=(lo, hi), ub_pts_frac=ub_pts,
                           ub_win_rate=ub_win)
        print(f"1. FA bootstrap [{K_name}]: 观测 {int(flags.sum())}/{n_pts} 点, "
              f"bootstrap 95% CI = [{lo:.0f}, {hi:.0f}] 点, "
              f"零事件上界 (点占比) <= {ub_pts*100:.1f}%")
        if ub_win:
            print(f"   融合统计量窗层面: 0/{n_win} 越限 -> 单窗率上界 <= "
                  f"{ub_win:.2e} (rule of three)")
    return out


def part2_alarm_noise_ci():
    """残差 bootstrap -> 报警时刻 95% CI (融合统计量)"""
    out = {}
    for line, fname in STALL_FILES.items():
        ep = read_raw_mat(os.path.join(OPEN, fname))
        P_ds, fs_ds = decimate(ep["P"], ep["fs"], FS_DS)
        gt = pressure_fluctuation_criterion(
            dict(t=np.arange(P_ds.shape[0]) / fs_ds, P=P_ds, p=P_ds[:, 0],
                 fs=fs_ds, f_rotor=ep["f_rotor"]), k_sigma=6.0, win_s=0.05)
        # 残差: 高频分量 (边缘安全的滚动均值扣除, 51 样本 = 25ms)
        from scipy.ndimage import uniform_filter1d
        smooth = uniform_filter1d(P_ds, size=51, axis=0, mode="nearest")
        resid = P_ds - smooth
        rng = np.random.default_rng(2026)
        alarms = []
        # 残差块仅从平稳背景段 (0.3-1.5s) 采样 —— 回答 "背景噪声实现变化
        # 对报警时刻的影响"; 不能从全记录采样 (会混入失稳段能量, 残差
        # 强非平稳)
        i_bg0, i_bg1 = int(0.3 * fs_ds), int(1.5 * fs_ds)
        blk, T = 200, P_ds.shape[0]
        n_blk = int(np.ceil(T / blk))
        bg_len = i_bg1 - i_bg0
        for rep in range(N_NOISE):
            starts = i_bg0 + rng.integers(0, bg_len - blk + 1, n_blk)
            idx = np.concatenate([(starts[b] + np.arange(blk)) % bg_len + i_bg0
                                  for b in range(n_blk)])[:T]
            P_p = P_ds + resid[idx]
            _, _, fus, alarm = channel_rms_norm(P_p, fs_ds, ep["f_rotor"])
            if alarm is not None:
                alarms.append(alarm)
        alarms = np.array(alarms)
        lo, hi = np.percentile(alarms, [2.5, 97.5])
        lead_lo, lead_hi = gt - hi, gt - lo
        out[line] = dict(n=len(alarms), mean=float(alarms.mean()),
                         ci=(float(lo), float(hi)), gt=gt,
                         lead_ci=(float(lead_lo), float(lead_hi)))
        print(f"2. 噪声扰动 CI [{line}] (rep={len(alarms)}/{N_NOISE}): "
              f"报警 {alarms.mean():.3f}s, 95% CI [{lo:.3f}, {hi:.3f}]s "
              f"-> lead CI [{lead_lo:+.2f}, {lead_hi:+.2f}]s (GT={gt:.3f}s)")
    return out


def part3_avoidance_seeds():
    """模态波闭环 10 seeds/场景 (closed, rate=1.0) -> 成功率"""
    scenarios = [("B=0.42", 0.42, 333.9), ("B=0.65", 0.65, 333.9)]
    res = {}
    print("3. 避喘成功率扩展 (10 seeds/场景, closed, rate=1.0):")
    for label, B, fr in scenarios:
        ok_flags = []
        for seed in range(1, 11):
            r = run_scenario(B, True, seed, fr, 1.0)
            ok_flags.append(r["avoided"])
        n_ok = sum(ok_flags)
        # Wilson 95% CI for proportion
        n = len(ok_flags)
        z = 1.96
        p = n_ok / n
        wil = ((p + z * z / (2 * n) - z * np.sqrt(
            p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n),
               (p + z * z / (2 * n) + z * np.sqrt(
            p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n))
        res[label] = dict(n_ok=n_ok, n=n, ci=wil, ok_flags=ok_flags)
        print(f"   [{label}]: {n_ok}/{n} 避喘, Wilson 95% CI = "
              f"[{wil[0]*100:.0f}%, {wil[1]*100:.0f}%]")
    return res


def main():
    print("=" * 76)
    print("R4 统计稳健性: bootstrap / 噪声扰动 CI / 成功率扩展")
    print("=" * 76)

    print("\n[准备] 26 稳态点融合统计量...")
    from false_alarm_eval import eval_point
    recs = [eval_point(f) for f in sorted(glob.glob(os.path.join(STEADY, "*.mat")))]

    fa = part1_fa_bootstrap(recs)
    alarm_ci = part2_alarm_noise_ci()
    avoid = part3_avoidance_seeds()

    out_csv = os.path.join(ROOT, "results", "bootstrap_ci.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "key", "value"])
        for K_name, d in fa.items():
            w.writerow(["FA_bootstrap", K_name,
                        f"obs={d['obs']}/{d['n_pts']}, CI={d['ci']}, "
                        f"ub_frac={d['ub_pts_frac']:.3f}, "
                        f"ub_win_rate={d['ub_win_rate']}"])
        for line, d in alarm_ci.items():
            w.writerow(["alarm_noise_CI", line,
                        f"rep={d['n']}/{N_NOISE}, mean={d['mean']:.3f}s, "
                        f"CI=[{d['ci'][0]:.3f},{d['ci'][1]:.3f}]s, "
                        f"lead_CI=[{d['lead_ci'][0]:+.2f},{d['lead_ci'][1]:+.2f}]s"])
        for label, d in avoid.items():
            w.writerow(["avoidance", label,
                        f"{d['n_ok']}/{d['n']}, Wilson95%=[{d['ci'][0]*100:.0f}%,"
                        f"{d['ci'][1]*100:.0f}%], flags={d['ok_flags']}"])
    print(f"\nCSV 已保存: {out_csv}")


if __name__ == "__main__":
    main()
