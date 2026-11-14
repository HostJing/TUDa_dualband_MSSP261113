# -*- coding: utf-8 -*-
"""
P0-2 实测前兆时频搜索 —— 成核前低频/模态波前兆的系统排查

动机: 检测器实测提前量 0.29s (相对包络跳变 GT), 仿真中低频前兆可提前 2.4s。
本脚本在 TUDa 双转速 stall run 上系统回答: 成核前是否存在可检测前兆?
若有, 提前量能到多少; 若无, 则属于 spike-type 成核 (类型自适应卖点)。

方法 (双转速 stall run, ch0 近前缘 + 21 通道融合):
  1. 分带能量扫描: 9 个归一化窄带 (0.001x ~ 0.625x f_rotor, 对数网格),
     WOLA 带通 + 0.3s 滑动 RMS, 背景 (0.4-1.5s) z-score 显著性,
     持续 3 hop (0.3s) 超阈 z>3 记为 "该带前兆起始时刻"
  2. ch0 时频瀑布图 (STFT 0-400 Hz) —— 成核前后频谱结构
  3. 最早期前兆带的 21 通道相干性 (modal wave 应跨通道相干且相位随
     轴向位置线性变化; spike 局部)

产出: results/precursor_tf_search.png / precursor_onsets.csv
用法: python src/precursor_tf.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, coherence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector import SurgeDetector, bandpass_bands, sliding_rms
from tuda_adapter import (read_raw_mat, decimate,
                          pressure_fluctuation_criterion, FS_RAW)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "tuda_data", "open_source")
STALL_FILES = ["Stall_Speed100.16_Kulite_data.mat",
               "Stall_Speed65.03_Kulite_data.mat"]
FS_DS = 2000.0
# 归一化窄带网格 (× f_rotor): 覆盖喘振带 (0.001-0.03) 到失速带上沿
BANDS_NORM = [(0.001, 0.002), (0.002, 0.005), (0.005, 0.01),
              (0.01, 0.02), (0.02, 0.03), (0.03, 0.06),
              (0.06, 0.125), (0.125, 0.25), (0.25, 0.625)]
BG_T = (0.4, 1.5)          # 背景窗 (s)
WIN_ENS = 0.30             # 带内 RMS 窗 (s)
HOP_ENS = 0.10             # 步长 (s)
Z_TH = 3.0                 # 显著性阈值
N_SUSTAIN = 3              # 持续 hop 数


def onset_time(z, tw):
    """首个 持续 N_SUSTAIN hop z>Z_TH 的时刻; 无 -> None"""
    exc = z > Z_TH
    cnt = 0
    for i, e in enumerate(exc):
        cnt = cnt + 1 if e else 0
        if cnt >= N_SUSTAIN:
            return tw[i - N_SUSTAIN + 1]
    return None


def analyze(fname):
    print("=" * 76)
    print(f"文件: {fname}")
    ep = read_raw_mat(os.path.join(DATA_DIR, fname))
    m = ep["meta"]
    P_ds, fs_ds = decimate(ep["P"], ep["fs"], FS_DS)
    t_ds = np.arange(P_ds.shape[0]) / fs_ds
    f_rotor = ep["f_rotor"]
    print(f"  ṁ={m['massflow']:.2f} kg/s, {m['speed_pct']:.1f}%N, "
          f"f_rotor={f_rotor:.1f} Hz")

    ep_ds = dict(t=t_ds, P=P_ds, p=P_ds[:, 0], fs=fs_ds, f_rotor=f_rotor)
    gt = pressure_fluctuation_criterion(ep_ds, k_sigma=6.0, win_s=0.05)
    print(f"  GT (包络跳变): {gt:.3f}s" if gt else "  GT: None")

    # ---- 1. 分带能量扫描 (ch0 与 21 通道融合) ----
    bands = [(lo * f_rotor, hi * f_rotor) for lo, hi in BANDS_NORM]
    centers = [np.sqrt(lo * hi) for lo, hi in bands]
    onsets_ch0, onsets_fus, zpeak_pre = [], [], []
    fus_series = {}
    for bi, (blo, bhi) in enumerate(bands):
        Pb = np.stack(
            [bandpass_bands(P_ds[:, k] - P_ds[:, k].mean(), fs_ds,
                            [(blo, bhi)], detrend=True)[0]
             for k in range(P_ds.shape[1])], axis=1)
        tw, R = None, []
        for k in range(Pb.shape[1]):
            tw, r = sliding_rms(Pb[:, k], fs_ds, WIN_ENS, HOP_ENS)
            R.append(r)
        R = np.stack(R, axis=1)   # (n_win, 21)
        fus = np.median(R, axis=1)
        r0 = R[:, 0]
        bgm = (tw >= BG_T[0]) & (tw <= BG_T[1])
        rows = []
        for sig in (r0, fus):
            mu, sd = np.median(sig[bgm]), np.std(sig[bgm]) + 1e-12
            z = (sig - mu) / sd
            # 起始瞬态 (滤波边缘/流动建立) 不参与 onset 搜索
            z_onset = np.where(tw >= BG_T[1], z, -np.inf)
            rows.append((z, onset_time(z_onset, tw)))
            # 成核前窗内峰值 z
            if gt:
                pre = (tw > BG_T[1]) & (tw < gt)
                zpeak_pre.append(np.max(z[pre]) if pre.any() else 0.0)
        onsets_ch0.append(rows[0][1]); onsets_fus.append(rows[1][1])
        fus_series[bi] = (tw, rows[1][0])
        print(f"  band {blo:7.2f}-{bhi:7.2f} Hz "
              f"(norm {bands_norm_str(bi)}): "
              f"onset ch0={fmt_t(rows[0][1])} fusion={fmt_t(rows[1][1])} "
              f"zpeak_pre={zpeak_pre[-1]:.1f}")

    # ---- 2. ch0 时频瀑布图 ----
    f_sp, t_sp, Sxx = spectrogram(P_ds[:, 0] - P_ds[:, 0].mean(),
                                  fs_ds, nperseg=2048, noverlap=1536)
    sel = f_sp <= 400
    Sdb = 10 * np.log10(Sxx[sel] + 1e-12)

    # ---- 3. 最早期前兆带的跨通道相干 ----
    pre_band = None
    cand = [(ot, bi) for bi, ot in enumerate(onsets_fus) if ot is not None]
    if gt:
        cand += [(gt, bi) for bi, ot in enumerate(onsets_fus) if ot is None]
    pre_band = min(cand)[1] if cand else None
    coh_info = None
    if pre_band is not None and gt:
        blo, bhi = bands[pre_band]
        # 相干性用未滤波原始通道 (滤波会引入虚假相干)
        m_pre = (t_ds > BG_T[1]) & (t_ds < gt + 0.2)
        nper = min(int(m_pre.sum()), int(fs_ds * 2))
        cohs = []
        for k in range(1, P_ds.shape[1]):
            f_c, Cxy = coherence(P_ds[m_pre, 0], P_ds[m_pre, k],
                                 fs=fs_ds, nperseg=nper)
            selc = (f_c > blo) & (f_c < bhi)
            cohs.append(np.mean(Cxy[selc]))
        coh_info = dict(band=bands[pre_band],
                        mean_coh=float(np.mean(cohs)),
                        max_coh=float(np.max(cohs)))
        print(f"  前兆带 {bands[pre_band][0]:.2f}-{bands[pre_band][1]:.2f} Hz "
              f"跨通道相干 (原始信号, 成核前窗): mean={coh_info['mean_coh']:.2f}, "
              f"max={coh_info['max_coh']:.2f}")

    return dict(fname=fname, meta=m, f_rotor=f_rotor, gt=gt,
                centers=centers, onsets_ch0=onsets_ch0,
                onsets_fus=onsets_fus, fus_series=fus_series,
                Sdb=Sdb, t_sp=t_sp, f_sp=f_sp[sel],
                pre_band=pre_band, coh=coh_info, det=SurgeDetector)


def bands_norm_str(bi):
    lo, hi = BANDS_NORM[bi]
    return f"{lo:.3f}-{hi:.3f} fr"


def fmt_t(t):
    return f"{t:.2f}s" if t is not None else "--"


def main():
    results = [analyze(f) for f in STALL_FILES]

    fig, axes = plt.subplots(len(results), 2, figsize=(13, 4 * len(results)))
    for row, r in enumerate(results):
        ax = axes[row, 0]
        ax.imshow(r["Sdb"], aspect="auto", origin="lower",
                  extent=[r["t_sp"][0], r["t_sp"][-1],
                          r["f_sp"][0], r["f_sp"][-1]],
                  cmap="magma", vmin=-60, vmax=0)
        ax.set_yscale("log")
        if r["gt"]:
            ax.axvline(r["gt"], color="cyan", ls=":", lw=1.5)
            ax.text(r["gt"], 100, f" GT {r['gt']:.2f}s", color="cyan", fontsize=8)
        ax.set_ylim(0.5, 400)
        ax.set_ylabel("f [Hz]")
        ax.set_title(f"{r['fname']}: ch0 spectrogram "
                     f"(ṁ={r['meta']['massflow']:.2f}, "
                     f"{r['meta']['speed_pct']:.1f}%N)", fontsize=10)

        ax = axes[row, 1]
        for bi, (tw, z) in r["fus_series"].items():
            ax.plot(tw, z, lw=0.8, alpha=0.7,
                    label=f"{r['centers'][bi]:.1f}Hz" if bi in (2, 4, 6, 8) else None)
        ax.axhline(Z_TH, color="gray", ls="--", lw=1)
        if r["gt"]:
            ax.axvline(r["gt"], color="green", ls=":", lw=1.5, label="GT")
        ax.set_ylim(-3, 15)
        ax.set_xlabel("t [s]")
        ax.set_ylabel("band RMS z-score (fusion)")
        ax.set_title("per-band energy climb (21-ch fusion)", fontsize=10)
        ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3)

        # CSV
    out_csv = os.path.join(ROOT, "results", "precursor_onsets.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = f.write
        w("file,f_rotor,gt_s,band_norm,band_lo_Hz,band_hi_Hz,"
          "onset_ch0_s,onset_fusion_s,zpeak_pre\n")
        for r in results:
            for bi in range(len(BANDS_NORM)):
                lo, hi = BANDS_NORM[bi]
                w(f"{r['fname']},{r['f_rotor']:.2f},"
                  f"{r['gt'] if r['gt'] else ''},"
                  f"{lo:.3f}-{hi:.3f},{lo*r['f_rotor']:.2f},{hi*r['f_rotor']:.2f},"
                  f"{r['onsets_ch0'][bi] if r['onsets_ch0'][bi] else ''},"
                  f"{r['onsets_fus'][bi] if r['onsets_fus'][bi] else ''},\n")
    print(f"\nCSV 已保存: {out_csv}")

    # ---- 汇总: 最早前兆与提前量 ----
    print("\n==== 前兆提前量汇总 ====")
    for r in results:
        if r["gt"] is None:
            continue
        cands = [(ot, bi) for bi, ot in enumerate(r["onsets_fus"])
                 if ot is not None and ot < r["gt"]]
        if cands:
            ot, bi = min(cands)
            lead = r["gt"] - ot
            lo, hi = BANDS_NORM[bi]
            print(f"  {r['fname']}: 最早前兆 band {lo:.3f}-{hi:.3f}×fr "
                  f"({lo*r['f_rotor']:.1f}-{hi*r['f_rotor']:.1f} Hz) "
                  f"@ {ot:.2f}s -> 提前 GT {lead:.2f}s")
        else:
            print(f"  {r['fname']}: 成核前无显著带内前兆 (z>3 持续 0.3s) "
                  f"-> spike-type 主导证据")

    fig.tight_layout()
    out_png = os.path.join(ROOT, "results", "precursor_tf_search.png")
    fig.savefig(out_png, dpi=150)
    print(f"图已保存: {out_png}")


if __name__ == "__main__":
    main()
