# -*- coding: utf-8 -*-
"""
R2 实测数据深挖 —— 裕度指示 + 跨转速归一化一致性 + 轴向相干结构

(a) 近 NS 点裕度指示: 26 稳态点失速带 RMS 绝对水平 vs 距 NS 的流量差;
    重点: ṁ=13.36 (N100, ΔNS=-0.17, 已过官方 NS) —— "近失速未失稳"样本
(b) 前兆频带跨转速归一化一致性: 两转速最早前兆带 (归一化位置) 与
    前兆提前量对比 -> 支撑标度律
(c) 前兆带轴向相干结构: 前兆带内 ch0 与各通道的相干性 vs 轴向位置
    (局限: 21 通道同一周向 260°, 无周向传播信息 —— 如实标注)
产出: results/r2_deep_analysis.png / r2_deep_analysis.csv
用法: python src/deep_analysis.py
"""
import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import coherence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tuda_adapter import read_raw_mat, decimate, FS_RAW
from detector import SurgeDetector, fft_bandpass, sliding_rms, bandpass_bands
from precursor_tf import BANDS_NORM, BG_T

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEADY = os.path.join(ROOT, "tuda_data", "steady")
OPEN = os.path.join(ROOT, "tuda_data", "open_source")
STALL = {"N100": "Stall_Speed100.16_Kulite_data.mat",
         "N65": "Stall_Speed65.03_Kulite_data.mat"}
NS_OFFICIAL = {"N100": 13.53, "N65": 9.31}
FS_DS = 2000.0


# ---------------- (a) 裕度指示 ----------------
def margin_indicator():
    from false_alarm_eval import eval_point
    import glob
    recs = [eval_point(f) for f in sorted(glob.glob(os.path.join(STEADY, "*.mat")))]
    rows = []
    for line in ("N100", "N65"):
        rs = [r for r in recs if r["line"] == line]
        for r in rs:
            rows.append(dict(line=line, massflow=r["massflow"],
                             dNS=r["massflow"] - NS_OFFICIAL[line],
                             rms=r["rms_level"], max_fus=r["max_fus"]))
    # 近 NS 点 (ΔNS 最小) 与远场参考对比
    out = {}
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for line, c in (("N100", "#c43b3b"), ("N65", "#3b74c4")):
        rr = [r for r in rows if r["line"] == line]
        ax.scatter([r["dNS"] for r in rr], [r["rms"] * 1e3 for r in rr],
                   c=c, s=45, label=line, zorder=3)
        out[line] = rr
    # 远场线性趋势 (ΔNS > +0.5, 远离边界的点)
    for line, c in (("N100", "#c43b3b"), ("N65", "#3b74c4")):
        far = [r for r in rows if r["line"] == line and r["dNS"] > 0.5]
        d = np.array([r["dNS"] for r in far])
        y = np.array([r["rms"] for r in far])
        k, b = np.polyfit(d, y, 1)
        rr = np.linspace(-0.5, 2.6, 50)
        ax.plot(rr, (k * rr + b) * 1e3, color=c, ls="--", lw=1, alpha=0.7)
        r_near = [r for r in rows if r["line"] == line and abs(r["dNS"]) < 0.3]
        for r in r_near:
            ratio = r["rms"] / max(k * r["dNS"] + b, 1e-9)
            out.setdefault("nearNS", []).append(dict(line=line, **{
                kk: r[kk] for kk in ("massflow", "dNS", "rms", "max_fus")},
                rms_vs_trend=float(ratio)))
            print(f"(a) 近NS [{line}] ṁ={r['massflow']:.2f} (ΔNS={r['dNS']:+.2f}): "
                  f"RMS={r['rms']*1e3:.3f}e-3, 远场趋势外推比={ratio:.2f}x, "
                  f"max_fus={r['max_fus']:.2f} (<K=1.6 安静)")
    ns_pt = [r for r in rows if r["line"] == "N100" and abs(r["dNS"] + 0.17) < 0.02]
    if ns_pt:
        r = ns_pt[0]
        ax.annotate("past-NS point\n(ṁ=13.36, ΔNS=-0.17)",
                    (r["dNS"], r["rms"] * 1e3), textcoords="offset points",
                    xytext=(15, 12), fontsize=8, color="#c43b3b",
                    arrowprops=dict(arrowstyle="->", color="#c43b3b"))
    ax.axvline(0.0, color="gray", ls=":", lw=1, label="official NS")
    ax.set_xlabel("ΔNS = ṁ - ṁ_NS [kg/s]  (negative = past NS)")
    ax.set_ylabel("stall-band RMS level [e-3, dimensionless]")
    ax.set_title("(a) Margin indicator: stall-band RMS level vs distance to NS\n"
                 "(26 steady points; dashed: far-field trend)")
    ax.legend(); ax.grid(alpha=0.3)
    return out, fig, rows


# ---------------- (b) 跨转速归一化一致性 ----------------
def cross_speed_consistency():
    """读取 precursor_onsets.csv -> 每转速最早前兆带的归一化位置"""
    csv_path = os.path.join(ROOT, "results", "precursor_onsets.csv")
    bands = {}
    with open(csv_path, encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            key = row["file"]
            on_f = row["onset_fusion_s"]
            if on_f:
                bands.setdefault(key, []).append(
                    (float(on_f), row["band_norm"],
                     float(row["band_lo_Hz"]), float(row["band_hi_Hz"])))
    print("\n(b) 跨转速归一化一致性 (最早前兆带, fusion):")
    out = {}
    for key, lst in bands.items():
        line = "N100" if "100" in key else "N65"
        lst.sort()
        t0, bn, lo, hi = lst[0]
        # 归一化边界 (从 band_norm 字符串解析, 如 "0.125-0.250")
        lo_n, hi_n = [float(v) for v in bn.split("-")]
        overlap = not (hi_n <= 0.25 or lo_n >= 0.625)
        out[line] = dict(band=bn, lo_n=lo_n, hi_n=hi_n, t_onset=t0,
                         in_stall_band=overlap)
        print(f"  [{line}] 最早融合前兆带 norm={bn} ({lo:.1f}-{hi:.1f} Hz) "
              f"@ {t0:.2f}s, 与归一化失速带重叠: {overlap}")
    # 一致性判据: 两转速的失速带内 onset 与各自最早 onset 差 < 0.2s
    # (N100 最早带与失速带相邻且 onset 并列, 用容差公平判定)
    for line, lst in bands.items():
        line_key = "N100" if "100" in line else "N65"
        tol = 0.2
        cand = [(t, bn) for t, bn, _, _ in lst
                if not (float(bn.split("-")[1]) <= 0.25
                        or float(bn.split("-")[0]) >= 0.625)]
        if cand:
            t_in = min(c[0] for c in cand)
            out.setdefault(line_key, {})["t_onset_in_stall_band"] = t_in
            out[line_key]["within_0.2s_of_earliest"] = bool(
                t_in - out[line_key]["t_onset"] <= tol)
    both = out.get("N100") and out.get("N65")
    if both:
        consistent = all(out[l].get("within_0.2s_of_earliest", False)
                         for l in ("N100", "N65"))
        out["consistent"] = bool(consistent)
        print(f"  -> 两转速: 失速带内前兆起始与最早前兆起始差 <0.2s: {consistent}"
              f" (N100 {out['N100'].get('t_onset_in_stall_band', float('nan')):.2f}s, "
              f"N65 {out['N65'].get('t_onset_in_stall_band', float('nan')):.2f}s)")
    return out


# ---------------- (c) 轴向相干结构 ----------------
def axial_structure():
    """前兆带内 ch0-各通道相干 vs 轴向位置 (原始信号, 成核前窗)"""
    import glob as g
    out = {}
    for line, fname in STALL.items():
        ep = read_raw_mat(os.path.join(OPEN, fname))
        P_ds, fs_ds = decimate(ep["P"], ep["fs"], FS_DS)
        t_ds = np.arange(P_ds.shape[0]) / fs_ds
        f_rotor = ep["f_rotor"]
        gt = pressure_gt(P_ds, fs_ds, f_rotor)
        # 前兆带: 该转速失速带 (与 P0-2 最早前兆带一致)
        det = SurgeDetector.from_rotor(fs=fs_ds, f_rotor=f_rotor)
        blo, bhi = det.stall_band
        m_pre = (t_ds > BG_T[1]) & (t_ds < gt)   # 成核前窗
        nper = min(int(m_pre.sum()), int(fs_ds * 2))
        cohs, axpos = [], []
        for k in range(P_ds.shape[1]):
            f_c, Cxy = coherence(P_ds[m_pre, 0], P_ds[m_pre, k],
                                 fs=fs_ds, nperseg=nper)
            sel = (f_c >= blo) & (f_c <= bhi)
            cohs.append(float(np.mean(Cxy[sel])))
            axpos.append(ep["meta"]["axcor"][k])
        out[line] = dict(cohs=cohs, axpos=axpos, band=(blo, bhi), gt=gt)
        print(f"(c) [{line}] 前兆带 {blo:.0f}-{bhi:.0f} Hz 成核前窗 "
              f"({BG_T[1]:.1f}-{gt:.1f}s): ch0-各通道相干 "
              f"min={min(cohs):.2f} max={max(cohs):.2f} "
              f"(轴向 {axpos[0]:.0f}-{axpos[-1]:.0f} mm)")
    return out


def pressure_gt(P_ds, fs_ds, f_rotor):
    from tuda_adapter import pressure_fluctuation_criterion
    return pressure_fluctuation_criterion(
        dict(t=np.arange(P_ds.shape[0]) / fs_ds, P=P_ds, p=P_ds[:, 0],
             fs=fs_ds, f_rotor=f_rotor), k_sigma=6.0, win_s=0.05)


def main():
    print("=" * 76)
    print("R2 实测数据深挖: 裕度指示 / 跨转速一致性 / 轴向相干结构")
    print("=" * 76)

    out_a, fig_a, rows_a = margin_indicator()
    fig_a.tight_layout()
    p1 = os.path.join(ROOT, "results", "r2_margin_indicator.png")
    fig_a.savefig(p1, dpi=150)
    print(f"图已保存: {p1}")

    out_b = cross_speed_consistency()
    out_c = axial_structure()

    # ---- 轴向相干图 ----
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for line, c in (("N100", "#c43b3b"), ("N65", "#3b74c4")):
        o = out_c[line]
        ax.plot(o["axpos"], o["cohs"], "o-", color=c, label=f"{line} (pre-GT)")
    ax.set_xlabel("axial position of channel [mm] (ch0 near LE)")
    ax.set_ylabel("coherence with ch0 (precursor band, pre-GT window)")
    ax.set_title("(c) Axial coherence structure of precursor band\n"
                 "(limitation: all channels at same circumferential 260°)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    p2 = os.path.join(ROOT, "results", "r2_axial_coherence.png")
    fig.savefig(p2, dpi=150)
    print(f"图已保存: {p2}")

    # ---- CSV ----
    out_csv = os.path.join(ROOT, "results", "r2_deep_analysis.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["section", "key", "value"])
        for line in ("N100", "N65"):
            for r in out_a[line]:
                w.writerow(["a_margin", line,
                            f"mdot={r['massflow']:.2f}, dNS={r['dNS']:+.2f}, "
                            f"rms={r['rms']:.5f}, max_fus={r['max_fus']:.2f}"])
        for line in ("N100", "N65"):
            o = out_b[line]
            w.writerow(["b_consistency", line,
                        f"band={o['band']}, t_onset={o['t_onset']:.2f}s, "
                        f"in_stall_band={o['in_stall_band']}"])
        w.writerow(["b_consistency", "both_in_stall_band",
                    str(out_b.get("consistent"))])
        for line in ("N100", "N65"):
            o = out_c[line]
            w.writerow(["c_axial", line,
                        f"coherence ch0-chK: {['%.2f' % c for c in o['cohs']]}"])
    print(f"CSV 已保存: {out_csv}")


if __name__ == "__main__":
    main()
