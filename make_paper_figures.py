# -*- coding: utf-8 -*-
"""
论文主图生成 (Science/AAAS 配色, PDF 矢量, 输出到 paper/figures/)

配色 (ggsci AAAS/Science): #3B4992 蓝, #EE0000 红, #008B45 绿,
#631879 紫, #008280 青, #BB0021 深红, #5F559B 蓝紫
用法: python src/make_paper_figures.py   (~3 min)
"""
import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIGDIR, exist_ok=True)

C = dict(blue="#3B4992", red="#EE0000", green="#008B45", purple="#631879",
         teal="#008280", darkred="#BB0021", slate="#5F559B", gray="#666666")

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "lines.linewidth": 1.0, "axes.linewidth": 0.7,
    "font.family": "DejaVu Sans", "pdf.fonttype": 42,
})

SINGLE, DOUBLE = (3.5, 2.6), (7.0, 2.8)


def save(fig, name):
    fig.savefig(os.path.join(FIGDIR, name), bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# ---------- F1: 特性线拟合 ----------
def f1_characteristic():
    from fit_mg_characteristic import (load_measured_lines, inlet_density,
                                       convert_to_mg_plane, psi_cubic)
    data = load_measured_lines()
    rho = inlet_density(data["pt_in"], data["Tt_in"])
    Phi, Psi = convert_to_mg_plane(data["mdot_corr"], data["pi_total"], rho)
    CAL = dict(psi_c0=-1.259, H=0.778, W=0.178)
    fig, ax = plt.subplots(figsize=SINGLE)
    xs = np.linspace(0.33, 0.41, 100)
    ax.plot(xs, psi_cubic(xs, **CAL), color=C["blue"], label="calibrated MG cubic")
    ax.plot(Phi, Psi, "o", color=C["red"], ms=3.5, label="TUDa measured (N100)")
    ax.axvline(2 * CAL["W"], color=C["gray"], ls=":", lw=0.8)
    ax.annotate("peak ($\\Phi=2W$)", (2 * CAL["W"], ax.get_ylim()[0]),
                xytext=(3, 10), textcoords="offset points", fontsize=7)
    ax.set_xlabel("$\\Phi = c_x/U_{tip}$")
    ax.set_ylabel("$\\Psi = \\Delta p_0/(\\rho U_{tip}^2)$")
    ax.legend(frameon=False)
    save(fig, "fig_characteristic.pdf")


# ---------- F2: stall run 融合检测 ----------
def f2_detection():
    from tuda_adapter import read_raw_mat, decimate, pressure_fluctuation_criterion
    from test_tuda_stall import channel_rms_norm, FS_DS
    fig, axes = plt.subplots(2, 1, figsize=(SINGLE[0], 3.6), sharex=False)
    for ax, (line, fname) in zip(
            axes, [("N100", "Stall_Speed100.16_Kulite_data.mat"),
                   ("N65", "Stall_Speed65.03_Kulite_data.mat")]):
        ep = read_raw_mat(os.path.join(ROOT, "tuda_data", "open_source", fname))
        P_ds, fs_ds = decimate(ep["P"], ep["fs"], FS_DS)
        t_ds = np.arange(P_ds.shape[0]) / fs_ds
        gt = pressure_fluctuation_criterion(
            dict(t=t_ds, P=P_ds, p=P_ds[:, 0], fs=fs_ds,
                 f_rotor=ep["f_rotor"]), k_sigma=6.0, win_s=0.05)
        tw, Rn, fus, alarm = channel_rms_norm(P_ds, fs_ds, ep["f_rotor"])
        ax.plot(tw, fus, color=C["blue"], label="fusion statistic")
        ax.axhline(1.6, color=C["gray"], ls="--", lw=0.7,
                   label="$K=1.6$ (manual)")
        ax.axhline(2.195, color=C["darkred"], ls="-.", lw=0.7,
                   label="$K=2.195$ (auto)")
        if alarm:
            ax.axvline(alarm, color=C["red"], ls="--", lw=0.9,
                       label=f"alarm $K$=1.6: {alarm:.2f} s")
        # 自动阈值报警时刻 (results/auto_threshold.csv)
        a_auto = {"N100": 2.801, "N65": 5.050}[line]
        ax.axvline(a_auto, color=C["darkred"], ls="-.", lw=0.9,
                   label=f"alarm $K$=2.195: {a_auto:.2f} s")
        if gt:
            ax.axvline(gt, color=C["green"], ls=":", lw=0.9,
                       label=f"ground truth {gt:.2f} s")
        ax.set_ylabel("median 21-ch\nRMS / baseline")
        ax.set_title(f"{line} line ({ep['meta']['speed_pct']:.1f}% speed)",
                     loc="left")
        ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=5.5)
        ax.set_ylim(0, 6)
    axes[1].set_xlabel("time [s]")
    fig.tight_layout()
    save(fig, "fig_detection.pdf")


# ---------- F3: 前兆带能量爬升 ----------
def f3_precursor():
    import precursor_tf as pf
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE)
    sel_bands = [2, 4, 6, 8]     # 归一化 0.005-0.01 / 0.02-0.03 / 0.06-0.125 / 0.25-0.625
    for ax, key in zip(axes, ["Stall_Speed100.16_Kulite_data.mat",
                              "Stall_Speed65.03_Kulite_data.mat"]):
        r = pf.analyze(key)
        gt = r["gt"]
        for bi in sel_bands:
            tw, z = r["fus_series"][bi]
            lo, hi = pf.BANDS_NORM[bi]
            ax.plot(tw, z, lw=0.9,
                    label=f"{lo:.3f}–{hi:.3f} $f_R$")
        ax.axhline(pf.Z_TH, color=C["gray"], ls="--", lw=0.7, label="$z=3$")
        if gt:
            ax.axvline(gt, color=C["green"], ls=":", lw=1.0, label="ground truth")
        ax.set_ylim(-3, 12)
        # peak z 超出量程时在框内标注真实峰值 (R5-5)
        zmax = max(np.max(np.clip(zv, -np.inf, np.inf)) for _, zv in
                   [r["fus_series"][bi] for bi in sel_bands])
        if zmax > 12:
            ax.annotate(f"peak $z$ = {zmax:.0f} (clipped)",
                        (0.98, 0.93), xycoords="axes fraction",
                        ha="right", va="top",
                        fontsize=5.5, color=C["darkred"])
        ax.set_title("N100" if "100" in key else "N65", loc="left")
        ax.legend(frameon=False, fontsize=5.5, ncol=2, loc="upper left")
    for ax in axes:
        ax.set_xlabel("time [s]")
    axes[0].set_ylabel("band RMS $z$-score (21-ch fusion)")
    fig.tight_layout()
    save(fig, "fig_precursor.pdf")


# ---------- F4: 稳态虚警 + 裕度指示 ----------
def f4_falsealarm_margin():
    from false_alarm_eval import eval_point, STEADY_DIR, NS_OFFICIAL, K_FUSION
    import glob
    files = sorted(glob.glob(os.path.join(STEADY_DIR, "*.mat")))
    recs = [eval_point(f) for f in files]
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE)
    ax = axes[0]
    for line, c in (("N100", C["red"]), ("N65", C["blue"])):
        rs = [r for r in recs if r["line"] == line]
        fa = [r for r in rs if r["alarm_fusion"]]
        no = [r for r in rs if not r["alarm_fusion"]]
        ax.scatter([r["massflow"] for r in no], [r["max_fus"] for r in no],
                   c=c, s=16, label=f"{line} (no alarm)")
        ax.scatter([r["massflow"] for r in fa], [r["max_fus"] for r in fa],
                   facecolors="none", edgecolors=C["darkred"], s=40, lw=1,
                   label=f"{line} (alarm)")
        ax.axvline(NS_OFFICIAL[line], color=c, ls=":", lw=0.7)
    ax.axhline(K_FUSION, color=C["gray"], ls="--", lw=0.8, label="$K=1.6$")
    ax.set_xlabel("corrected mass flow [kg/s]")
    ax.set_ylabel("max fusion statistic")
    ax.set_title("(a) steady operating points", loc="left")
    ax.legend(frameon=False, fontsize=5.5)
    ax = axes[1]
    for line, c in (("N100", C["red"]), ("N65", C["blue"])):
        rs = [r for r in recs if r["line"] == line]
        d = [r["massflow"] - NS_OFFICIAL[line] for r in rs]
        y = [r["rms_level"] * 1e3 for r in rs]
        ax.scatter(d, y, c=c, s=16, label=line)
        far = [(di, yi) for di, yi in zip(d, y) if di > 0.5]
        if far:
            dd, yy = zip(*far)
            k, b = np.polyfit(dd, yy, 1)
            xr = np.linspace(-0.4, 2.5, 40)
            ax.plot(xr, k * xr + b, color=c, ls="--", lw=0.8, alpha=0.7)
    ax.axvline(0.0, color=C["gray"], ls=":", lw=0.8)
    ax.annotate("past-NS\npoint", (-0.17, 0.685), textcoords="offset points",
                xytext=(22, 20), fontsize=6.5, color=C["red"],
                arrowprops=dict(arrowstyle="->", color=C["red"], lw=0.6))
    ax.set_xlabel("$\\Delta$NS = $\\dot{m}-\\dot{m}_{NS}$ [kg/s]")
    ax.set_ylabel("stall-band RMS level [$\\times10^{-3}$]")
    ax.set_title("(b) margin indicator", loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig_falsealarm_margin.pdf")


# ---------- F5: 轴向相干 ----------
def f5_axial():
    STALL = {"N100": "Stall_Speed100.16_Kulite_data.mat",
             "N65": "Stall_Speed65.03_Kulite_data.mat"}
    fig, ax = plt.subplots(figsize=SINGLE)
    from tuda_adapter import read_raw_mat, decimate, pressure_fluctuation_criterion
    from detector import SurgeDetector
    from scipy.signal import coherence
    from precursor_tf import BG_T
    for line, c, mk in (("N100", C["red"], "o"), ("N65", C["blue"], "s")):
        ep = read_raw_mat(os.path.join(ROOT, "tuda_data", "open_source",
                                       STALL[line]))
        P_ds, fs_ds = decimate(ep["P"], ep["fs"], 2000.0)
        t_ds = np.arange(P_ds.shape[0]) / fs_ds
        gt = pressure_fluctuation_criterion(
            dict(t=t_ds, P=P_ds, p=P_ds[:, 0], fs=fs_ds,
                 f_rotor=ep["f_rotor"]), k_sigma=6.0, win_s=0.05)
        det = SurgeDetector.from_rotor(fs=fs_ds, f_rotor=ep["f_rotor"])
        blo, bhi = det.stall_band
        m = (t_ds > BG_T[1]) & (t_ds < gt)
        N = int(m.sum())
        nper = min(N, int(fs_ds * 2))
        if N <= nper:
            nper = N // 2
        # 50% 重叠分段, 与正文 dof 声明一致 (Bendat & Piersol)
        K_seg = max(int(np.floor((N - nper) / (nper / 2))) + 1, 2)
        gam95 = float(np.sqrt(1 - 0.05 ** (1.0 / (K_seg - 1))))
        cohs = []
        for k in range(P_ds.shape[1]):
            f_c, Cxy = coherence(P_ds[m, 0], P_ds[m, k], fs=fs_ds,
                                 nperseg=nper, noverlap=nper // 2)
            cohs.append(np.mean(Cxy[(f_c >= blo) & (f_c <= bhi)]))
        ax.plot(ep["meta"]["axcor"], cohs, mk + "-", color=c, ms=3,
                label=f"{line} (pre-stall window)")
        ax.axhline(gam95, color=c, ls="--", lw=0.7, alpha=0.7,
                   label=f"{line} 95% line ({gam95:.2f})")
    ax.set_xlabel("axial position [mm] (channel 1 near leading edge)")
    ax.set_ylabel("coherence with channel 1")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig_axial.pdf")


# ---------- F6: 方法对比 (报警时刻) ----------
def f6_method_comparison():
    rows = {}
    with open(os.path.join(ROOT, "results", "dl_cross_domain.csv"),
              encoding="utf-8") as f:
        for r in csv.reader(f):
            rows[tuple(r)] = True
    # 直接用已知数值汇总 (来源: dl_cross_domain.csv / steady_false_alarm.csv)
    methods = ["CNN (zero-shot)", "CNN (few-shot, all\nlabeled windows)",
               "Classical energy\n(threshold from steady data)",
               "Proposed detector\n(fusion, $K=1.6$)"]
    alarm = {"N100": [0.50, 0.50, 2.6475, 2.387],
             "N65": [0.50, 0.50, 4.69, 4.780]}
    gt = {"N100": 2.675, "N65": 5.075}
    fig, axes = plt.subplots(1, 2, figsize=(SINGLE[0] + 1.4, 2.8), sharey=True)
    y = np.arange(len(methods))
    for ax, line in zip(axes, ("N100", "N65")):
        ax.barh(y, alarm[line], color=C["blue"], height=0.55)
        ax.axvline(gt[line], color=C["green"], ls=":", lw=1.0)
        ax.set_yticks(y)
        if line == "N100":
            ax.set_yticklabels(methods, fontsize=6)
        else:
            ax.set_yticklabels([])
        ax.set_title(f"{line} (GT {gt[line]:.2f} s)", loc="left")
        ax.set_xlabel("alarm time [s]")
        ax.invert_yaxis()
    fig.tight_layout()
    save(fig, "fig_method_comparison.pdf")


# ---------- F7: 模态波闭环 ----------
def f7_modal():
    import mg_modal as mm
    from modal_closed_loop import run_scenario
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE[0], 5.2))
    for (label, B, fr), (axo, axc) in zip(
            [("B=0.42", 0.42, 333.9), ("B=0.65", 0.65, 333.9)], axes):
        for ax, closed, tag, col in ((axo, False, "open loop", C["red"]),
                                     (axc, True, "closed loop", C["green"])):
            r = run_scenario(B, closed, 1, fr, 1.0)
            rc = r["rec"]
            ax.plot(rc["t"], rc["J"], color=col, lw=1.0, label="$J$ (stall)")
            ax.plot(rc["t"], rc["A"] * 10, color=C["purple"], lw=0.9,
                    label="$10A$ (modal)")
            ax.plot(rc["t"], rc["gammaT"], color=C["teal"], lw=0.9,
                    label="$\\gamma_T$ (valve)")
            if r["alarm_t"]:
                ax.axvline(r["alarm_t"], color=C["darkred"], ls="--", lw=0.9,
                           label=f"alarm {r['alarm_t']:.2f} s")
            ax.set_title(f"{label}, {tag}", loc="left")
            ax.set_ylim(-0.05, 3.0)
            if tag == "open loop":
                ax.set_ylabel("$J$, $10A$, $\\gamma_T$ (non-dim.)")
            ax.legend(frameon=False, fontsize=5.5, loc="upper left")
            ax.set_xlabel("time [s]")
    fig.tight_layout()
    save(fig, "fig_modal_closedloop.pdf")


# ---------- F8: 自动阈值族 ----------
def f9_critical_lead():
    """R5-2: 避喘成功率 vs 前兆提前量 (critical lead 扫描)"""
    p_list, leads, rates = [], [], []
    with open(os.path.join(ROOT, "results", "critical_lead_sweep.csv"),
              encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    agg = {}
    for r in rows:
        xm = float(r["x_m"])
        agg.setdefault(xm, dict(lead=float(r["lead_open_mean_s"]),
                                ok=0, n=0))
        agg[xm]["ok"] += int(r["avoided"] == "True")
        agg[xm]["n"] += 1
    xs = sorted(agg)
    fig, ax = plt.subplots(figsize=SINGLE)
    ax.axvspan(0.23, 0.55, color=C["gray"], alpha=0.15)
    ax.plot([agg[x]["lead"] for x in xs],
            [100 * agg[x]["ok"] / agg[x]["n"] for x in xs],
            "o-", color=C["blue"], ms=4, lw=1.0)
    for x in xs:
        ax.annotate(f"{agg[x]['ok']}/{agg[x]['n']}",
                    (agg[x]["lead"], 100 * agg[x]["ok"] / agg[x]["n"] + 6),
                    fontsize=5.5, ha="center")
    ax.axvline(0.72, color=C["green"], ls=":", lw=1.0, label="measured lead")
    ax.axvline(0.92, color=C["green"], ls=":", lw=1.0)
    ax.text(0.82, 62, "measured\nlead", color=C["green"], fontsize=5.5,
            ha="center")
    ax.text(0.39, 25, "critical\nlead", fontsize=5.5, ha="center")
    ax.set_xlabel("precursor lead [s]")
    ax.set_ylabel("avoidance success rate [%]")
    ax.set_ylim(-5, 110)
    ax.legend(frameon=False, fontsize=5.5, loc="lower right")
    fig.tight_layout()
    save(fig, "fig_critical_lead.pdf")


def f8_autothreshold():
    p_list, Ks, fas, a100, a65 = [], [], [], [], []
    with open(os.path.join(ROOT, "results", "auto_threshold.csv"),
              encoding="utf-8") as f:
        for r in csv.reader(f):
            if r and r[0].startswith("sweep_p="):
                p = float(r[0].split("=")[1])
                kv = dict(kv.split("=") for kv in r[1].split(", "))
                p_list.append(p); Ks.append(float(kv["K"]))
                fas.append(int(kv["FA"]))
                a100.append(float(kv["N100"])); a65.append(float(kv["N65"]))
    fig, ax = plt.subplots(figsize=SINGLE)
    ax.plot(Ks, fas, "o-", color=C["blue"], ms=3.5, label="steady FA points")
    for p, K, a1, a6 in zip(p_list, Ks, a100, a65):
        ax.annotate(f"$p_t$={p:g}", (K, fas[p_list.index(p)]),
                    textcoords="offset points", xytext=(0, 5), fontsize=5.5,
                    ha="center")
    ax.axvline(2.195, color=C["red"], ls="--", lw=0.9, label="$K_{auto}$")
    ax.set_xlabel("threshold $K$")
    ax.set_ylabel("steady points with FA episode")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig_autothreshold.pdf")


if __name__ == "__main__":
    f1_characteristic()
    f2_detection()
    f3_precursor()
    f4_falsealarm_margin()
    f5_axial()
    # f6_method_comparison()  # superseded by Table 2 in the paper
    #                         (kept for the supplementary repository)
    f7_modal()
    f9_critical_lead()
    f8_autothreshold()
    print("all figures done ->", FIGDIR)
