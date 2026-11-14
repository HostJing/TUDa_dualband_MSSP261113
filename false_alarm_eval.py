# -*- coding: utf-8 -*-
"""
P0-1 稳态工况点虚警率评估 —— 检测管线的实测误报定量

数据: tuda_data/steady/*.mat (N100 16 点 + N65 10 点, 各 8s @ 500kHz, 21 通道)
  稳态点定义上不发生失稳 -> 任何报警均为误报 (false alarm)

指标:
  - 融合统计量 (21 通道失速带 RMS/基线 的中位数, K=1.6, 持续 5 窗报警)
    与 stall run 检测管线完全同款 -> 同一阈值下的误报率
  - 单通道 det.run (ch0) 双频带报警
  - 阈值扫描: K in [1.2, 2.6] -> FP 率曲线 (选阈依据)
  - 裕度指示: 失速带 RMS 绝对水平/最大融合统计量 vs 校正流量 (逼近 NS 点趋势)

产出: results/steady_false_alarm.csv / steady_false_alarm.png
用法: python src/false_alarm_eval.py
"""
import os
import sys
import glob
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector import SurgeDetector
from tuda_adapter import read_raw_mat, decimate
from test_tuda_stall import channel_rms_norm, FS_DS, K_FUSION, N_SUSTAIN_FUSION

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEADY_DIR = os.path.join(ROOT, "tuda_data", "steady")
NS_OFFICIAL = {"N100": 13.53, "N65": 9.31}   # 官方共享工况点表 NS (kg/s)
K_SWEEP = np.arange(1.2, 2.61, 0.1)


def eval_point(path):
    ep = read_raw_mat(path)
    line = "N100" if ep["f_rotor"] > 250 else "N65"
    P_ds, fs_ds = decimate(ep["P"], ep["fs"], FS_DS)

    # 融合统计量 (与 stall run 管线同款)
    tw, Rn, fus, alarm_f = channel_rms_norm(P_ds, fs_ds, ep["f_rotor"])
    # 单通道双频带检测 (ch0)
    det = SurgeDetector.from_rotor(fs=fs_ds, f_rotor=ep["f_rotor"])
    t_ds = np.arange(P_ds.shape[0]) / fs_ds
    res = det.run(t_ds, P_ds[:, 0])

    # 失速带绝对 RMS 水平 (21 通道中位, 裕度指示)
    det2 = SurgeDetector.from_rotor(fs=fs_ds, f_rotor=ep["f_rotor"])
    from detector import fft_bandpass, sliding_rms
    pb = fft_bandpass(P_ds[:, 0] - P_ds[:, 0].mean(), fs_ds, *det2.stall_band)
    _, rms_abs = sliding_rms(pb, fs_ds, det2.win_s, det2.hop_s)
    rms_level = float(np.median(rms_abs))

    m = ep["meta"]
    return dict(
        line=line, file=os.path.basename(path),
        massflow=m["massflow"], speed=m["speed_pct"], f_rotor=ep["f_rotor"],
        alarm_fusion=(alarm_f is not None),
        t_alarm_fusion=alarm_f,
        alarm_single=(res["alarm_time"] is not None),
        type_single=res["alarm_type"],
        max_fus=float(np.max(fus)), p95_fus=float(np.percentile(fus, 95)),
        frac_exc=float(np.mean(fus > K_FUSION)),
        rms_level=rms_level,
        fus=fus, tw=tw,           # 留给阈值扫描
    )


def main():
    files = sorted(glob.glob(os.path.join(STEADY_DIR, "*.mat")))
    print("=" * 78)
    print(f"P0-1 稳态点虚警评估: {len(files)} 个稳态点 (每点 8s @ 2kHz, 21 通道)")
    print("=" * 78)
    recs = [eval_point(f) for f in files]

    # ---- CSV ----
    out_csv = os.path.join(ROOT, "results", "steady_false_alarm.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["line", "file", "massflow_kgs", "speed_pct", "f_rotor_Hz",
                    "alarm_fusion", "t_alarm_fusion_s", "alarm_single",
                    "type_single", "max_fusion", "p95_fusion",
                    "frac_win_over_K", "stallband_rms_level"])
        for r in recs:
            w.writerow([r["line"], r["file"], f"{r['massflow']:.3f}",
                        f"{r['speed']:.3f}", f"{r['f_rotor']:.2f}",
                        r["alarm_fusion"],
                        f"{r['t_alarm_fusion']:.3f}" if r["t_alarm_fusion"] else "",
                        r["alarm_single"], r["type_single"] or "",
                        f"{r['max_fus']:.3f}", f"{r['p95_fus']:.3f}",
                        f"{r['frac_exc']:.4f}", f"{r['rms_level']:.5f}"])
    print(f"\nCSV 已保存: {out_csv}")

    # ---- 汇总统计 ----
    total_s = len(recs) * 8.0
    print(f"\n监测总时长: {total_s:.0f}s ({total_s/3600:.2f} h)")
    for line in ("N100", "N65"):
        rs = [r for r in recs if r["line"] == line]
        n_fa = sum(r["alarm_fusion"] for r in rs)
        print(f"\n[{line}] {len(rs)} 点, NS官方={NS_OFFICIAL[line]} kg/s")
        print(f"  融合报警点数: {n_fa} -> 虚警率 {n_fa} 次/{total_s/len(recs)*len(rs):.0f}s "
              f"= {n_fa/(total_s/3600):.1f} 次/h (外推)")
        print(f"  单通道报警点数: {sum(r['alarm_single'] for r in rs)}")
        for r in sorted(rs, key=lambda x: x["massflow"]):
            dm = r["massflow"] - NS_OFFICIAL[line]
            flag = " <-- 近 NS" if abs(dm) < 0.3 else ""
            print(f"    ṁ={r['massflow']:.2f} (ΔNS={dm:+.2f}) "
                  f"max_fus={r['max_fus']:.2f} frac>K={r['frac_exc']:.3f} "
                  f"rms={r['rms_level']:.4f} "
                  f"FA={'Y' if r['alarm_fusion'] else 'n'}"
                  f"{' type=' + str(r['type_single']) if r['alarm_single'] else ''}{flag}")

    # ---- 阈值扫描 ----
    print("\n阈值扫描 (持续 5 窗判据不变):")
    sweep = {}
    for line in ("N100", "N65"):
        rs = [r for r in recs if r["line"] == line]
        sweep[line] = [sum(bool(np.any(_sustained(r["fus"] > k, N_SUSTAIN_FUSION)))
                           for r in rs) for k in K_SWEEP]
        print(f"  [{line}] K={K_SWEEP[0]:.1f}..{K_SWEEP[-1]:.1f}: "
              f"FA 点数 {sweep[line]}")

    # ---- 图 ----
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    ax = axes[0]
    for line, c in (("N100", "#c43b3b"), ("N65", "#3b74c4")):
        rs = [r for r in recs if r["line"] == line]
        ax.scatter([r["massflow"] for r in rs], [r["max_fus"] for r in rs],
                   c=c, s=45, label=line, zorder=3)
        ax.axvline(NS_OFFICIAL[line], color=c, ls=":", lw=1)
    ax.axhline(K_FUSION, color="gray", ls="--", lw=1, label=f"K={K_FUSION}")
    ax.set_xlabel("corrected massflow [kg/s]  (dotted: official NS)")
    ax.set_ylabel("max fusion statistic (RMS/base, median 21 ch)")
    ax.set_title("Steady operating points: fusion statistic vs massflow "
                 "(margin indicator & false alarms)")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    for line, c in (("N100", "#c43b3b"), ("N65", "#3b74c4")):
        ax.plot(K_SWEEP, np.array(sweep[line]) / len([r for r in recs if r["line"] == line]),
                "o-", color=c, label=line)
    ax.axvline(K_FUSION, color="gray", ls="--", lw=1, label=f"K={K_FUSION} (stall-run tuned)")
    ax.set_xlabel("threshold K (sustained-5-windows)")
    ax.set_ylabel("fraction of steady points alarming")
    ax.set_title("False-alarm rate vs threshold K")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(ROOT, "results", "steady_false_alarm.png")
    fig.savefig(out_png, dpi=150)
    print(f"\n图已保存: {out_png}")


def _sustained(exc, n):
    cnt = 0
    for e in exc:
        cnt = cnt + 1 if e else 0
        if cnt >= n:
            return True
    return False


if __name__ == "__main__":
    main()
