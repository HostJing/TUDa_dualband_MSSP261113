# -*- coding: utf-8 -*-
"""
R1 在线融合版评测 —— 21 通道融合统计量的因果在线实现

问题: online_eval.py 是单通道在线检测 (报警 2.81s), 论文主线数字
2.387/4.780s 来自 21 通道融合统计量 —— 二者必须一致才能支撑
"在线"主张。本脚本实现融合统计量的因果在线版本:

  数据按 CHUNK_S=0.25s 流入 -> 维护全部历史缓冲 -> 每块对缓冲
  重算融合统计量序列 (与离线 channel_rms_norm 完全同款) ->
  首个 sustained-5 越限即报警 -> 报警后锁定

由于缓冲从 t=0 起保留 (8s 总长), 基线窗与离线完全一致 ->
在线报警 = 离线报警 + 至多一个块流出延迟。
验收: |在线 - 离线| < 0.1s; CPU 占比报告。
产出: results/online_fusion_eval.csv
用法: python src/online_fusion_eval.py
"""
import os
import sys
import csv
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tuda_adapter import read_raw_mat, decimate, pressure_fluctuation_criterion
from test_tuda_stall import channel_rms_norm, FS_DS
from detector import SurgeDetector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "tuda_data", "open_source")
STALL_FILES = {"N100": "Stall_Speed100.16_Kulite_data.mat",
               "N65": "Stall_Speed65.03_Kulite_data.mat"}
CHUNK_S = 0.25


def eval_line(line):
    fname = STALL_FILES[line]
    ep = read_raw_mat(os.path.join(DATA_DIR, fname))
    P_ds, fs_ds = decimate(ep["P"], ep["fs"], FS_DS)
    t_ds = np.arange(P_ds.shape[0]) / fs_ds
    gt = pressure_fluctuation_criterion(
        dict(t=t_ds, P=P_ds, p=P_ds[:, 0], fs=fs_ds, f_rotor=ep["f_rotor"]),
        k_sigma=6.0, win_s=0.05)
    det = SurgeDetector.from_rotor(fs=fs_ds, f_rotor=ep["f_rotor"])

    # 离线基准 (融合)
    t0 = time.perf_counter()
    _, _, fus_off, alarm_off = channel_rms_norm(P_ds, fs_ds, ep["f_rotor"])
    cpu_off = time.perf_counter() - t0

    # 在线流式 (融合): 每块对全部历史重算
    n_chunk = int(CHUNK_S * fs_ds)
    alarm_on, alarm_type = None, None
    cpu_total, n_eval, first_out_delay = 0.0, 0, None
    for end in range(n_chunk, P_ds.shape[0] + 1, n_chunk):
        t0 = time.perf_counter()
        tw, Rn, fus, _ = channel_rms_norm(P_ds[:end], fs_ds, ep["f_rotor"])
        cpu_total += time.perf_counter() - t0
        n_eval += 1
        if alarm_on is None and fus is not None:
            exc = fus > 1.6
            cnt = 0
            for i, e in enumerate(exc):
                cnt = cnt + 1 if e else 0
                if cnt >= 5:
                    alarm_on = tw[i - 4]
                    alarm_type = "fusion_stall_band"
                    first_out_delay = (end / fs_ds) - alarm_on
                    break
    cpu_per = cpu_total / max(n_eval, 1)

    dev = (abs(alarm_on - alarm_off)
           if (alarm_on is not None and alarm_off is not None) else None)
    print("=" * 76)
    print(f"[{line}] {fname} (ṁ={ep['meta']['massflow']:.2f}, "
          f"{ep['meta']['speed_pct']:.1f}%N, GT={gt})")
    print(f"  离线融合报警: {alarm_off}  CPU={cpu_off*1000:.0f} ms (8s)")
    print(f"  在线融合报警: {alarm_on}  块流出延迟={first_out_delay*1000:.0f} ms")
    print(f"  CPU/次={cpu_per*1000:.1f} ms (块 {CHUNK_S*1000:.0f} ms) -> "
          f"占比 {cpu_per/CHUNK_S*100:.2f}% 单核")
    ok = dev is not None and dev <= 0.10
    print(f"  偏差: {dev*1000:.0f} ms -> {'PASS (<100ms)' if ok else 'FAIL'}")
    return dict(line=line, gt=gt, alarm_off=alarm_off, alarm_on=alarm_on,
                dev=dev, ok=bool(ok), cpu_pct=cpu_per / CHUNK_S * 100,
                first_out_delay=first_out_delay, fus=fus)


def main():
    results = [eval_line(k) for k in ("N100", "N65")]
    all_ok = all(r["ok"] for r in results)
    print(f"\n==== R1 验收汇总: {'PASS' if all_ok else 'FAIL'} "
          f"(双转速在线 vs 离线融合偏差 <100ms) ====")

    out_csv = os.path.join(ROOT, "results", "online_fusion_eval.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["line", "gt_s", "alarm_offline_s", "alarm_online_s",
                    "dev_ms", "block_out_delay_ms", "cpu_pct_single_core",
                    "pass"])
        for r in results:
            w.writerow([r["line"], f"{r['gt']:.3f}", f"{r['alarm_off']:.3f}",
                        f"{r['alarm_on']:.3f}", f"{r['dev']*1000:.0f}",
                        f"{r['first_out_delay']*1000:.0f}",
                        f"{r['cpu_pct']:.2f}", r["ok"]])
    print(f"CSV 已保存: {out_csv}")

    # 图
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    for ax, r in zip(axes, results):
        ep = read_raw_mat(os.path.join(DATA_DIR, STALL_FILES[r["line"]]))
        P_ds, fs_ds = decimate(ep["P"], ep["fs"], FS_DS)
        tw, Rn, fus, alarm_off = channel_rms_norm(P_ds, fs_ds, ep["f_rotor"])
        ax.plot(tw, fus, lw=0.9, color="#c43b3b", label="fusion statistic")
        ax.axhline(1.6, color="gray", ls="--", lw=1, label="K=1.6")
        ax.axvline(r["alarm_off"], color="blue", ls="--", lw=1.2,
                   label=f"offline {r['alarm_off']:.3f}s")
        ax.axvline(r["alarm_on"], color="red", ls=":", lw=1.5,
                   label=f"online {r['alarm_on']:.3f}s")
        if r["gt"]:
            ax.axvline(r["gt"], color="green", ls=":", lw=1.2,
                       label=f"GT {r['gt']:.2f}s")
        ax.set_xlabel("t [s]"); ax.set_ylabel("median 21-ch RMS/base")
        ax.set_title(f"R1 online fusion ({r['line']})", fontsize=10)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(ROOT, "results", "online_fusion_eval.png")
    fig.savefig(out_png, dpi=150)
    print(f"图已保存: {out_png}")


if __name__ == "__main__":
    main()
