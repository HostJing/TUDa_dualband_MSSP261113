# -*- coding: utf-8 -*-
"""
P0-3 因果在线化评测 —— 滚动缓冲 + 块状 WOLA 的在线检测架构

架构 (论文在线检测主张的实现):
  - 数据按 CHUNK_S=0.25s 分块流入 (模拟采集流)
  - 维护滚动缓冲 T_BUF=8s: 每收到一块, 对整个缓冲跑离线 WOLA 检测器
    (检测语义与离线版完全一致 -> 报警可复现)
  - 延迟预算: 检测延迟 = 块长 (采集) + 缓冲评估计算时间 (CPU)
    报警语义延迟 = RMS 窗 + 持续判据窗

验收: 在线报警时刻与离线一次成算差 < 0.1s (双转速 stall run)
产出: results/online_eval.png / 延迟预算表 (控制台)
用法: python src/online_eval.py
"""
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector import SurgeDetector
from tuda_adapter import (read_raw_mat, decimate,
                          pressure_fluctuation_criterion, FS_RAW)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "tuda_data", "open_source")
STALL_FILES = ["Stall_Speed100.16_Kulite_data.mat",
               "Stall_Speed65.03_Kulite_data.mat"]
FS_DS = 2000.0
CHUNK_S = 0.25
T_BUF = 8.0


def online_detect(fname):
    ep = read_raw_mat(os.path.join(DATA_DIR, fname))
    P_ds, fs_ds = decimate(ep["P"], ep["fs"], FS_DS)
    t_ds = np.arange(P_ds.shape[0]) / fs_ds
    p = P_ds[:, 0].astype(np.float64)
    gt = pressure_fluctuation_criterion(
        dict(t=t_ds, P=P_ds, p=p, fs=fs_ds, f_rotor=ep["f_rotor"]),
        k_sigma=6.0, win_s=0.05)
    det = SurgeDetector.from_rotor(fs=fs_ds, f_rotor=ep["f_rotor"])
    # 离线基准 (一次成算)
    t0 = time.perf_counter()
    off = det.run(t_ds, p)
    cpu_off = time.perf_counter() - t0

    # 在线流式
    n_chunk = int(CHUNK_S * fs_ds)
    n_buf = int(T_BUF * fs_ds)
    alarm_t = None
    alarms = []                       # (t_report, type)
    cpu_total, n_eval = 0.0, 0
    first_alarm_delay = None
    armed_logged = False
    # 检测器在 t < max(baseline_s, win_surge_s) 不可能报警 -> 缓冲够长才开始评估
    n_min = int((det.baseline_s + det.win_surge_s) * fs_ds)
    for end in range(n_chunk, len(p) + 1, n_chunk):
        if end < n_min:
            continue
        lo = max(0, end - n_buf)
        tb = t_ds[lo:end]
        pb = p[lo:end] - p[lo:end].mean() if False else p[lo:end]
        t0 = time.perf_counter()
        res = det.run(tb, pb)
        cpu_total += time.perf_counter() - t0
        n_eval += 1
        if res["alarm_time"] is not None:
            t_rep = tb[-1]            # 报警随当前块流出 (报告时刻)
            if alarm_t is None:
                alarm_t = res["alarm_time"]
                first_alarm_delay = t_rep - res["alarm_time"]
            alarms.append((t_rep, res["alarm_type"]))
    cpu_per_eval = cpu_total / max(n_eval, 1)

    print("=" * 76)
    print(f"文件: {fname} (ṁ={ep['meta']['massflow']:.2f} kg/s, "
          f"{ep['meta']['speed_pct']:.1f}%N)")
    print(f"  离线基准: alarm={off['alarm_time']} ({off['alarm_type']}), "
          f"GT={gt}, CPU={cpu_off*1000:.0f} ms (8s 信号)")
    first_type = alarms[0][1] if alarms else "none"
    print(f"  在线流式: 首报警={alarm_t} ({first_type}), "
          f"评估 {n_eval} 次, CPU/次={cpu_per_eval*1000:.1f} ms "
          f"(块 {CHUNK_S*1000:.0f} ms)")
    print(f"  实时率: 1 次评估 / {CHUNK_S:.2f}s 数据 -> "
          f"计算占比 {cpu_per_eval/CHUNK_S*100:.2f}% (单核)")
    if alarm_t is not None and off["alarm_time"] is not None:
        dev = abs(alarm_t - off["alarm_time"])
        ok = dev <= 0.10
        print(f"  在线 vs 离线偏差: {dev*1000:.0f} ms "
              f"(含检测语义 {first_alarm_delay*1000:.0f} ms 块流出延迟) "
              f"-> {'PASS (<100ms)' if ok else 'FAIL'}")
        return dict(fname=fname, gt=gt, offline=off["alarm_time"],
                    online=alarm_t, dev=dev, ok=ok,
                    alarms=alarms, cpu_per_eval=cpu_per_eval,
                    t_ds=t_ds, p=p, det=det)
    return dict(fname=fname, gt=gt, offline=off["alarm_time"],
                online=alarm_t, dev=None, ok=False, alarms=alarms,
                cpu_per_eval=cpu_per_eval, t_ds=t_ds, p=p, det=det)


def main():
    results = [online_detect(f) for f in STALL_FILES]

    fig, axes = plt.subplots(len(results), 1, figsize=(11, 4 * len(results)))
    for i, r in enumerate(results):
        ax = axes[i] if len(results) > 1 else axes
        det = r["det"]
        res = det.run(r["t_ds"], r["p"])
        ax.plot(res["t_win"], res["r_stall"] / res["base_stall"],
                lw=0.8, color="#3b74c4", label="stall band RMS/base")
        ax.plot(res["t_surge_win"], res["r_surge"] / res["base_surge"],
                lw=0.8, color="#c47b3b", alpha=0.7, label="surge band RMS/base")
        ax.axhline(2.0, color="gray", ls="--", lw=0.8)
        for t_rep, typ in r["alarms"]:
            ax.axvline(t_rep, color="red", ls=":", lw=0.8, alpha=0.5)
        if r["online"]:
            ax.axvline(r["online"], color="red", lw=1.5,
                       label=f"online alarm {r['online']:.2f}s")
        if r["offline"]:
            ax.axvline(r["offline"], color="blue", ls="--", lw=1.2,
                       label=f"offline alarm {r['offline']:.2f}s")
        if r["gt"]:
            ax.axvline(r["gt"], color="green", ls=":", lw=1.2,
                       label=f"GT {r['gt']:.2f}s")
        ax.set_ylim(0, 8)
        ax.set_xlabel("t [s]")
        ax.set_ylabel("RMS / baseline")
        ax.set_title(f"{r['fname']}: online streaming (red ticks = per-chunk "
                     f"reports)", fontsize=10)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    print("\n==== 验收汇总 ====")
    all_ok = all(r["ok"] for r in results)
    for r in results:
        print(f"  {r['fname']}: offline={r['offline']} online={r['online']} "
              f"dev={r['dev'] if r['dev'] is None else f'{r[chr(100)+"ev"]*1000:.0f}ms'}"
              f" -> {'PASS' if r['ok'] else 'FAIL'}")
    print(f"总体: {'PASS' if all_ok else 'FAIL'}")

    fig.tight_layout()
    out = os.path.join(ROOT, "results", "online_eval.png")
    fig.savefig(out, dpi=150)
    print(f"图已保存: {out}")


if __name__ == "__main__":
    main()
