# -*- coding: utf-8 -*-
"""
TUDa Stall run 端到端检测验证 —— adapter 实装后的最终管线测试

数据: tuda_data/open_source/Stall_Speed{100.16,65.03}_Kulite_data.mat
  - kudatabase (4e6, 21) 500 kHz, 无量纲 ps/ptref, 轴向 21 点 (同一周向 260°)
  - 转速由 speed 字段换算 f_rotor, 再用 BPF 谱峰独立交叉验证

验证内容:
  1. sanity_check (采样率/Nyquist/幅值/NaN)
  2. f_rotor 交叉验证 (官方换算 vs BPF 谱峰估计)
  3. 降采样 2 kHz -> SurgeDetector.from_rotor 离线检测
     - 单通道 (近前缘 ch0)
     - 多传感器融合 (21 通道失速带 RMS 相对基线倍数的中位数)
  4. ground truth: pressure_fluctuation_criterion (失速带 RMS 6σ MAD 跳变)
  5. 检测提前量/延迟报告 + 图

用法: python test_tuda_stall.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector import SurgeDetector, fft_bandpass, sliding_rms
from tuda_adapter import (read_raw_mat, decimate, estimate_f_rotor,
                          sanity_check, pressure_fluctuation_criterion,
                          WPTLayout, FS_RAW)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "tuda_data", "open_source")
STALL_FILES = ["Stall_Speed100.16_Kulite_data.mat",
               "Stall_Speed65.03_Kulite_data.mat"]
FS_DS = 2000.0            # 降采样目标 (检测频带最高 0.625*f_rotor ≈ 209 Hz)
K_FUSION = 1.6            # 融合统计量 (中位数相对基线倍数) 报警阈值
N_SUSTAIN_FUSION = 5      # 融合持续窗口数


def channel_rms_norm(P_ds, fs_ds, f_rotor):
    """各通道失速带 RMS / 基线 中位数统计 -> 融合检测序列"""
    det = SurgeDetector.from_rotor(fs=fs_ds, f_rotor=f_rotor)
    band = det.stall_band
    t = np.arange(P_ds.shape[0]) / fs_ds
    tw = None
    R = []
    for k in range(P_ds.shape[1]):
        pb = fft_bandpass(P_ds[:, k] - P_ds[:, k].mean(), fs_ds, *band)
        tw, r = sliding_rms(pb, fs_ds, det.win_s, det.hop_s)
        R.append(r)
    R = np.stack(R, axis=1)                      # (n_win, n_ch)
    n_base = int((det.baseline_s * 0.85) * fs_ds / (det.hop_s * fs_ds)) \
        if False else int(np.searchsorted(tw, det.baseline_s))
    m0 = int(np.searchsorted(tw, det.baseline_skip_s))
    base = np.median(R[m0:n_base, :], axis=0) + 1e-9
    Rn = R / base                                # 相对基线倍数
    fus = np.median(Rn, axis=1)                  # 融合统计量
    exc = fus > K_FUSION
    alarm = None
    cnt = 0
    for i, e in enumerate(exc):
        cnt = cnt + 1 if e else 0
        if cnt >= N_SUSTAIN_FUSION:
            alarm = tw[i - N_SUSTAIN_FUSION + 1]
            break
    return tw, Rn, fus, alarm


def run_one(fname, ax_row):
    path = os.path.join(DATA_DIR, fname)
    print("=" * 74)
    print(f"文件: {fname}")
    ep = read_raw_mat(path)
    meta = ep["meta"]
    print(f"  工况: ṁ={meta['massflow']:.2f} kg/s, speed={meta['speed_pct']:.3f}%"
          f" -> f_rotor={ep['f_rotor']:.2f} Hz (官方换算)")
    ok = sanity_check(ep)
    print(f"  sanity_check: {'PASS' if ok else 'FAIL'}")

    # f_rotor 交叉验证: 取近前缘 1s 数据估计 BPF
    seg = ep["P"][:int(FS_RAW), 0]
    f_est, bpf, sig = estimate_f_rotor(seg, FS_RAW)
    dev = abs(f_est - ep["f_rotor"]) / ep["f_rotor"]
    print(f"  BPF 交叉验证: BPF={bpf:.1f} Hz (峰显著性 x{sig:.0f}) "
          f"-> f_rotor_est={f_est:.2f} Hz, 与换算偏差 {dev*100:.2f}%")
    print(f"  {'OK' if dev < 0.02 else 'WARN'}: |Δ| < 2%"
          if dev < 0.02 else "  WARN: BPF 估计偏差 > 2%")

    # 降采样
    P_ds, fs_ds = decimate(ep["P"], ep["fs"], FS_DS)
    print(f"  降采样: {ep['P'].shape} @ {ep['fs']/1e3:.0f}kHz -> "
          f"{P_ds.shape} @ {fs_ds:.0f} Hz")

    # ground truth (失速带 RMS 6σ MAD 跳变, 近前缘通道)
    ep_ds = dict(t=np.arange(P_ds.shape[0]) / fs_ds, P=P_ds, p=P_ds[:, 0],
                 fs=fs_ds, f_rotor=ep["f_rotor"])
    gt = pressure_fluctuation_criterion(ep_ds, k_sigma=6.0, win_s=0.05)

    # 单通道检测 (近前缘 ch0) —— 仿真管线同款
    res = None
    from detector import SurgeDetector as SD
    det = SD.from_rotor(fs=fs_ds, f_rotor=ep["f_rotor"])
    res = det.run(ep_ds["t"], ep_ds["p"])
    print(f"  单通道 ch0 (x={meta['axcor'][0]:.0f}mm): "
          f"alarm={res['alarm_time']} ({res['alarm_type']})")

    # 多传感器融合
    tw_f, Rn, fus, alarm_f = channel_rms_norm(P_ds, fs_ds, ep["f_rotor"])
    print(f"  融合检测 (21 通道中位数, K={K_FUSION}, n={N_SUSTAIN_FUSION}): "
          f"alarm={alarm_f}")

    # ground truth 报告
    print(f"  ground truth (包络跳变): {gt}")
    for tag, a in [("单通道", res["alarm_time"]), ("融合", alarm_f)]:
        if a is not None and gt is not None:
            lead = gt - a
            print(f"  {tag} 相对 ground truth: {lead:+.2f} s "
                  f"({'提前' if lead > 0 else '滞后'})")

    # ---- 图 ----
    ax_row[0].plot(tw_f, fus, color="#c43b3b", lw=1,
                   label="fusion (median of 21 ch)")
    rn0 = res["r_stall"] / res["base_stall"]
    ax_row[0].plot(res["t_win"], rn0, color="#3b74c4", lw=0.8, alpha=0.8,
                   label="single ch0 (RMS/base)")
    ax_row[0].axhline(K_FUSION, color="gray", ls="--", lw=1, label=f"K={K_FUSION}")
    if alarm_f:
        ax_row[0].axvline(alarm_f, color="#c43b3b", ls=":", lw=1.5)
        ax_row[0].text(alarm_f, ax_row[0].get_ylim()[1] * 0.9,
                       f"fusion\n{alarm_f:.2f}s", fontsize=8, ha="left")
    if gt:
        ax_row[0].axvline(gt, color="green", ls=":", lw=1.5)
        ax_row[0].text(gt, ax_row[0].get_ylim()[1] * 0.7, f"GT\n{gt:.2f}s",
                       fontsize=8, ha="left", color="green")
    if res["alarm_time"]:
        ax_row[0].axvline(res["alarm_time"], color="#3b74c4", ls=":", lw=1.5)
    ax_row[0].set_ylabel("stall-band RMS / baseline")
    ax_row[0].set_title(f"{fname}: ṁ={meta['massflow']:.2f} kg/s, "
                        f"{meta['speed_pct']:.1f}% N", fontsize=10)
    ax_row[0].legend(fontsize=7); ax_row[0].grid(alpha=0.3)

    # 逐通道首次持续超阈时刻 (与融合同判据: 连续 N_SUSTAIN_FUSION 窗)
    first_exc = []
    for k in range(Rn.shape[1]):
        exc_k = Rn[:, k] > K_FUSION
        cnt_k, t_k = 0, np.nan
        for i, e in enumerate(exc_k):
            cnt_k = cnt_k + 1 if e else 0
            if cnt_k >= N_SUSTAIN_FUSION:
                t_k = tw_f[i - N_SUSTAIN_FUSION + 1]
                break
        first_exc.append(t_k)
    ax_row[1].bar(range(len(first_exc)), first_exc, color="#3b74c4", alpha=0.8)
    ax_row[1].set_xlabel("sensor index (axial: ch0 upstream -> ch20 downstream)")
    ax_row[1].set_ylabel("first exc> K [s]")
    ax_row[1].grid(alpha=0.3, axis="y")
    return dict(sanity=ok, f_dev=dev, gt=gt,
                alarm_single=res["alarm_time"], alarm_fusion=alarm_f)


def main():
    fig, axes = plt.subplots(4, 1, figsize=(11, 11))
    results = {}
    rows = [(axes[0], axes[1]), (axes[2], axes[3])]
    for fname, ax_row in zip(STALL_FILES, rows):
        results[fname] = run_one(fname, ax_row)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "results", "tuda_stall_detection.png")
    fig.savefig(out, dpi=150)
    print("=" * 74)
    print(f"图已保存: {out}")

    print("\n==== 汇总 ====")
    all_pass = True
    for fname, r in results.items():
        ok = r["sanity"] and r["f_dev"] < 0.02 and r["alarm_fusion"] is not None
        all_pass &= ok
        print(f"  {fname}: sanity={r['sanity']} f_dev={r['f_dev']*100:.2f}% "
              f"GT={r['gt']} fusion_alarm={r['alarm_fusion']} -> "
              f"{'PASS' if ok else 'CHECK'}")
    print(f"\n总体: {'PASS' if all_pass else 'NEEDS REVIEW'}")


if __name__ == "__main__":
    main()
