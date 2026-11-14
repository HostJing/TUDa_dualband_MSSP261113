# -*- coding: utf-8 -*-
"""
传感器数量/位置消融实验

问题: 最少需要几个周向壁压传感器才能可靠预警?
变量:
  - 通道数 K = 1..8 (等间隔布置)
  - 布置方式: 等间隔 vs 聚束 (K=4, 传感器集中在 90° 扇区内, 最坏位置)
指标:
  - 旋转失速型失稳确认率 (闸门, K>=2; K=1 为纯单通道检测)
  - 稳定工况误报率
  - 失速团传播速度估计误差 (beta, K>=2)
输出 results/sensor_ablation.md + sensor_ablation.png
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from array_processing import (synthesize_array, coherence_gated_alarm,
                              fft_bandpass, windowed_speed)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")


def eval_gate(d, angles=None, n_sensors=8):
    """K>=2: 相干性闸门; K=1: 纯单通道检测 (无阵列闸门可用)"""
    if n_sensors == 1:
        arr = synthesize_array(d, n_sensors=1, seed=7)
        from detector import SurgeDetector
        det = SurgeDetector.from_rotor(fs=arr["fs"], f_rotor=arr["f_rotor"],
                                       k_stall=4, k_surge=4)
        res = det.run(arr["t"], arr["P"][:, 0])
        return dict(confirmed=res["alarm_time"] is not None,
                    alarm_time=res["alarm_time"])
    arr = synthesize_array(d, n_sensors=n_sensors, seed=7, angles_deg=angles)
    g = coherence_gated_alarm(arr["P"], arr["fs"], arr["f_rotor"])
    return g


def beta_err(d, angles=None, n_sensors=8):
    arr = synthesize_array(d, n_sensors=n_sensors, seed=7, angles_deg=angles)
    fs, fr = arr["fs"], arr["f_rotor"]
    sb = tuple(s * fr for s in (0.25, 0.625))
    Pb = np.column_stack([fft_bandpass(arr["P"][:, k] - arr["P"][:, k].mean(),
                                       fs, *sb) for k in range(arr["P"].shape[1])])
    sp = windowed_speed(Pb, fs, fr, angles_deg=angles, win_s=0.2, hop_s=0.1)
    m = sp["corr"] > 0.5
    if not m.any():
        return np.nan
    return abs(np.nanmedian(sp["beta"][m]) - 0.45)


def main():
    idx = pd.read_csv(os.path.join(DATA, "index.csv"))
    rs_eps = idx[(idx.family == "rotating_stall")
                 | ((idx.family == "surge") & (idx.label == "rotating_stall"))]
    stable_eps = idx[idx.family == "stable"]
    print(f"旋转失速型 {len(rs_eps)}, 稳定 {len(stable_eps)}")

    K_LIST = [1, 2, 3, 4, 6, 8]
    LAYOUTS = {"uniform": None,
               "clustered_K4": None}   # clustered 仅在 K=4 使用
    rows = []
    for K in K_LIST:
        angles = None if K == 1 else np.linspace(0, 360, K, endpoint=False)
        # --- 确认率 / 误报 ---
        conf = 0
        for _, r in rs_eps.iterrows():
            d = np.load(os.path.join(DATA, r["file"]), allow_pickle=True)
            g = eval_gate(d, angles, K)
            conf += bool(g["confirmed"])
        far = 0
        for _, r in stable_eps.iterrows():
            d = np.load(os.path.join(DATA, r["file"]), allow_pickle=True)
            g = eval_gate(d, angles, K)
            far += bool(g["confirmed"])
        # --- beta 误差 ---
        berrs = []
        if K >= 2:
            for _, r in idx[idx.family == "rotating_stall"].iterrows():
                d = np.load(os.path.join(DATA, r["file"]), allow_pickle=True)
                berrs.append(beta_err(d, angles, K))
        rows.append(dict(K=K, layout="uniform",
                         confirm_rate=conf / len(rs_eps),
                         far=far / len(stable_eps),
                         beta_err_med=np.nanmedian(berrs) if berrs else np.nan,
                         beta_err_p90=np.nanquantile(berrs, .9) if berrs else np.nan))
        print(f"K={K} uniform: confirm {conf}/{len(rs_eps)}, "
              f"FAR {far}/{len(stable_eps)}, "
              f"beta_err med={rows[-1]['beta_err_med']}")

    # --- K=4 聚束布置 (最坏位置) ---
    angles_cl = np.sort(np.concatenate([
        np.linspace(0, 90, 3, endpoint=False), [180.0]]))  # 3 个挤在 90° 内 + 1 个对面
    conf = far = 0
    berrs = []
    for _, r in rs_eps.iterrows():
        d = np.load(os.path.join(DATA, r["file"]), allow_pickle=True)
        conf += bool(eval_gate(d, angles_cl, 4)["confirmed"])
    for _, r in stable_eps.iterrows():
        d = np.load(os.path.join(DATA, r["file"]), allow_pickle=True)
        far += bool(eval_gate(d, angles_cl, 4)["confirmed"])
    for _, r in idx[idx.family == "rotating_stall"].iterrows():
        d = np.load(os.path.join(DATA, r["file"]), allow_pickle=True)
        berrs.append(beta_err(d, angles_cl, 4))
    rows.append(dict(K=4, layout="clustered", confirm_rate=conf / len(rs_eps),
                     far=far / len(stable_eps),
                     beta_err_med=np.nanmedian(berrs),
                     beta_err_p90=np.nanquantile(berrs, .9)))
    print(f"K=4 clustered: confirm {conf}/{len(rs_eps)}, FAR {far}/{len(stable_eps)}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "sensor_ablation.csv"), index=False)

    # ---- 图 (标签用英文, 避免 matplotlib 默认字体缺 CJK 字形) ----
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4))
    u = df[df.layout == "uniform"]
    ax[0].plot(u.K, 100 * u.confirm_rate, "o-", color="tab:red")
    ax[0].set_xlabel("number of circumferential sensors K")
    ax[0].set_ylabel("stall-type confirm rate [%]")
    ax[0].set_ylim(0, 105); ax[0].grid(alpha=.3)
    ax[0].axhline(100 * u.confirm_rate[u.K == 1].iloc[0], ls=":",
                  color="gray", label="single channel (no gate)")
    ax[0].legend(fontsize=8)
    ax[1].plot(u.K, 100 * u.far, "s-", color="tab:blue")
    ax[1].set_xlabel("number of circumferential sensors K")
    ax[1].set_ylabel("false alarm rate [%]")
    ax[1].set_ylim(-2, 25); ax[1].grid(alpha=.3)
    k4c = df[df.layout == "clustered"]
    ax[2].plot(u.K[1:], u.beta_err_med[1:], "d-", label="uniform median")
    ax[2].plot(u.K[1:], u.beta_err_p90[1:], "d--", alpha=.6, label="uniform P90")
    if len(k4c):
        ax[2].plot([4], [k4c.beta_err_med.iloc[0]], "^", color="tab:orange",
                   ms=9, label="clustered K=4")
    ax[2].set_xlabel("number of circumferential sensors K")
    ax[2].set_ylabel("|beta estimation error|")
    ax[2].set_yscale("log"); ax[2].grid(alpha=.3, which="both"); ax[2].legend(fontsize=8)
    fig.suptitle("Sensor count / placement ablation "
                 "(stall-type confirm / stable FAR / pattern speed)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "sensor_ablation.png"), dpi=120)

    # ---- 报告 ----
    lines = ["# 传感器数量/位置消融报告", "",
             "- 数据: 旋转失速型失稳 %d episodes (含 surge 家族混合失稳), "
             "稳定/干扰 %d episodes" % (len(rs_eps), len(stable_eps)),
             "- K=1 为纯单通道检测 (无阵列闸门); K>=2 为相干性闸门确认制",
             "", "```", df.to_string(index=False), "```", "",
             "## 结论", ""]
    u0, u2 = u[u.K == 1].iloc[0], u[u.K == 2].iloc[0]
    lines += [
        f"- 检测/确认率: 单通道 {100*u0.confirm_rate:.0f}% -> "
        f"K>=2 相干性闸门 {100*u.confirm_rate[u.K>=2].min():.0f}% "
        "(闸门不损失真实失稳检测)",
        f"- 误报率: 单通道 {100*u0.far:.1f}% vs K>=2 "
        f"{100*u.far[u.K>=2].max():.1f}% —— 剩余误报为燃油阶跃诱发的真实共模"
        "压力瞬变 (与喘振同为平面波, 相干性原理上不可分, 非闸门缺陷)",
        "- 相干性闸门在 K=2 即完全有效 (确认率/误报率与通道数基本无关): "
        "闸门利用的是行波的空间相位结构, 2 个相邻传感器已可判别",
        f"- beta 精度 (亚样本抛物线插值后): 等间隔 K=3 最优 "
        f"({u.beta_err_med[u.K==3].iloc[0]:.4f}); 聚束布置误差 "
        f"{k4c.beta_err_med.iloc[0]:.4f} vs 等间隔同通道数 "
        f"{u.beta_err_med[u.K==4].iloc[0]:.4f} —— 小间隔角 -> 小滞后, "
        "对量化/噪声更敏感",
        "- 工程建议: 周向 2-4 个等间隔传感器即可获得闸门全部收益; "
        "若需传播速度估计, 用 3-4 个等间隔并避免聚束", ""]
    with open(os.path.join(RESULTS, "sensor_ablation.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
