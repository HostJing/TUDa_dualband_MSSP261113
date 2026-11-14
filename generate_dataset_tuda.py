# -*- coding: utf-8 -*-
"""
TUDa 校准版 MG 仿真数据集生成 + 检测管线复跑

校准来源 (见 docs/TUDa_边界条件核对.md 第 4 节, fit_mg_characteristic.py):
  - 特性线: 实测 100% 转速线拟合的等效 MG 参数
      calib_A (形状最优): psi_c0=-1.259, H=0.778, W=0.178
        (psi_c0<0 非物理, 但工作区 Φ∈[0.357,0.399] 形状最贴合实测;
         工作区外行为非物理, 检测验证只关心失稳前后的工作区动态)
      峰值: Φ_peak = 2W = 0.357, Ψ_peak = 0.298 (与实测 0.2939 差 0.004)
  - B 参数: TCD 2 台架官方估计 B=0.65 @ 设计转速
      (GPPS JGPPS 156119: duct+plenum 简化模型, B_crit < 0.65;
       OpenStage/TSV1 台架几何同型, plenum 同样很小)
      低转速按 B ∝ N 缩放: 65% -> B≈0.42
  - 工况: 设计点 ṁ=16 kg/s -> Φ0≈0.399, 节流至 NS Φ≈0.357
      稳态节气门 gammaT = Phi/sqrt(Psi)

内容:
  1. B 扫描 × 校准参数 -> 生成 episodes (含事件真值)
  2. 检测器 from_rotor (零重整定) 离线检测复跑
  3. 提前量报告 + 对比图
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mg_model as mg
from detector import SurgeDetector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- 校准参数 ----
CALIB = dict(psi_c0=-1.259, H=0.778, W=0.178, lc=4.0, a=0.30, m=1.75)
PHI0 = 0.399          # 设计点 (ṁ=16 kg/s 实测换算)
PHI_NS = 0.357        # 实测 NS 点 (= 2W, 特性线峰值即失速边界)
B_N100 = 0.65         # TCD2 官方估计 @ 100% 转速
B_N65 = 0.65 * 0.65   # B ∝ N
XI_PER_SEC = 50.0
F_ROTOR_N100 = 333.9  # Hz (官方换算, 与 BPF 核验一致)

# 信号合成幅值标定: 实测失速带事件 RMS ≈ 30x 基线 (TUDa N100 stall run)
# 仿真幅值保持原 dataset 整定 -> 检测器归一化常数不变


def gamma_design(phi0, psi0):
    return phi0 / np.sqrt(psi0)


def run_case(B, label, f_rotor, seed=42):
    """单个校准工况: 设计点 -> 节流 -> 失稳 -> 检测"""
    rng = np.random.default_rng(seed)
    params = dict(CALIB, B=B, gammaT=0.5)
    psi0 = mg.compressor_char(PHI0, **{k: CALIB[k] for k in ("psi_c0", "H", "W")})
    g0 = gamma_design(PHI0, psi0)
    # 节流: 从设计点缓关到 NS 附近再略过冲 (触发失稳)
    g_end = gamma_design(PHI_NS, mg.compressor_char(
        PHI_NS, **{k: CALIB[k] for k in ("psi_c0", "H", "W")})) * 0.97
    sched = mg.make_throttle_schedule(g0, g_end, xi_ramp_start=50.0,
                                      xi_ramp_end=300.0)
    # xi_end=400 -> 8s 物理时间 (匹配 TUDa stall run 时长);
    # 节流 1s->6s 缓变 (匹配真实试验的缓慢节流机动)
    xi, Y, meta = mg.integrate([params], sched, xi_end=400.0, dt=2e-3,
                               phi0=PHI0, j_floor=1e-12)
    sig = mg.synthesize_signal(xi, Y, meta, rng, xi_per_sec=XI_PER_SEC,
                               f_rotor=f_rotor, stall_frac=0.45,
                               K_p=10000.0, A_rs=350.0, noise_std=5.0,
                               seed_burst=False)  # 抗扰性已在原数据集验证,
                                                  # 此处目的为参数迁移验证
    ev = mg.event_ground_truth(xi, Y, meta, [dict(params, gammaT_end=g_end)])
    return dict(xi=xi, Y=Y, sig=sig, ev=ev[0], label=label, B=B,
                f_rotor=f_rotor, meta=meta)


def detect_and_report(case):
    """from_rotor 零重整定离线检测 -> 报警时刻 vs 真值"""
    sig = case["sig"]
    t = sig["t"]
    det = SurgeDetector.from_rotor(fs=1.0 / (t[1] - t[0]),
                                   f_rotor=case["f_rotor"])
    res = det.run(t, sig["p"][:, 0])
    ev = case["ev"]
    xi2s = case["xi"] / (case["xi"][-1] / t[-1])   # xi -> 秒 (线性)
    gt_pre = float(ev["t_pre"]) * (t[-1] / case["xi"][-1]) if np.isfinite(ev["t_pre"]) else None
    gt_stall = float(ev["t_stall"]) * (t[-1] / case["xi"][-1]) if np.isfinite(ev["t_stall"]) else None
    gt_surge = float(ev["t_surge"]) * (t[-1] / case["xi"][-1]) if np.isfinite(ev["t_surge"]) else None
    alarm = res["alarm_time"]
    ref = gt_pre if gt_pre is not None else gt_stall
    lead = (ref - alarm) if (alarm is not None and ref is not None) else None
    return dict(res=res, alarm=alarm, gt_pre=gt_pre, gt_stall=gt_stall,
                gt_surge=gt_surge, ref=ref, lead=lead, label=ev["label"])


def main():
    print("=" * 74)
    print("TUDa 校准版 MG 数据集: 特性线=实测拟合, B=TCD2 官方估计 0.65")
    print(f"  calib: psi_c0={CALIB['psi_c0']}, H={CALIB['H']}, W={CALIB['W']}")
    print(f"  峰值: Phi_peak=2W={2*CALIB['W']:.3f}, Psi_peak={CALIB['psi_c0']+2*CALIB['H']:.3f}")
    print(f"  工况: Phi0={PHI0} (设计点) -> 节流 -> Phi_NS={PHI_NS}")
    print("=" * 74)

    cases = [
        run_case(B_N100 * 0.65, "N100 B=0.42 (sub-critical)", F_ROTOR_N100, seed=11),
        run_case(B_N100, "N100 B=0.65 (official est.)", F_ROTOR_N100, seed=22),
        run_case(0.90, "B=0.90", F_ROTOR_N100, seed=33),
        run_case(1.20, "B=1.20 (classic surge expected)", F_ROTOR_N100, seed=44),
        run_case(B_N65, "N65 B=0.42", F_ROTOR_N100 * 0.65, seed=55),
    ]

    print(f"\n{'case':38s} {'type':14s} {'GT_pre':>8s} {'GT_stall':>8s} "
          f"{'GT_surge':>8s} {'alarm':>8s} {'lead':>7s} {'J_max':>7s} {'Phi_min':>8s}")
    fig, axes = plt.subplots(len(cases), 2, figsize=(12, 3.2 * len(cases)),
                             sharex=False)
    all_ok = True
    for i, (case, ax) in enumerate(zip(cases, axes)):
        r = detect_and_report(case)
        sig, xi = case["sig"], case["xi"]
        t = sig["t"]
        ev = case["ev"]
        ok = r["alarm"] is not None
        all_ok &= ok
        J_max = float(case["Y"][:, 2, 0].max())
        Phi_min = float(case["Y"][:, 0, 0].min())
        print(f"{case['label']:38s} {r['label']:14s} "
              f"{r['gt_pre'] if r['gt_pre'] is not None else float('nan'):8.2f} "
              f"{r['gt_stall'] if r['gt_stall'] is not None else float('nan'):8.2f} "
              f"{r['gt_surge'] if r['gt_surge'] is not None else float('nan'):8.2f} "
              f"{r['alarm'] if r['alarm'] is not None else float('nan'):8.2f} "
              f"{r['lead'] if r['lead'] is not None else float('nan'):+7.2f} "
              f"{J_max:7.3f} {Phi_min:8.3f}")
        # 左: p 信号; 右: 检测器失速带 RMS/基线 + 报警/真值
        ax[0].plot(t, sig["p"][:, 0], lw=0.5, color="#333")
        ax[0].set_title(f"{case['label']}  (B={case['B']:.2f}, type={r['label']})",
                        fontsize=9)
        ax[0].set_ylabel("p [Pa]")
        ax[0].grid(alpha=0.3)
        rn = r["res"]["r_stall"] / r["res"]["base_stall"]
        ax[1].semilogy(r["res"]["t_win"], rn, lw=1, color="#3b74c4",
                       label="stall-band RMS/base")
        ax[1].axhline(5.0, color="gray", ls="--", lw=0.8, label="k_stall=5")
        if r["alarm"]:
            ax[1].axvline(r["alarm"], color="#c43b3b", ls=":", lw=1.5,
                          label=f"alarm {r['alarm']:.2f}s")
        for gt, tag, col in [(r["gt_pre"], "pre", "green"),
                             (r["gt_stall"], "stall", "orange"),
                             (r["gt_surge"], "surge", "purple")]:
            if gt is not None:
                ax[1].axvline(gt, color=col, ls=":", lw=1)
                ax[1].text(gt, ax[1].get_ylim()[0] * 1.3, tag, fontsize=7,
                           color=col)
        ax[1].set_xlabel("t [s]")
        ax[1].set_ylabel("RMS/base")
        ax[1].legend(fontsize=7)
        ax[1].grid(alpha=0.3, which="both")

    fig.tight_layout()
    out = os.path.join(ROOT, "results", "tuda_calib_dataset_detection.png")
    fig.savefig(out, dpi=140)
    print(f"\n图已保存: {out}")
    print(f"总体: {'PASS (5/5 检出)' if all_ok else 'NEEDS REVIEW'}")

    # 保存 npz 供 DL 基线复用
    out_npz = os.path.join(ROOT, "results", "tuda_calib_dataset.npz")
    np.savez_compressed(
        out_npz,
        n_cases=len(cases),
        labels=[c["label"] for c in cases],
        **{f"case{i}_p": c["sig"]["p"][:, 0] for i, c in enumerate(cases)},
        **{f"case{i}_t": c["sig"]["t"] for i, c in enumerate(cases)},
        **{f"case{i}_J": c["sig"]["J"][:, 0] for i, c in enumerate(cases)},
        **{f"case{i}_ev": np.array([c["ev"]["label"]],
                                   dtype=object) for i, c in enumerate(cases)},
    )
    print(f"数据已保存: {out_npz} (DL 基线复用)")


if __name__ == "__main__":
    main()
