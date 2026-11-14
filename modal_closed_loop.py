# -*- coding: utf-8 -*-
"""
P2-8 模态波扩展验证 + P2-9 闭环重定位 (方案 A: 前兆触发 + 快执行机构)

Part A (P2-8): 模态波扩展模型开环 episode -> 检测器零重整定检出
  验证: 模态波 (0.5×f_rotor, 特性线峰值前增长) 是否被归一化失速带
  在 J 成核**之前**捕获 -> 前兆提前量 = t_alarm - t_stall(J>0.05)

Part B (P2-9): 同模型闭环避喘
  报警 -> 节气门回开 (rate_rec), 成功判据 J_max<0.05 且 Phi_min>0
  对照 MG 三状态闭环 0/6 的归因: 前兆 (模态波) 触发使报警提前 ~1-2s,
  给执行机构留出动作时间 -> 避喘成功率应显著改善
用法: python src/modal_closed_loop.py
"""
import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mg_model as mg
import mg_modal as mm
from detector import SurgeDetector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CALIB = dict(psi_c0=-1.259, H=0.778, W=0.178, lc=4.0, a=0.30, m=1.75)
PHI0, PHI_NS = 0.399, 0.357
XI_PER_SEC = 50.0
DT = 4e-3
XI_END = 400.0                     # 8 s
FS_SIG = 1250.0
K_P, A_RS, NOISE = 10000.0, 350.0, 5.0
DET_BLOCK_S = 1.0
N_SEEDS = 3
GT_STALL_J = 0.05


def run_scenario(B, closed_loop, seed, f_rotor, rate_rec=1.0):
    """四状态 (Phi,Psi,J,A) 逐步 RK4 + 准在线分块检测 + 节气门恢复"""
    rng = np.random.default_rng(seed)
    params = dict(CALIB)
    psi0 = mg.compressor_char(PHI0, **{k: CALIB[k]
                                       for k in ("psi_c0", "H", "W")})
    g0 = PHI0 / np.sqrt(psi0)
    gNS = mg.compressor_char(PHI_NS, **{k: CALIB[k]
                                        for k in ("psi_c0", "H", "W")})
    g_end = (PHI_NS / np.sqrt(gNS)) * 0.97

    det = SurgeDetector.from_rotor(fs=FS_SIG, f_rotor=f_rotor)
    blk_n = int(DET_BLOCK_S * FS_SIG)
    p_buf = []
    alarm_t, alarm_type = None, None

    steps = int(XI_END / DT) + 1
    y = np.array([PHI0, psi0, 1e-12, mm.MODAL_PARAMS["a_floor"]])
    n_sig = steps // 10
    rec = {k: np.zeros(n_sig) for k in
           ("t", "Phi", "Psi", "J", "A", "gammaT", "alarm")}
    ph1 = rng.uniform(0, 2 * np.pi)
    ph2 = rng.uniform(0, 2 * np.pi)
    isig = 0
    xi_ramp0, xi_ramp1 = 50.0, 300.0   # 1-6s smoothstep (与闭环 0/6 同款)

    for i in range(steps - 1):
        xi = i * DT
        t_s = xi / XI_PER_SEC
        if closed_loop and alarm_t is not None:
            rec_frac = min(max((t_s - alarm_t) * rate_rec, 0.0), 1.0)
            gT = g_alarm + rec_frac * (g0 - g_alarm)
        else:
            frac = np.clip((xi - xi_ramp0) / (xi_ramp1 - xi_ramp0), 0.0, 1.0)
            frac = 0.5 * (1.0 - np.cos(np.pi * frac))
            gT = g0 + frac * (g_end - g0)
        # RK4 (四状态)
        p_all = {**params, "W": CALIB["W"]}
        k1 = mm.modal_deriv(y, B, gT, p_all)
        k2 = mm.modal_deriv(y + 0.5 * DT * k1, B, gT, p_all)
        k3 = mm.modal_deriv(y + 0.5 * DT * k2, B, gT, p_all)
        k4 = mm.modal_deriv(y + DT * k3, B, gT, p_all)
        y = y + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        y[2] = max(y[2], 1e-12)
        y[3] = max(y[3], mm.MODAL_PARAMS["a_floor"])
        y[0] = max(y[0], -0.35)

        if (i + 1) % 10 == 0 and isig < n_sig:
            Psi, J, A = y[1], max(y[2], 0.0), y[3]
            p = (Psi * K_P
                 + A_RS * np.sqrt(J) * np.cos(2 * np.pi * 0.45 * f_rotor * t_s + ph1)
                 + mm.A_M * A * np.cos(2 * np.pi * mm.F_MODAL_FRAC * f_rotor * t_s + ph2)
                 + rng.normal(0.0, NOISE))
            rec["t"][isig], rec["Phi"][isig] = t_s, y[0]
            rec["Psi"][isig], rec["J"][isig] = Psi, J
            rec["A"][isig], rec["gammaT"][isig] = A, gT
            rec["alarm"][isig] = 1.0 if alarm_t is not None else 0.0
            p_buf.append(p)
            if len(p_buf) >= blk_n and alarm_t is None:
                p_hist = np.asarray(p_buf)
                st = det.run(np.arange(len(p_hist)) / FS_SIG, p_hist)
                if st["alarm_time"] is not None:
                    alarm_t = float(st["alarm_time"])
                    alarm_type = st["alarm_type"]
                    g_alarm = gT
            isig += 1

    # 轨迹 GT (物理时间)
    t_modal = rec["t"][np.argmax(rec["A"] > mm.MODAL_PARAMS["A_gt"])] \
        if (rec["A"] > mm.MODAL_PARAMS["A_gt"]).any() else None
    t_stall = rec["t"][np.argmax(rec["J"] > GT_STALL_J)] \
        if (rec["J"] > GT_STALL_J).any() else None
    J_max = float(rec["J"].max())
    Phi_min = float(rec["Phi"].min())
    A_peak = float(rec["A"].max())
    return dict(rec=rec, alarm_t=alarm_t, alarm_type=alarm_type,
                t_modal=t_modal, t_stall=t_stall,
                J_max=J_max, Phi_min=Phi_min, A_peak=A_peak,
                avoided=bool(J_max < GT_STALL_J and Phi_min > 0.0))


def main():
    print("=" * 76)
    print("P2-8 模态波前兆检测 + P2-9 闭环重定位 (modal-extended MG)")
    print("=" * 76)

    scenarios = [("B=0.42 (stall-dominated)", 0.42, 333.9),
                 ("B=0.65 (surge-dominated)", 0.65, 333.9)]

    # ---- Part A: 开环 -> 离线检测 -> 前兆提前量 ----
    print("\n---- Part A: 模态波前兆检测 (开环 episode, 离线检测) ----")
    print(f"{'scenario':26s} {'seed':>4s} {'t_modal_GT':>10s} {'t_stall_GT':>10s} "
          f"{'alarm':>7s} {'type':11s} {'提前量':>7s}")
    partA = []
    for label, B, fr in scenarios:
        for seed in range(1, N_SEEDS + 1):
            r = run_scenario(B, False, seed, fr)
            alarm = r["alarm_t"]
            lead = (r["t_stall"] - alarm) if (alarm and r["t_stall"]) else None
            partA.append(dict(scenario=label, seed=seed, **{
                k: r[k] for k in ("t_modal", "t_stall", "alarm_t",
                                  "alarm_type")}, lead=lead))
            print(f"{label:26s} {seed:>4d} "
                  f"{r['t_modal'] or float('nan'):10.2f} "
                  f"{r['t_stall'] or float('nan'):10.2f} "
                  f"{alarm or float('nan'):7.2f} {str(r['alarm_type']):11s} "
                  f"{lead if lead else float('nan'):7.2f}")

    # ---- Part B: 闭环避喘 ----
    print("\n---- Part B: 闭环避喘 (前兆触发 -> 节气门回开) ----")
    print(f"{'scenario':26s} {'seed':>4s} {'rate':>5s} {'alarm_t':>8s} "
          f"{'J_max':>7s} {'Phi_min':>8s} {'result':>8s}")
    partB = []
    for label, B, fr in scenarios:
        for seed in range(1, N_SEEDS + 1):
            for rate in (1.0, 3.0):
                r_open = run_scenario(B, False, seed, fr, rate)
                r_cl = run_scenario(B, True, seed, fr, rate)
                for r, mode in ((r_open, "open"), (r_cl, "closed")):
                    partB.append(dict(scenario=label, seed=seed, rate=rate,
                                      mode=mode, alarm_t=r["alarm_t"],
                                      alarm_type=r["alarm_type"],
                                      J_max=r["J_max"], Phi_min=r["Phi_min"],
                                      avoided=r["avoided"]))
                print(f"{label:26s} {seed:>4d} {rate:>5.1f} "
                      f"{r_cl['alarm_t'] or float('nan'):8.2f} "
                      f"{r_cl['J_max']:7.3f} {r_cl['Phi_min']:8.3f} "
                      f"{'AVOIDED' if r_cl['avoided'] else 'STALLED':>8s}")

    n_cl = [r for r in partB if r["mode"] == "closed"]
    n_ok = sum(r["avoided"] for r in n_cl)
    print(f"\n==== 汇总 (Part B): 闭环避喘成功 {n_ok}/{len(n_cl)} "
          f"(MG 三状态基线为 0/6) ====")

    # ---- CSV ----
    out_csv = os.path.join(ROOT, "results", "modal_closed_loop.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["part", "scenario", "seed", "rate", "mode", "alarm_t",
                    "alarm_type", "t_modal_GT", "t_stall_GT", "lead_s",
                    "J_max", "Phi_min", "A_peak", "avoided"])
        for r in partA:
            w.writerow(["A", r["scenario"], r["seed"], "", "open",
                        r["alarm_t"], r["alarm_type"], r["t_modal"],
                        r["t_stall"], r["lead"], "", "", "", ""])
        for r in partB:
            w.writerow(["B", r["scenario"], r["seed"], r["rate"], r["mode"],
                        r["alarm_t"], r["alarm_type"], "", "", "",
                        r["J_max"], r["Phi_min"], "", r["avoided"]])
    print(f"CSV 已保存: {out_csv}")

    # ---- 图 ----
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.5))
    for (label, B, fr), (axo, axc) in zip(scenarios, axes):
        r_open = run_scenario(B, False, 1, fr, 3.0)
        r_cl = run_scenario(B, True, 1, fr, 3.0)
        for ax, r, tag, col in ((axo, r_open, "OPEN", "#c43b3b"),
                                (axc, r_cl, "CLOSED", "#2a9d5c")):
            rc = r["rec"]
            ax.plot(rc["t"], rc["J"], color=col, lw=1.2, label="J (stall)")
            ax.plot(rc["t"], rc["A"] * 10, color="#8e44ad", lw=1.0,
                    label="A x10 (modal wave)")
            ax.plot(rc["t"], rc["gammaT"], color="#3b74c4", lw=1,
                    label="gammaT (valve)")
            if r["t_modal"]:
                ax.axvline(r["t_modal"], color="#8e44ad", ls=":", lw=1,
                           label=f"modal GT {r['t_modal']:.2f}s")
            if r["alarm_t"]:
                ax.axvline(r["alarm_t"], color="#c43b3b", ls="--", lw=1.2,
                           label=f"alarm {r['alarm_t']:.2f}s")
            ax.set_title(f"{label} - {tag}", fontsize=9)
            ax.set_xlabel("t [s]")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    out_png = os.path.join(ROOT, "results", "modal_closed_loop.png")
    fig.savefig(out_png, dpi=140)
    print(f"图已保存: {out_png}")


if __name__ == "__main__":
    main()
