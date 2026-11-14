# -*- coding: utf-8 -*-
"""
检测 -> 避喘 闭环仿真 (准在线分块检测 + MG 动态模型 + 节气门执行机构)

架构:
  逐 xi 步 RK4 积分 MG 三状态 -> 每个信号采样点合成机匣压力样本
  -> 每 DET_BLOCK_S 秒用 SurgeDetector.from_rotor (零重整定) 对
  "至今为止的信号"评估 -> 报警 -> 节气门开度回开到安全值 g_safe
  -> OP 退回稳定区

检测器选择: 准在线分块 WOLA 检测器 (块延迟 0.5s << 前兆提前量 ~2.4s)。
不用在线 IIR 版的原因: 单级 IIR 带通对 <1.5Hz 近临界前兆衰减 >20dB
(带宽 3.9 个倍频程无法用单个 biquad 矩形化), 实测暴露其漏报。

对照: 开环 (报警不动作) 完成同样的节流 maneuver -> 失稳。
关键物理结论 (仿真证实): stall 团成核后 ~0.16s 内充分发展, 阀门动作
(0.3-1s) 物理上来不及 —— 避喘必须依赖失稳前兆 (近临界低频振荡) 触发。
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

CALIB = dict(psi_c0=-1.259, H=0.778, W=0.178, lc=4.0, a=0.30, m=1.75)
PHI0, PHI_NS = 0.399, 0.357
XI_PER_SEC = 50.0
DT = 4e-3                 # RK4 步 (xi)
XI_END = 400.0            # 8 s
FS_SIG = 1250.0           # 信号采样率 = 每 10 RK4 步 1 样本 = XI_PER_SEC/(10*DT)
K_P, A_RS, NOISE = 10000.0, 350.0, 5.0
RATE_REC = 1.0            # 节气门恢复速率 [gammaT/s] (现代防喘阀 <1s 行程)
N_SEEDS = 3
DET_BLOCK_S = 1.0          # 准在线检测块 (延迟; 需 >= 检测器喘振窗 0.6s)


def gamma_schedule_manifold(xi, g0, g_end, alarm_action=None):
    """节流 maneuver (smoothstep 1-6s), alarm_action=(t_alarm, g_safe, rate)
    时报警后回开"""
    xi_ramp0, xi_ramp1 = 50.0, 300.0
    g = np.full_like(np.atleast_1d(xi) * 1.0, g0)
    frac = np.clip((xi - xi_ramp0) / (xi_ramp1 - xi_ramp0), 0.0, 1.0)
    frac = 0.5 * (1.0 - np.cos(np.pi * frac))
    g = g0 + frac * (g_end - g0)
    if alarm_action is not None:
        t_alarm, g_safe, rate = alarm_action
        # 报警时刻 (xi 域) 后按速率回开
        xi_alarm = t_alarm * XI_PER_SEC
        rec = np.clip((xi - xi_alarm) * XI_PER_SEC * rate, 0.0, 1.0) \
            if np.ndim(xi) else min(max((xi - xi_alarm) * XI_PER_SEC * rate, 0.0), 1.0)
        g = np.where(xi >= xi_alarm, g + rec * (g_safe - g), g)
    return g


def run_scenario(B, closed_loop, seed, f_rotor):
    """单场景: 逐 RK4 步积分 + 流式检测 + 可选节气门恢复"""
    rng = np.random.default_rng(seed)
    params = dict(CALIB, B=B, gammaT=0.5)
    psi0 = mg.compressor_char(PHI0, **{k: CALIB[k] for k in ("psi_c0", "H", "W")})
    g0 = PHI0 / np.sqrt(psi0)
    gNS_psi = mg.compressor_char(PHI_NS, **{k: CALIB[k] for k in ("psi_c0", "H", "W")})
    g_end = (PHI_NS / np.sqrt(gNS_psi)) * 0.97

    det = SurgeDetector.from_rotor(fs=FS_SIG, f_rotor=f_rotor)
    det_block_n = int(DET_BLOCK_S * FS_SIG)   # 准在线检测块长 (样本)
    p_buf = []                                # 至今为止的信号缓存
    alarm_t, alarm_type = None, None
    g_safe = g0

    steps = int(XI_END / DT) + 1
    y = np.array([PHI0, psi0, 1e-8])
    # 记录 (每 10 步 = 1 信号样本)
    n_sig = steps // 10
    rec = dict(t=np.zeros(n_sig), Phi=np.zeros(n_sig), Psi=np.zeros(n_sig),
               J=np.zeros(n_sig), gammaT=np.zeros(n_sig), alarm=np.zeros(n_sig))
    p_rs_phase = rng.uniform(0, 2 * np.pi)
    isig = 0
    t_blk_last = 0.0

    def deriv(yi, gT):
        return mg.mg_deriv(yi.reshape(3, 1), B, np.array([gT]),
                           {k: np.array([params[k]]) for k in
                            ("psi_c0", "H", "W", "lc", "a", "m", "B")}
                           ).ravel()

    for i in range(steps - 1):
        xi = i * DT
        # 节气门: 开环按调度; 闭环在报警后回开
        if closed_loop and alarm_t is not None:
            rec_frac = min(max((xi / XI_PER_SEC - alarm_t) * RATE_REC, 0.0), 1.0)
            gT = g_alarm + rec_frac * (g_safe - g_alarm)
        else:
            gT = float(gamma_schedule_manifold(xi, g0, g_end))
        # RK4
        k1 = deriv(y, gT)
        k2 = deriv(y + 0.5 * DT * k1, gT)
        k3 = deriv(y + 0.5 * DT * k2, gT)
        k4 = deriv(y + DT * k3, gT)
        y = y + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        y[2] = max(y[2], 1e-12)
        y[0] = max(y[0], -0.35)

        if (i + 1) % 10 == 0 and isig < n_sig:
            ts = xi / XI_PER_SEC
            Psi, J = y[1], max(y[2], 0.0)
            p = Psi * K_P + A_RS * np.sqrt(J) * np.cos(
                2 * np.pi * 0.45 * f_rotor * ts + p_rs_phase) \
                + rng.normal(0.0, NOISE)
            rec["t"][isig], rec["Phi"][isig] = ts, y[0]
            rec["Psi"][isig], rec["J"][isig] = Psi, J
            rec["gammaT"][isig] = gT
            rec["alarm"][isig] = 1.0 if alarm_t is not None else 0.0
            # 准在线分块检测: 每 DET_BLOCK_S 对全部历史信号评估一次
            p_buf.append(p)
            if len(p_buf) >= det_block_n and alarm_t is None:
                p_hist = np.array(p_buf)
                st = det.run(np.arange(len(p_hist)) / FS_SIG, p_hist)
                if st["alarm_time"] is not None:
                    alarm_t = float(st["alarm_time"])
                    alarm_type = st["alarm_type"]
                    g_alarm = gT       # 记录触发时刻的开度
                    # 锁定检测器状态 (触发后不再评估)
            isig += 1

    J_max = float(rec["J"].max())
    Phi_min = float(rec["Phi"].min())
    ok = (J_max < 0.05) and (Phi_min > 0.0)
    return dict(rec=rec, alarm_t=alarm_t, alarm_type=alarm_type,
                J_max=J_max, Phi_min=Phi_min, avoided=bool(ok),
                g0=g0, g_end=g_end)


def main():
    print("=" * 76)
    print("检测 -> 避喘 闭环仿真 (在线检测器 from_rotor 零重整定)")
    print(f"  执行机构: 报警后节气门回开速率 {RATE_REC} gammaT/s")
    print(f"  失稳判据: J_max >= 0.05 或 Phi 倒流; 成功 = 两者都不发生")
    print("=" * 76)
    print(f"\n{'scenario':30s} {'mode':8s} {'seed':>4s} {'alarm_t':>8s} "
          f"{'type':11s} {'J_max':>7s} {'Phi_min':>8s} {'result':>8s}")

    scenarios = [("B=0.42 (stall-dominated)", 0.42, 333.9),
                 ("B=0.65 (surge-dominated)", 0.65, 333.9)]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.5))
    summary = {}
    for (label, B, f_rotor), (axo, axc) in zip(scenarios, axes):
        rows = []
        for seed in range(1, N_SEEDS + 1):
            r_open = run_scenario(B, False, seed, f_rotor)
            r_cl = run_scenario(B, True, seed, f_rotor)
            for r, mode in [(r_open, "open"), (r_cl, "closed")]:
                print(f"{label:30s} {mode:8s} {seed:>4d} "
                      f"{r['alarm_t'] if r['alarm_t'] else float('nan'):8.2f} "
                      f"{str(r['alarm_type']):11s} {r['J_max']:7.3f} "
                      f"{r['Phi_min']:8.3f} "
                      f"{'AVOIDED' if r['avoided'] else 'STALLED':>8s}")
            rows.append((r_open, r_cl))
        # 图: 开环 vs 闭环 (第一个 seed)
        r_open, r_cl = rows[0]
        for ax, r, tag, col in [(axo, r_open, "OPEN LOOP", "#c43b3b"),
                                (axc, r_cl, "CLOSED LOOP", "#2a9d5c")]:
            rc = r["rec"]
            ax.plot(rc["t"], rc["J"], color=col, lw=1.2, label="J (stall amp^2)")
            ax.plot(rc["t"], rc["Psi"] / 5, color="#555", lw=0.7, alpha=0.7,
                    label="Psi/5")
            ax.plot(rc["t"], rc["gammaT"], color="#3b74c4", lw=1,
                    label="gammaT (valve)")
            if r["alarm_t"]:
                ax.axvline(r["alarm_t"], color="#c43b3b", ls=":", lw=1.5,
                           label=f"alarm {r['alarm_t']:.2f}s ({r['alarm_type']})")
            ax.set_title(f"{label} - {tag}", fontsize=9)
            ax.set_xlabel("t [s]")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, loc="upper left")
        summary[label] = rows

    fig.tight_layout()
    out = os.path.join(ROOT, "results", "closed_loop_anti_surge.png")
    fig.savefig(out, dpi=140)
    print(f"\n图已保存: {out}")

    n_open = sum(1 for rows in summary.values() for o, _ in rows if not o["avoided"])
    n_cl_ok = sum(1 for rows in summary.values() for _, c in rows if c["avoided"])
    n_tot = sum(len(rows) for rows in summary.values())
    print(f"\n==== 汇总: 开环失稳 {n_open}/{n_tot}, 闭环避喘成功 {n_cl_ok}/{n_tot} ====")


if __name__ == "__main__":
    main()
