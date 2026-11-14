# -*- coding: utf-8 -*-
"""校准脚本: 验证 MG 模型四类典型工况动力学 + 信号综合效果"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mg_model as mg

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUT, exist_ok=True)

# 四类工况: 稳定(B小,节流不过界) / 旋转失速(B小,节流过界) / 深喘振(B大) / 混合失稳(B中)
scenarios = [
    dict(name="stable",          B=0.8, gammaT_start=0.68, gammaT_end=0.63),
    dict(name="rotating_stall",  B=0.5, gammaT_start=0.68, gammaT_end=0.55),
    dict(name="deep_surge",      B=2.0, gammaT_start=0.68, gammaT_end=0.55),
    dict(name="mixed",           B=1.0, gammaT_start=0.68, gammaT_end=0.55),
]

rng = np.random.default_rng(42)
params_list = [{**mg.DEFAULT_PARAMS, "a": 0.5, **s, "gammaT_end": s["gammaT_end"]} for s in scenarios]
sched = mg.make_throttle_schedule(0.68, 0.55, 10, 90)

# 各工况需要不同节气门终值 -> 用逐工况时间表
def sched_vec(xi, gammaT0):
    out = np.zeros_like(gammaT0)
    for i, s in enumerate(scenarios):
        gs, ge, x0, x1 = s["gammaT_start"], s["gammaT_end"], 10.0, 90.0
        if xi >= x1:
            out[i] = ge
        elif xi >= x0:
            f = (xi - x0) / (x1 - x0)
            out[i] = gs + f * (ge - gs)
        else:
            out[i] = gs
    return out

xi, Y, meta = mg.integrate(params_list, sched_vec, xi_end=200.0, dt=5e-4, J0=1e-4)
events = mg.event_ground_truth(xi, Y, meta, params_list)
sig = mg.synthesize_signal(xi, Y, meta, rng, xi_per_sec=50.0)

fig, axes = plt.subplots(4, 4, figsize=(18, 12))
for c, sc in enumerate(scenarios):
    ev = events[c]
    axes[0, c].plot(xi, Y[:, 0, c]); axes[0, c].set_title(f"{sc['name']} Phi (B={sc['B']})")
    axes[1, c].plot(xi, Y[:, 1, c]); axes[1, c].set_title("Psi")
    axes[2, c].plot(xi, Y[:, 2, c]); axes[2, c].set_yscale("log"); axes[2, c].set_title("J")
    axes[2, c].axhline(1e-4, color="gray", ls=":"); axes[2, c].axhline(0.10, color="red", ls=":")
    t, p = sig["t"], sig["p"][:, c]
    axes[3, c].plot(t[::20], p[::20], lw=0.5); axes[3, c].set_title("sensor p(t)")
    for k, col in [("t_pre", "orange"), ("t_stall", "red"), ("t_surge", "darkred")]:
        v = ev[k] / 50.0
        if not np.isnan(v):
            axes[3, c].axvline(v, color=col, ls="--")
    print(f"[{sc['name']:>15}] B={sc['B']}  t_pre={ev['t_pre']:.1f}  "
          f"t_stall={ev['t_stall']:.1f}  t_surge={ev['t_surge']:.1f}  label={ev['label']}")
plt.tight_layout()
fp = os.path.join(OUT, "calibration_scenarios.png")
plt.savefig(fp, dpi=120)
print("saved:", fp)
