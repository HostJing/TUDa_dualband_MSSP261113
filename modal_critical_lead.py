# -*- coding: utf-8 -*-
"""
审稿修订 R5-2/R4-3: critical lead 扫描
对 x_m ∈ {1.10, 1.15, 1.18, 1.21} (对应开环 lead 约 0.26-1.27 s),
在两个 B 场景 × 5 seeds × rate 1.0 下跑闭环避喘,
统计成功率 vs lead, 给出避喘成功所需的最小提前量 (critical lead)。
"""
import os
import sys
import csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mg_modal as mm
import modal_closed_loop as mcl

ROOT = mcl.ROOT

SCENARIOS = [("B=0.42", 0.42, 333.9), ("B=0.65", 0.65, 333.9)]
X_M_LIST = [1.10, 1.15, 1.18, 1.21]
SEEDS = range(1, 6)
RATE = 1.0


def main():
    rows = []
    print(f"{'x_m':>5s} {'lead_open':>9s} {'B':>5s} {'success':>8s}")
    for x_m in X_M_LIST:
        # 1) 用开环 run 测该 x_m 的前兆提前量 (3 seeds 均值)
        mm.MODAL_PARAMS["x_m"] = x_m
        leads = []
        for label, B, fr in SCENARIOS:
            for seed in (1, 2, 3):
                r = mcl.run_scenario(B, False, seed, fr, RATE)
                if r["alarm_t"] and r["t_stall"]:
                    leads.append(r["t_stall"] - r["alarm_t"])
        lead_mean = float(np.mean(leads)) if leads else float("nan")

        # 2) 闭环避喘成功率
        n_ok, n_tot = 0, 0
        for label, B, fr in SCENARIOS:
            for seed in SEEDS:
                r = mcl.run_scenario(B, True, seed, fr, RATE)
                n_tot += 1
                n_ok += int(r["avoided"])
                rows.append(dict(x_m=x_m, lead_open=lead_mean, B=label,
                                 seed=seed, rate=RATE, alarm_t=r["alarm_t"],
                                 J_max=r["J_max"], Phi_min=r["Phi_min"],
                                 avoided=r["avoided"]))
        print(f"{x_m:>5.2f} {lead_mean:>9.3f} {'':>5s} {n_ok:>4d}/{n_tot}")

    # 汇总每个 x_m 的成功率
    print("\n==== 汇总: 避喘成功率 vs 前兆提前量 (rate=1.0) ====")
    summary = []
    for x_m in X_M_LIST:
        rs = [r for r in rows if r["x_m"] == x_m]
        ok = sum(r["avoided"] for r in rs)
        lm = rs[0]["lead_open"]
        summary.append((x_m, lm, ok, len(rs)))
        print(f"  x_m={x_m:.2f}  lead={lm:.2f}s  success {ok}/{len(rs)}")

    ok_lead = [lm for xm, lm, ok, n in summary if ok == n]
    fail_lead = [lm for xm, lm, ok, n in summary if ok < n]
    crit_lo = max(fail_lead) if fail_lead else None
    crit_hi = min(ok_lead) if ok_lead else None
    print(f"\ncritical lead 区间: ({crit_lo:.2f}, {crit_hi:.2f}) s"
          if (crit_lo is not None and crit_hi is not None)
          else "\ncritical lead 未包络")

    out_csv = os.path.join(ROOT, "results", "critical_lead_sweep.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["x_m", "lead_open_mean_s", "B", "seed", "rate",
                    "alarm_t", "J_max", "Phi_min", "avoided"])
        for r in rows:
            w.writerow([r["x_m"], f"{r['lead_open']:.3f}", r["B"], r["seed"],
                        r["rate"], r["alarm_t"], f"{r['J_max']:.4f}",
                        f"{r['Phi_min']:.4f}", r["avoided"]])
    print(f"CSV 已保存: {out_csv}")


if __name__ == "__main__":
    main()
