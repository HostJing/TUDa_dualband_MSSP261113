# -*- coding: utf-8 -*-
"""
预失速检测验证流程
1. 二维阈值网格扫描 (k_stall, k_surge), 在子集上选优:
   检出率最高且误报率 <= 5% 的组合
2. 用最优阈值对全数据集评估: 检出率 / 预警提前量 / 误报率
3. 输出 results/validation_report.md + 示例图 + per_episode_results.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from detector import SurgeDetector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

FAR_LIMIT = 0.05


def nanmed(s):
    s = s.dropna()
    return float(s.median()) if len(s) else float("nan")


def evaluate_episode(row, detector_kwargs=None):
    """对单个 episode 运行检测器, 返回评估记录 (不读 ground truth 状态进检测器)"""
    d = np.load(os.path.join(DATA, row["file"]), allow_pickle=True)
    t, p = d["t"], d["p"]
    det = SurgeDetector.from_rotor(fs=float(d["fs"]),
                                   f_rotor=float(d["f_rotor"]),
                                   **(detector_kwargs or {}))
    res = det.run(t, p)

    t_surge = row.get("t_surge", np.nan)
    t_stall = row.get("t_stall", np.nan)
    t_pre = row.get("t_pre", np.nan)
    cand = [v for v in (t_surge, t_stall) if np.isfinite(v)]
    onset = min(cand) if cand else np.nan

    rec = dict(
        file=row["file"], family=row["family"], B=row["B"],
        fuel_step=bool(row.get("fuel_step", False)),
        t_pre=t_pre, onset=onset,
        alarm_time=res["alarm_time"], alarm_type=res["alarm_type"],
    )
    if onset == onset:
        if res["alarm_time"] is None:
            rec["detection"], rec["lead"] = "missed", np.nan
        else:
            rec["lead"] = onset - res["alarm_time"]
            rec["detection"] = "early" if res["alarm_time"] < onset else "late"
    else:
        rec["lead"] = np.nan
        rec["detection"] = "false_alarm" if res["alarm_time"] is not None else "correct_reject"
    return rec, d, res


def sweep_thresholds(rows):
    """二维网格扫描, 返回 (best_kwargs, 全部扫描结果表)"""
    grid_ks = [3, 4, 5, 6, 8]
    grid_kg = [4, 5, 6, 8, 10, 12]
    out = []
    for ks in grid_ks:
        for kg in grid_kg:
            recs = [evaluate_episode(r, dict(k_stall=ks, k_surge=kg))[0]
                    for _, r in rows.iterrows()]
            df = pd.DataFrame(recs)
            inst = df[df["onset"].notna()]
            stable = df[df["onset"].isna()]
            out.append(dict(
                k_stall=ks, k_surge=kg,
                detect_rate=float((inst["detection"] == "early").mean()) if len(inst) else np.nan,
                late_rate=float((inst["detection"] == "late").mean()) if len(inst) else np.nan,
                miss_rate=float((inst["detection"] == "missed").mean()) if len(inst) else np.nan,
                false_alarm=float((stable["detection"] == "false_alarm").mean()) if len(stable) else np.nan,
                median_lead_s=nanmed(inst["lead"]),
            ))
    sw = pd.DataFrame(out)
    ok = sw[sw["false_alarm"] <= FAR_LIMIT]
    pick_from = ok if len(ok) else sw
    best = pick_from.sort_values(
        ["detect_rate", "median_lead_s"], ascending=[False, False]).iloc[0]
    return dict(k_stall=float(best["k_stall"]), k_surge=float(best["k_surge"])), sw


def main():
    idx = pd.read_csv(os.path.join(DATA, "index.csv"))
    print(f"数据集: {len(idx)} episodes")

    # --- 1) 阈值扫描 (子集) ---
    sub = idx.sample(min(len(idx), 36), random_state=1)
    best_kwargs, sw = sweep_thresholds(sub)
    sw.to_csv(os.path.join(RESULTS, "threshold_sweep.csv"), index=False)
    print(f"最优阈值: {best_kwargs}")
    print(sw.to_string(index=False))

    # --- 2) 全集评估 (最优阈值) ---
    recs, plots_done = [], set()
    for _, r in idx.iterrows():
        rec, d, res = evaluate_episode(r, best_kwargs)
        recs.append(rec)
        key = (r["family"], bool(r.get("fuel_step", False)))
        if key not in plots_done and rec["alarm_time"] is not None:
            plots_done.add(key)
            fig, ax = plt.subplots(3, 1, figsize=(11, 8))
            ax[0].plot(d["t"], d["p"], lw=0.4)
            ax[0].set_ylabel("p(t) [Pa]")
            ax[1].semilogy(d["t"], np.maximum(d["J"], 1e-12), lw=0.8)
            ax[1].set_ylabel("J (ground truth)")
            for tv, c, lb in [(rec["t_pre"], "orange", "pre-stall GT"),
                              (rec["onset"], "red", "onset GT")]:
                if tv == tv and np.isfinite(tv):
                    for a in ax[:2]:
                        a.axvline(tv, color=c, ls="--", label=lb)
            ax[2].semilogy(res["t_surge_win"],
                           np.maximum(res["r_surge"] / res["base_surge"], 1e-3),
                           label="surge-band RMS ratio")
            ax[2].semilogy(res["t_win"],
                           np.maximum(res["r_stall"] / res["base_stall"], 1e-3),
                           label="stall-band RMS ratio")
            ax[2].axhline(best_kwargs["k_surge"], color="darkred", ls=":", lw=0.8)
            ax[2].axhline(best_kwargs["k_stall"], color="red", ls=":", lw=0.8)
            if rec["alarm_time"] is not None:
                for a in ax:
                    a.axvline(rec["alarm_time"], color="k", ls="--", lw=1.2)
                ax[2].annotate("alarm", (rec["alarm_time"], ax[2].get_ylim()[1]),
                               fontsize=8)
            ax[2].set_xlabel("t [s]"); ax[2].set_ylabel("band RMS / baseline")
            ax[2].legend(fontsize=8)
            lead_s = f"{rec['lead']*1000:.0f} ms" if rec["lead"] == rec["lead"] else "n/a"
            fig.suptitle(
                f"{r['family']} (B={r['B']:.2f}, "
                f"{'with' if r.get('fuel_step', False) else 'no'} interference) "
                f"detection={rec['detection']}, lead={lead_s}")
            fig.tight_layout()
            fp = os.path.join(RESULTS,
                              f"example_{r['family']}_{'int' if r.get('fuel_step', False) else 'clean'}.png")
            fig.savefig(fp, dpi=120); plt.close(fig)

    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(RESULTS, "per_episode_results.csv"), index=False)

    # --- 3) 报告 ---
    inst = df[df["onset"].notna()]
    stable = df[df["onset"].isna()]
    early = inst[inst["detection"] == "early"]
    lines = ["# 预失速检测验证报告", "",
             f"- 数据集规模: {len(df)} episodes "
             f"(喘振 {len(inst[inst.family=='surge'])}, "
             f"旋转失速 {len(inst[inst.family=='rotating_stall'])}, "
             f"稳定/干扰 {len(stable)})",
             f"- 检测器最优阈值 (扫描选优, 误报率约束 {int(FAR_LIMIT*100)}%): "
             f"k_stall={best_kwargs['k_stall']}, k_surge={best_kwargs['k_surge']}",
             "", "## 总体指标", ""]
    if len(inst):
        lines += [
            f"- 失稳工况预警成功率: **{100*(inst['detection']=='early').mean():.1f}%** "
            f"({len(early)}/{len(inst)})",
            f"- 迟报: {(inst['detection']=='late').sum()}, "
            f"漏报: {(inst['detection']=='missed').sum()}",
            f"- 预警提前量中位数: **{nanmed(early['lead'])*1000:.0f} ms**"
            + (f" (均值 {early['lead'].mean()*1000:.0f} ms, "
               f"P25={early['lead'].quantile(.25)*1000:.0f} ms, "
               f"P75={early['lead'].quantile(.75)*1000:.0f} ms)" if len(early) else ""),
            "", "按失稳类型分组:", "",
            "```",
            str(inst.groupby("family").agg(
                n=("lead", "size"),
                early_rate=("detection", lambda s: f"{100*(s=='early').mean():.0f}%"),
                late=("detection", lambda s: int((s=='late').sum())),
                missed=("detection", lambda s: int((s=='missed').sum())),
                median_lead_ms=("lead", lambda s: f"{nanmed(s)*1000:.0f}"))),
            "```", ""]
    if len(stable):
        lines += [
            f"- 稳定/干扰工况误报率: **{100*(stable['detection']=='false_alarm').mean():.1f}%** "
            f"({(stable['detection']=='false_alarm').sum()}/{len(stable)})", ""]
    lines += ["## 方法说明", "",
              "- 检测器仅使用机匣壁面动态压力信号 p(t); ground truth 状态量 (Psi/Phi/J) 只用于打标签与评估。",
              "- ground truth: J>1e-3 判先兆; J>0.1 判失速发展; Φ<0(倒流) 或 Ψ 滑动 std>0.15H 判喘振。",
              "- 提前量 = 失稳起始时刻(t_surge 与 t_stall 较早者) − 报警时刻。",
              "- 阈值扫描: k_stall×k_surge 二维网格, 在 36 个 episode 子集上以"
              "『误报率≤5% 且 预警成功率最高』选优。", ""]
    report = "\n".join(lines)
    with open(os.path.join(RESULTS, "validation_report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
