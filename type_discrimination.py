# -*- coding: utf-8 -*-
"""
失稳类型早期区分: 报警之后, 尽早判定本次失稳将发展为旋转失速还是喘振

处置策略完全不同: 旋转失速可短时继续运行 (失速团可退出), 喘振必须立即收油门。
关键: 报警往往发生在失稳起始前数秒 (先兆触发), 报警瞬间特征尚无区分力 ——
因此特征取 [报警, 报警+delay] 窗口内的累积最大值 (失速团一旦出现, 相干性
立即跃升并保持; 深喘振则始终无相干结构而喘振带能量持续增长)。

特征:
  f1 = max 失速带 RMS / 基线
  f2 = max 喘振带 RMS / 基线
  f3 = max 失速带空间相干性   (旋转失速团为行波 -> 高; 深喘振 -> 低)
分类决策 (物理规则, 无训练):
  f3 > 0.4 -> stall (出现相干行波结构)
  否则 f2 > 4 -> surge (喘振带能量确认)
  都不满足 -> undecided
评估: 分类准确率 (已判定样本) + 判定覆盖率 随 delay 变化
输出 results/type_discrimination.md + type_discrimination.png
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from detector import SurgeDetector
from array_processing import synthesize_array, spatial_coherence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
DELAYS = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0]


def coh_at(tc, coh, t, win=0.5):
    m = (tc >= t - win / 2) & (tc <= t + win / 2)
    return float(np.median(coh[m])) if m.any() else np.nan


def main():
    idx = pd.read_csv(os.path.join(DATA, "index.csv"))
    rows = []
    for _, r in idx.iterrows():
        if r["label"] == "stable":
            continue
        gt_type = "stall" if r["label"] == "rotating_stall" else "surge"
        d = np.load(os.path.join(DATA, r["file"]), allow_pickle=True)
        fs, fr = float(d["fs"]), float(d["f_rotor"])
        arr = synthesize_array(d, n_sensors=8, seed=7)
        det = SurgeDetector.from_rotor(fs=fs, f_rotor=fr, k_stall=4, k_surge=5)
        res = det.run(arr["t"], arr["P"][:, 0])
        if res["alarm_time"] is None:
            continue
        t_al = res["alarm_time"]
        stall_band = tuple(s * fr for s in SurgeDetector.STALL_NORM)
        surge_band = tuple(s * fr for s in SurgeDetector.SURGE_NORM)
        tc_s, coh_s = spatial_coherence(arr["P"], fs, stall_band)
        for dl in DELAYS:
            m1 = (res["t_win"] >= t_al) & (res["t_win"] <= t_al + dl)
            f1 = float(np.max(res["r_stall"][m1] / res["base_stall"])) if m1.any() else 0.0
            m2 = (res["t_surge_win"] >= t_al) & (res["t_surge_win"] <= t_al + dl)
            f2 = float(np.max(res["r_surge"][m2] / res["base_surge"])) if m2.any() else 0.0
            m3 = (tc_s >= t_al) & (tc_s <= t_al + dl)
            f3 = float(np.max(coh_s[m3])) if m3.any() else 0.0
            if f3 > 0.4:
                pred = "stall"
            elif f2 > 4:
                pred = "surge"
            else:
                pred = "undecided"
            rows.append(dict(file=r["file"], gt=gt_type, delay=dl,
                             f1=f1, f2=f2, f3=f3, pred=pred))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "type_discrimination.csv"), index=False)

    # ---- 报告 ----
    n_eps = df.file.nunique()
    lines = ["# 失稳类型早期区分报告", "",
             f"- 样本: {n_eps} 个失稳 episode "
             f"(stall {df[df['gt']=='stall'].file.nunique()}, "
             f"surge {df[df['gt']=='surge'].file.nunique()})",
             "- 特征为报警以来累积最大值 (报警常在失稳起始前 ~2.8 s 触发, "
             "瞬时特征在先兆期无区分力)",
             "- 决策规则: f3 (失速带相干性) > 0.4 -> stall; 否则 f2 (喘振带"
             "能量比) > 4 -> surge; 都不满足 -> undecided", "",
             "## 判定准确率与覆盖率 vs 报警后延迟", "", "```"]
    acc = []
    for dl in DELAYS:
        sub = df[df.delay == dl]
        decided = sub[sub.pred != "undecided"]
        a = 100 * (decided.pred == decided["gt"]).mean() if len(decided) else np.nan
        cov = 100 * len(decided) / len(sub)
        acc.append((dl, a, cov))
        lines.append(f"delay={dl:3.1f}s  准确率 {a:5.1f}%  "
                     f"已判定覆盖率 {cov:5.1f}%  (n={len(sub)})")
    lines += ["```", ""]
    sub4 = df[df.delay == DELAYS[-1]]
    cm = pd.crosstab(sub4["gt"], sub4.pred)
    lines += ["## 混淆矩阵 (delay=%.1f s)" % DELAYS[-1], "", "```",
              str(cm), "```", "",
              "## 特征物理含义", "",
              "- f3 (失速带相干性): 行波失速团 -> 高 (~0.9); 深喘振无失速团 -> 低 (~0.1)",
              "- f1/f2 (频带能量比): 旋转失速 -> 失速带主导; 深喘振 -> 喘振带主导",
              "- 混合失稳 (先失速团后喘振) -> 正确判为 stall, 处置策略兼容", ""]
    last = acc[-1]
    lines += ["## 结论", "",
              f"- delay={last[0]:.1f} s 时: 覆盖率 {last[2]:.0f}%, "
              f"已判定样本准确率 {last[1]:.0f}%",
              "- 旋转失速型失稳的判定由相干性跃升驱动 (物理上确定); "
              "纯深喘振只能以'无相干结构 + 喘振带能量确认'排除法判定, "
              "判定时间受 dive 时刻支配",
              "- 处置联动: 判为 stall -> 允许短时观察; 判为 surge -> "
              "立即收油门; undecided -> 继续监视 (误处置风险低于误判)",
              ""]
    report = "\n".join(lines)
    with open(os.path.join(RESULTS, "type_discrimination.md"), "w",
              encoding="utf-8") as f:
        f.write(report)

    # ---- 图 ----
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    sub0 = df[df.delay == 2.0]
    for gt, c, m in [("stall", "tab:red", "o"), ("surge", "tab:blue", "s")]:
        s = sub0[sub0["gt"] == gt]
        ax[0].scatter(np.maximum(s.f1 / np.maximum(s.f2, 1e-3), 1e-3), s.f3,
                      c=c, marker=m, alpha=.7, label=gt)
    ax[0].axhline(0.4, color="k", ls=":", lw=1)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("max f1/f2 (失速带/喘振带能量比)")
    ax[0].set_ylabel("max f3 (失速带相干性)")
    ax[0].legend(fontsize=9); ax[0].grid(alpha=.3)
    ax[0].set_title("特征空间 (报警后 2 s 累积)")
    ax[1].plot([d for d, _, _ in acc], [c for _, _, c in acc], "^--",
               color="gray", label="已判定覆盖率")
    # 准确率仅在已判定样本上
    xs = [d for d, a, _ in acc]
    ys = [a if a == a else np.nan for _, a, _ in acc]
    ax[1].plot(xs, ys, "o-", color="tab:green", label="已判定样本准确率")
    ax[1].set_xlabel("报警后累积窗口 [s]")
    ax[1].set_ylabel("[%]"); ax[1].set_ylim(0, 105)
    ax[1].grid(alpha=.3); ax[1].legend(fontsize=9)
    ax[1].set_title("类型判定: 覆盖率与准确率")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "type_discrimination.png"), dpi=120)
    print(report)


if __name__ == "__main__":
    main()
