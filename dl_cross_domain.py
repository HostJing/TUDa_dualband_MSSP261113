# -*- coding: utf-8 -*-
"""
P0-4 DL 对照补全 —— few-shot 跨域微调 + 窗级 PR-AUC + 经典能量检测基线

内容:
  A. 窗级 PR-AUC: CNN/LSTM 在 MG held-out 与 TUDa-calib 窗上的
     precision-recall (补齐 dl_baseline.py 只有事件级指标的缺口)
  B. few-shot 跨域: MG 预训练 CNN 用源转速 stall run 的 n_shot 个标注窗
     微调 (n=10/20/50, 3 seeds), 测目标转速 -> 恢复代价曲线
     (报警时刻 vs GT, 虚警率), 双向 N65->N100 与 N100->N65
  C. 经典基线: 固定绝对阈值能量检测器 (阈值由 26 稳态点标定,
     2x 最大稳态失速带 RMS) —— 零学习基线的公平对照

产出: results/dl_cross_domain.png / dl_cross_domain.csv
用法: python src/dl_cross_domain.py
"""
import os
import sys
import glob
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dl_baseline import (CNN1D, LSTMNet, load_episodes, make_windows,
                         predict, FS_DS, T_LEAD, SEED)
from tuda_adapter import (read_raw_mat, decimate,
                          pressure_fluctuation_criterion, FS_RAW)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STALL_FILES = {"N100": "Stall_Speed100.16_Kulite_data.mat",
               "N65": "Stall_Speed65.03_Kulite_data.mat"}
FS_MODEL = 1000.0            # 与 dl_baseline 一致
WIN = int(0.5 * FS_MODEL)
HOP = int(0.25 * FS_MODEL)


# ---------------- 数据 ----------------
def tuda_windows(line, gt_cache={}):
    """stall run -> (p_1kHz, tw, labels, gt) 标签语义与 dl_baseline 一致"""
    if line in gt_cache:
        return gt_cache[line]
    ep = read_raw_mat(os.path.join(ROOT, "tuda_data", "open_source",
                                   STALL_FILES[line]))
    p_ds, fs_ds = decimate(ep["P"], ep["fs"], FS_MODEL)
    p = p_ds[:, 0]
    t = np.arange(len(p)) / fs_ds
    gt = pressure_fluctuation_criterion(
        dict(t=t, P=p_ds, p=p, fs=fs_ds, f_rotor=ep["f_rotor"]),
        k_sigma=6.0, win_s=0.05)
    tw, y = [], []
    for i in range(0, len(p) - WIN, HOP):
        t_end = (i + WIN) / fs_ds
        tw.append(t_end)
        y.append(1 if (gt is not None and t_end >= gt - T_LEAD) else 0)
    out = (p, np.array(tw), np.array(y, dtype=np.float32), gt)
    gt_cache[line] = out
    return out


def norm_windows(p, mu, sd):
    return ((p - mu) / sd).astype(np.float32)


def eval_stream(model, p_n, tw, gt, armed_s=1.0):
    """按时间流评估: tw[k] 对应窗起点 k*HOP; 返回 (alarm_t, lead, fp_rate_pre)"""
    ks = np.arange(len(tw))
    probs = np.array([
        predict(model, norm_windows(p_n[k * HOP:k * HOP + WIN], MU_G, SD_G)
                [None, :, None], "cnn")[0]
        for k in ks])
    hit = probs > 0.5
    alarm = tw[ks][np.argmax(hit)] if hit.any() else None
    lead = (gt - alarm) if (alarm is not None and gt) else None
    pre = (tw[ks] >= armed_s) & (tw[ks] < (gt - T_LEAD if gt else len(p_n) / FS_MODEL))
    fp = float(np.mean(hit[pre])) if pre.any() else None
    return alarm, lead, fp


# ---------------- 全局归一化 (与预训练一致) ----------------
MU_G, SD_G = None, None


def main():
    global MU_G, SD_G
    torch.set_num_threads(max(os.cpu_count() - 1, 1))
    ckpt = torch.load(os.path.join(ROOT, "results", "dl_baseline.pt"),
                      map_location="cpu", weights_only=False)
    MU_G, SD_G = float(ckpt["mu"]), float(ckpt["sd"])

    # ============ A. 窗级 PR-AUC ============
    print("=" * 76)
    print("A. 窗级 PR-AUC (MG held-out 与 TUDa-calib)")
    try:
        from sklearn.metrics import average_precision_score
        ap = lambda y, s: average_precision_score(y, s)
    except ImportError:
        def ap(y, s):
            o = np.argsort(-s); y = np.asarray(y)[o]
            cum = np.cumsum(y) / (np.arange(len(y)) + 1)
            return float((cum * y).sum() / max(y.sum(), 1))
    eps = load_episodes()
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(eps))
    n_te = max(len(eps) // 5, 4)
    test_eps = [e for i, e in enumerate(eps) if i in set(idx[:n_te].tolist())]
    pr_rows = []
    for name, cls in (("CNN", CNN1D), ("LSTM", LSTMNet)):
        model = cls()
        model.load_state_dict(ckpt["cnn" if name == "CNN" else "lstm"])
        for tag, eps_set in (("MG-holdout", test_eps),):
            ys, ss = [], []
            for e in eps_set:
                X, y, _ = make_windows(e)
                if len(X) == 0:
                    continue
                ss.append(predict(model, ((X - MU_G) / SD_G).astype(np.float32),
                                  name.lower()))
                ys.append(y)
            y = np.concatenate(ys); s = np.concatenate(ss)
            a = ap(y, s)
            pr_rows.append((name, tag, a))
            print(f"  {name} {tag}: AP( window-level )={a:.3f} "
                  f"(正窗 {int(y.sum())}/{len(y)})")

    # ============ B. few-shot 跨域 ============
    print("\nB. few-shot 跨域微调 (MG 预训练 -> 源转速标注窗 -> 目标转速)")
    rows = []
    for src, tgt in (("N65", "N100"), ("N100", "N65")):
        p_s, tw_s, y_s, gt_s = tuda_windows(src)
        p_t, tw_t, y_t, gt_t = tuda_windows(tgt)
        pos_idx = np.where((y_s == 1) & (tw_s < gt_s + 0.5))[0]
        neg_idx = np.where((y_s == 0) & (tw_s > 1.0) & (tw_s < gt_s - T_LEAD))[0]
        for n_shot in (10, 20, 50):
            leads, fps, alarms = [], [], []
            for seed in (1, 2, 3):
                rs = np.random.default_rng(seed)
                pi = rs.choice(pos_idx, size=min(n_shot, len(pos_idx)),
                               replace=False)
                ni = rs.choice(neg_idx, size=min(n_shot, len(neg_idx)),
                               replace=False)
                model = CNN1D()
                model.load_state_dict(ckpt["cnn"])
                # 微调 (整网, 小 lr, 少 epoch)
                opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        # 收集微调窗
                Xs, ys = [], []
                for i in list(pi) + list(ni):
                    i0 = int(i * HOP)
                    Xs.append(p_s[i0:i0 + WIN])
                    ys.append(y_s[i])
                Xs = np.stack(Xs)[:, :, None].astype(np.float32)
                Xs_n = ((Xs - MU_G) / SD_G).astype(np.float32)
                ys = np.array(ys, dtype=np.float32)
                Xt = torch.tensor(Xs_n).transpose(1, 2)
                yt = torch.tensor(ys)
                w = torch.tensor([len(ys) / (2 * max(ys.sum(), 1.0))])
                lossf = torch.nn.BCEWithLogitsLoss(pos_weight=w)
                model.train()
                for epn in range(15):
                    perm = torch.randperm(len(ys))
                    for i in range(0, len(ys), 8):
                        ii = perm[i:i + 8]
                        opt.zero_grad()
                        loss = lossf(model(Xt[ii]), yt[ii])
                        loss.backward()
                        opt.step()
                # 目标转速评估
                alarm, lead, fp = eval_stream(model, p_t, tw_t, gt_t)
                alarms.append(alarm)
                if lead is not None:
                    leads.append(lead)
                if fp is not None:
                    fps.append(fp)
            rows.append(dict(src=src, tgt=tgt, n_shot=n_shot,
                             alarm=np.mean([a for a in alarms if a]),
                             lead=np.mean(leads) if leads else np.nan,
                             lead_sd=np.std(leads) if leads else np.nan,
                             fp=np.mean(fps) if fps else np.nan,
                             det_rate=len(leads) / 3))
            r = rows[-1]
            print(f"  {src}->{tgt} n={n_shot}: 检出 {r['det_rate']}/3, "
                  f"报警 {r['alarm']:.2f}s (GT {gt_t:.2f}s, "
                  f"lead {r['lead']:.2f}±{r['lead_sd']:.2f}s), "
                  f"FP_pre {r['fp']*100:.0f}%")

    # ============ C. 经典固定阈值能量基线 ============
    print("\nC. 经典基线: 固定绝对阈值能量检测 (阈值=2x 稳态最大失速带 RMS)")
    from detector import SurgeDetector, fft_bandpass, sliding_rms
    thr = {}
    for line in ("N100", "N65"):
        levels = []
        for f in glob.glob(os.path.join(ROOT, "tuda_data", "steady",
                                        f"{line}_*.mat")):
            e = read_raw_mat(f)
            P_ds, fs_ds = decimate(e["P"], e["fs"], FS_DS)
            det = SurgeDetector.from_rotor(fs=fs_ds, f_rotor=e["f_rotor"])
            pb = fft_bandpass(P_ds[:, 0] - P_ds[:, 0].mean(), fs_ds,
                              *det.stall_band)
            _, r = sliding_rms(pb, fs_ds, det.win_s, det.hop_s)
            levels.append(np.median(r))
        thr[line] = 2.0 * max(levels)
        print(f"  [{line}] 阈值={thr[line]:.5f} (稳态最大中位 RMS "
              f"{max(levels):.5f} x2)")
    rows_c = []
    for line in ("N100", "N65"):
        p, tw, y, gt = tuda_windows(line)
        e = read_raw_mat(os.path.join(ROOT, "tuda_data", "open_source",
                                      STALL_FILES[line]))
        P_ds, fs_ds = decimate(e["P"], e["fs"], FS_DS)
        det = SurgeDetector.from_rotor(fs=fs_ds, f_rotor=e["f_rotor"])
        pb = fft_bandpass(P_ds[:, 0] - P_ds[:, 0].mean(), fs_ds, *det.stall_band)
        tw2, r = sliding_rms(pb, fs_ds, det.win_s, det.hop_s)
        exc = r > thr[line]
        cnt, alarm = 0, None
        for i, ee in enumerate(exc):
            cnt = cnt + 1 if ee else 0
            if cnt >= 5:
                alarm = tw2[i - 4]
                break
        lead = gt - alarm if alarm and gt else None
        rows_c.append(dict(line=line, alarm=alarm, lead=lead, gt=gt))
        print(f"  [{line}] alarm={alarm} (GT {gt}) -> "
              f"lead {lead:+.2f}s" if lead else f"  [{line}] alarm=None")

    # ============ 保存 ============
    out_csv = os.path.join(ROOT, "results", "dl_cross_domain.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["section", "key1", "key2", "key3", "value"])
        for name, tag, a in pr_rows:
            w.writerow(["PR-AUC", name, tag, "", f"{a:.3f}"])
        for r in rows:
            w.writerow(["few-shot", f"{r['src']}->{r['tgt']}",
                        f"n={r['n_shot']}", "det_rate", f"{r['det_rate']}/3"])
            w.writerow(["few-shot", f"{r['src']}->{r['tgt']}",
                        f"n={r['n_shot']}", "lead_mean_sd",
                        f"{r['lead']:.2f}±{r['lead_sd']:.2f}"])
            w.writerow(["few-shot", f"{r['src']}->{r['tgt']}",
                        f"n={r['n_shot']}", "fp_pre", f"{r['fp']:.3f}"])
        for r in rows_c:
            w.writerow(["classical", r["line"], "alarm", "",
                        f"{r['alarm']:.2f}" if r["alarm"] else "None"])
            w.writerow(["classical", r["line"], "lead", "",
                        f"{r['lead']:+.2f}" if r["lead"] else "None"])
    print(f"\nCSV 已保存: {out_csv}")

    # ---- 图: few-shot 恢复代价曲线 ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for tgt, c in (("N100", "#c43b3b"), ("N65", "#3b74c4")):
        pts = [(r["n_shot"], r["lead"]) for r in rows
               if r["tgt"] == tgt and np.isfinite(r["lead"])]
        pts.sort()
        ax.errorbar([p[0] for p in pts], [p[1] for p in pts],
                    yerr=[r["lead_sd"] for r in rows if r["tgt"] == tgt
                          and np.isfinite(r["lead"])],
                    marker="o", color=c, label=f"fine-tune -> {tgt}")
        gt_t = tuda_windows(tgt)[3]
        ax.axhline(gt_t - 1.0, color=c, ls=":", lw=0.8, alpha=0.5)
    ax.set_xlabel("n_shot (labeled windows from source speed)")
    ax.set_ylabel("alarm lead vs GT [s]")
    ax.set_title("DL cross-domain recovery cost (MG pretrain + few-shot)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(ROOT, "results", "dl_cross_domain.png")
    fig.savefig(out_png, dpi=150)
    print(f"图已保存: {out_png}")


if __name__ == "__main__":
    main()
