# -*- coding: utf-8 -*-
"""
第三轮审稿补强:
A. R4-1: runs declustering 敏感性 (间隔 >= r 窗的 exceedance 视为新簇,
   只保留簇内最大值重拟合 GPD -> K_auto@p=1e-3 变化) + p_t/episode 率经验映射
B. R4-2: GPD 拟合 Q-Q 图 (保存 results/gpd_qq.png)
C. R3-1: order-resampled CNN (每转 32 点重采样, 16 转窗) 从头训练跨速
D. R3-3: LSTM (window 协议) 跨速微调/从头训练, 补 Table 2 的 RNN 行
"""
import os
import sys
import glob
import csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.stats import genpareto
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from false_alarm_eval import eval_point, STEADY_DIR
from tuda_adapter import read_raw_mat, decimate
from dl_baseline import CNN1D, LSTMNet
from dl_cross_domain import tuda_windows, FS_MODEL, WIN, HOP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STALL = {"N100": "Stall_Speed100.16_Kulite_data.mat",
         "N65": "Stall_Speed65.03_Kulite_data.mat"}


def decluster(exc_vals, r):
    """runs declustering: 相邻 exceedance 间隔 <= r 个窗视为一簇, 取簇内最大"""
    if len(exc_vals) == 0:
        return exc_vals
    clusters, cur = [], [exc_vals[0]]
    for v in exc_vals[1:]:
        if v[0] - cur[-1][0] <= r:
            cur.append(v)
        else:
            clusters.append(cur)
            cur = [v]
    clusters.append(cur)
    return np.array([max(c, key=lambda t: t[1]) for c in clusters])


def part_a_declustering():
    print("=" * 76)
    print("A. runs declustering 敏感性 (p_target=1e-3)")
    files = sorted(glob.glob(os.path.join(STEADY_DIR, "*.mat")))
    recs = [eval_point(f) for f in files]
    pooled = np.concatenate([r["fus"] for r in recs])
    u0 = np.quantile(pooled, 0.90)
    idx = np.where(pooled > u0)[0]
    exc = [(i, pooled[i] - u0) for i in idx]
    print(f"  raw exceedances: {len(exc)}")
    for r in (3, 5, 10, 20):
        dec = decluster(exc, r)
        vals = dec[:, 1] if dec.ndim == 2 else np.array([v for _, v in dec])
        c, _, s = genpareto.fit(vals, floc=0)
        # 重解 K@p=1e-3: 用去聚簇后的超出率
        rate_u = len(vals) / len(pooled)
        p_cond = 1e-3 / rate_u
        K = u0 + genpareto.ppf(1 - p_cond, c, loc=0, scale=s)
        K_emp = float(np.quantile(pooled, 1 - 1e-3))
        print(f"  r={r:2d} 窗: 簇数={len(vals)}, c={c:.3f}, scale={s:.3f}, "
              f"K_gpd={K:.3f} (K_emp={K_emp:.3f})")
    # p_t -> episode 率经验映射: 单窗超 K 率 vs 5 连窗 episode 率 (26 点)
    print("\n  p_t -> episode 率经验映射 (26 稳态点):")
    for K in (1.6, 2.195):
        exc_win = 0
        ep = 0
        n_win = 0
        for r in recs:
            fus = r["fus"]
            exc_win += int((fus > K).sum())
            n_win += len(fus)
            hit = fus > K
            cnt = 0
            for e in hit:
                cnt = cnt + 1 if e else 0
                if cnt >= 5:
                    ep += 1
                    break
        print(f"    K={K}: 单窗超阈率 {exc_win/n_win:.2e}, "
              f"episode 率 (点) {ep}/26")


def part_b_qq():
    print("\nB. GPD Q-Q 图")
    files = sorted(glob.glob(os.path.join(STEADY_DIR, "*.mat")))
    recs = [eval_point(f) for f in files]
    pooled = np.concatenate([r["fus"] for r in recs])
    u0 = np.quantile(pooled, 0.90)
    exc = np.sort(pooled[pooled > u0] - u0)
    c, _, s = genpareto.fit(exc, floc=0)
    n = len(exc)
    p = (np.arange(1, n + 1) - 0.5) / n
    q_theo = genpareto.ppf(p, c, loc=0, scale=s)
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot(q_theo, exc, "o", ms=3, color="#3B4992", label="exceedances")
    lim = [0, max(q_theo.max(), exc.max()) * 1.05]
    ax.plot(lim, lim, "--", color="#BB0021", lw=1, label="identity")
    ax.set_xlabel("GPD fitted quantile")
    ax.set_ylabel("empirical exceedance")
    ax.set_title(f"GPD Q-Q plot (c={c:.3f}, scale={s:.3f}, n={n})",
                 fontsize=9)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = os.path.join(ROOT, "results", "gpd_qq.png")
    fig.savefig(out, dpi=150)
    print(f"  已保存: {out}  (max exceedance {exc.max():.3f}, "
          f"max fitted quantile {q_theo.max():.3f})")


def order_resample(p, fs, f_rotor, n_ppr=32):
    """角度域重采样: 每转 n_ppr 点 (线性插值)"""
    t = np.arange(len(p)) / fs
    theta = 2 * np.pi * f_rotor * t          # 累计相位
    n_rev = int(theta[-1] / (2 * np.pi))
    th_new = np.arange(0, n_rev, 1.0 / n_ppr) * 2 * np.pi
    return np.interp(th_new, theta, p), n_ppr * f_rotor  # 等效采样率


def part_c_order_resampled_cnn():
    print("\nC. order-resampled CNN (每转 32 点, 16 转窗) 从头训练跨速")
    torch = __import__("torch")
    from dl_baseline import FS_DS
    rows = []
    for src, tgt in (("N65", "N100"), ("N100", "N65")):
        ep_s = read_raw_mat(os.path.join(ROOT, "tuda_data", "open_source",
                                         STALL[src]))
        ep_t = read_raw_mat(os.path.join(ROOT, "tuda_data", "open_source",
                                         STALL[tgt]))
        p_s, fs_s = decimate(ep_s["P"][:, 0], ep_s["fs"], FS_DS)
        p_t, fs_t = decimate(ep_t["P"][:, 0], ep_t["fs"], FS_DS)
        po_s, fs_os = order_resample(p_s, fs_s, ep_s["f_rotor"])
        po_t, fs_ot = order_resample(p_t, fs_t, ep_t["f_rotor"])
        # 16 转窗 (与检测器 20 转窗同级), hop 8 转
        win = 16 * 32
        hop = 8 * 32
        from dl_cross_domain import tuda_windows
        # 目标 GT 用 tuda_windows 的 gt (与原任务一致)
        _, tw_t, y_t, gt_t = tuda_windows(tgt)
        t_win_ord = (np.arange(len(po_t) - win + 1)[::hop] + win) / fs_ot
        # 源标签: 与原任务同语义 (窗末 >= GT - 1s 为正)
        _, tw_s, y_s, gt_s = tuda_windows(src)
        Xs, ys = [], []
        i0s = list(range(0, len(po_s) - win + 1, hop))
        for i0 in i0s:
            t_end = (i0 + win) / fs_os
            Xs.append(po_s[i0:i0 + win])
            ys.append(1 if t_end >= gt_s - 1.0 else 0)
        Xs = np.stack(Xs)[:, :, None].astype(np.float32)
        ys = np.array(ys, dtype=np.float32)
        print(f"  src {src}: order windows {len(ys)} (pos {int(ys.sum())})")
        leads, fps, alarms = [], [], []
        for seed in (1, 2, 3):
            torch.manual_seed(2026 + seed)
            rs = np.random.default_rng(seed)
            perm = rs.permutation(len(ys))
            mu, sd = Xs.mean(), Xs.std() + 1e-9
            Xn = ((Xs - mu) / sd).astype(np.float32)
            model = CNN1D()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            yt = torch.tensor(ys)
            w = torch.tensor([len(ys) / (2 * max(ys.sum(), 1.0))])
            lossf = torch.nn.BCEWithLogitsLoss(pos_weight=w)
            Xt = torch.tensor(Xn).transpose(1, 2)
            model.train()
            for epn in range(30):
                for i in range(0, len(ys), 8):
                    ii = perm[i:i + 8] if i + 8 <= len(ys) else perm[i:]
                    if len(ii) == 0:
                        continue
                    opt.zero_grad()
                    loss = lossf(model(Xt[ii]), yt[ii])
                    loss.backward()
                    opt.step()
            # 目标转速流评估 (逐窗 window 协议标准化)
            model.eval()
            probs, tws = [], []
            with torch.no_grad():
                for i0 in range(0, len(po_t) - win + 1, hop):
                    w0 = po_t[i0:i0 + win].astype(np.float32)
                    w0 = (w0 - w0.mean()) / (w0.std() + 1e-9)
                    pr = float(torch.sigmoid(model(torch.tensor(
                        w0[None, :, None]).transpose(1, 2))))
                    probs.append(pr)
                    tws.append((i0 + win) / fs_ot)
            probs = np.array(probs)
            tws = np.array(tws)
            hit = probs > 0.5
            alarm = tws[np.argmax(hit)] if hit.any() else None
            lead = (gt_t - alarm) if alarm is not None else None
            pre = tws < (gt_t - 1.0)
            fp = float(np.mean(hit[pre])) if pre.any() else None
            alarms.append(alarm)
            if lead is not None:
                leads.append(lead)
            if fp is not None:
                fps.append(fp)
        r = dict(src=src, tgt=tgt,
                 alarm=np.mean([a for a in alarms if a]) if any(alarms) else np.nan,
                 lead=np.mean(leads) if leads else np.nan,
                 fp=np.mean(fps) if fps else np.nan)
        rows.append(r)
        print(f"  -> {tgt}: 报警 {r['alarm']:.2f}s (GT {gt_t:.2f}), "
              f"FP_pre {r['fp']*100 if np.isfinite(r['fp']) else float('nan'):.0f}%")
    out = os.path.join(ROOT, "results", "order_resampled_cnn.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["src", "tgt", "alarm_s", "lead_s", "fp_pre"])
        for r in rows:
            w.writerow([r["src"], r["tgt"],
                        f"{r['alarm']:.3f}" if np.isfinite(r["alarm"]) else "None",
                        f"{r['lead']:.3f}" if np.isfinite(r["lead"]) else "None",
                        f"{r['fp']:.3f}" if np.isfinite(r["fp"]) else ""])
    print(f"  CSV: {out}")


def part_d_lstm_cross_speed():
    print("\nD. LSTM (window 协议) 跨速: 微调 + 从头训练")
    torch = __import__("torch")
    from dl_norm_ablation import train, eval_stream, norm_protocol
    ckpt = torch.load(os.path.join(ROOT, "results", "dl_baseline.pt"),
                      map_location="cpu", weights_only=False)
    MU_G, SD_G = float(ckpt["mu"]), float(ckpt["sd"])
    rows = []
    for task in ("finetune", "scratch"):
        for src, tgt in (("N65", "N100"), ("N100", "N65")):
            p_s, tw_s, y_s, gt_s = tuda_windows(src)
            p_t, tw_t, y_t, gt_t = tuda_windows(tgt)
            Xall = np.stack([p_s[int(i * HOP):int(i * HOP) + WIN]
                             for i in range(len(tw_s))])[:, :, None].astype(np.float32)
            gmu, gsd = Xall.mean(), Xall.std() + 1e-9
            if task == "finetune":
                pos_idx = np.where((y_s == 1) & (tw_s < gt_s + 0.5))[0]
                neg_idx = np.where((y_s == 0) & (tw_s > 1.0) & (tw_s < gt_s - 1.0))[0]
                idx = list(pos_idx) + list(neg_idx[:2 * len(pos_idx)])
                Xtr = np.stack([p_s[int(i * HOP):int(i * HOP) + WIN]
                                for i in idx])[:, :, None].astype(np.float32)
                ytr = np.array([y_s[i] for i in idx], dtype=np.float32)
                lr, epochs = 1e-4, 15
            else:
                Xtr, ytr = Xall, y_s
                lr, epochs = 1e-3, 30
            leads, fps, alarms = [], [], []
            for seed in (1, 2, 3):
                torch.manual_seed(2026 + seed)
                Xn = (Xtr - MU_G) / SD_G if False else norm_protocol(Xtr, "window")
                model = LSTMNet()
                if task == "finetune":
                    model.load_state_dict(ckpt["lstm"])
                model = train(model, Xn, ytr, lr=lr, epochs=epochs)
                alarm, lead, fp = eval_stream(model, p_t, tw_t, gt_t,
                                              "window", MU_G, SD_G)
                alarms.append(alarm)
                if lead is not None:
                    leads.append(lead)
                if fp is not None:
                    fps.append(fp)
            r = dict(task=task, src=src, tgt=tgt,
                     alarm=np.mean([a for a in alarms if a]) if any(alarms) else np.nan,
                     lead=np.mean(leads) if leads else np.nan,
                     fp=np.mean(fps) if fps else np.nan)
            rows.append(r)
            print(f"  [{task} {src}->{tgt}] 报警 "
                  f"{r['alarm']:.2f}s (GT {gt_t:.2f}), FP_pre "
                  f"{r['fp']*100 if np.isfinite(r['fp']) else float('nan'):.0f}%")
    out = os.path.join(ROOT, "results", "lstm_cross_speed.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["task", "src", "tgt", "alarm_s", "lead_s", "fp_pre"])
        for r in rows:
            w.writerow([r["task"], r["src"], r["tgt"],
                        f"{r['alarm']:.3f}" if np.isfinite(r["alarm"]) else "None",
                        f"{r['lead']:.3f}" if np.isfinite(r["lead"]) else "None",
                        f"{r['fp']:.3f}" if np.isfinite(r["fp"]) else ""])
    print(f"  CSV: {out}")


if __name__ == "__main__":
    part_a_declustering()
    part_b_qq()
    part_c_order_resampled_cnn()
    part_d_lstm_cross_speed()
