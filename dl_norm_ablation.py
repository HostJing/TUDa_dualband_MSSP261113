# -*- coding: utf-8 -*-
"""
审稿修订 R3-1/R3-2: DL 标准化协议消融 + TUDa 跨速从头训练

A. 标准化协议消融 (MG 预训练 CNN + 源转速 few-shot 微调 -> 目标转速):
   - global : 训练集全局均值/方差 (与 dl_baseline 预训练一致)
   - window : 每个输入窗用自身均值/方差标准化 (常用于分布偏移场景)
   每个协议报告: 目标转速报警/提前量/基线期虚警率, 以及
   微调窗自检 (训练窗上的正/负样本平均概率, 判断"恒报警"是
   实现问题还是真实分布偏移)。

B. TUDa 跨速从头训练 (与论文主张对齐的任务):
   源转速 stall run 全部窗训练 CNN (from scratch) -> 目标转速测试,
   同样两种协议, 3 seeds。

产出: results/dl_norm_ablation.csv
"""
import os
import sys
import csv
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dl_baseline import CNN1D, FS_DS
from dl_cross_domain import tuda_windows, FS_MODEL, WIN, HOP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STALL_FILES = {"N100": "Stall_Speed100.16_Kulite_data.mat",
               "N65": "Stall_Speed65.03_Kulite_data.mat"}
SEEDS = (1, 2, 3)


def predict(model, X):
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32).transpose(1, 2)
        return torch.sigmoid(model(Xt)).numpy()


def norm_protocol(X, proto):
    """X: (N, T, 1) float32. global 协议在调用方处理, window 协议逐窗标准化"""
    if proto == "window":
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True) + 1e-9
        return ((X - mu) / sd).astype(np.float32)
    return X.astype(np.float32)


def train(model, X_n, y, lr=1e-4, epochs=15, batch=8, pos_weight=True):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    yt = torch.tensor(y, dtype=torch.float32)
    w = torch.tensor([len(y) / (2 * max(y.sum(), 1.0))]) if pos_weight else None
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=w)
    Xt = torch.tensor(X_n).transpose(1, 2)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(len(y))
        for i in range(0, len(y), batch):
            ii = perm[i:i + batch]
            opt.zero_grad()
            loss = lossf(model(Xt[ii]), yt[ii])
            loss.backward()
            opt.step()
    return model


def eval_stream(model, p, tw, gt, proto, gmu=None, gsd=None):
    ks = np.arange(len(tw))
    probs = []
    for k in ks:
        w = p[k * HOP:k * HOP + WIN].astype(np.float32)[:, None]
        if proto == "window":
            w = (w - w.mean()) / (w.std() + 1e-9)
        elif gmu is not None:
            w = (w - gmu) / gsd
        probs.append(predict(model, w[None])[0])
    probs = np.array(probs)
    hit = probs > 0.5
    alarm = tw[ks][np.argmax(hit)] if hit.any() else None
    lead = (gt - alarm) if (alarm is not None and gt) else None
    pre = tw[ks] < (gt - 1.0)
    fp = float(np.mean(hit[pre])) if pre.any() else None
    return alarm, lead, fp


def windows_from(p, tw, y, idx):
    Xs, ys = [], []
    for i in idx:
        i0 = int(i * HOP)
        Xs.append(p[i0:i0 + WIN])
        ys.append(y[i])
    return np.stack(Xs)[:, :, None].astype(np.float32), np.array(
        ys, dtype=np.float32)


def main():
    torch.set_num_threads(max(os.cpu_count() - 1, 1))
    rows = []

    # ============ A. few-shot 微调协议消融 ============
    print("=" * 76)
    print("A. few-shot 微调: global vs window 标准化 (MG 预训练 CNN)")
    ckpt = torch.load(os.path.join(ROOT, "results", "dl_baseline.pt"),
                      map_location="cpu", weights_only=False)
    MU_G, SD_G = float(ckpt["mu"]), float(ckpt["sd"])
    print(f"预训练全局标准化参数: mu={MU_G:.4f}, sd={SD_G:.4f}")

    for src, tgt in (("N65", "N100"), ("N100", "N65")):
        p_s, tw_s, y_s, gt_s = tuda_windows(src)
        p_t, tw_t, y_t, gt_t = tuda_windows(tgt)
        pos_idx = np.where((y_s == 1) & (tw_s < gt_s + 0.5))[0]
        neg_idx = np.where((y_s == 0) & (tw_s > 1.0) & (tw_s < gt_s - 1.0))[0]
        print(f"\n源 {src}: pos {len(pos_idx)}, neg {len(neg_idx)}; "
              f"目标 {tgt}: GT {gt_t:.2f}s")
        n_shot = min(25, len(pos_idx))
        for proto in ("global", "window"):
            leads, fps, alarms, insamp = [], [], [], []
            for seed in SEEDS:
                rs = np.random.default_rng(seed)
                pi = rs.choice(pos_idx, size=min(n_shot, len(pos_idx)),
                               replace=False)
                ni = rs.choice(neg_idx, size=min(2 * n_shot, len(neg_idx)),
                               replace=False)
                Xs, ys = windows_from(p_s, tw_s, y_s, list(pi) + list(ni))
                Xs_n = (Xs - MU_G) / SD_G if proto == "global" \
                    else norm_protocol(Xs, "window")
                model = CNN1D()
                model.load_state_dict(ckpt["cnn"])
                model = train(model, Xs_n, ys)

                # 微调窗自检: 正/负样本平均概率 (in-sample sanity)
                with torch.no_grad():
                    Xt = torch.tensor(Xs_n).transpose(1, 2)
                    pr = torch.sigmoid(model(Xt)).numpy()
                insamp.append((float(pr[ys == 1].mean()),
                               float(pr[ys == 0].mean())))

                alarm, lead, fp = eval_stream(model, p_t, tw_t, gt_t,
                                              proto, MU_G, SD_G)
                alarms.append(alarm)
                if lead is not None:
                    leads.append(lead)
                if fp is not None:
                    fps.append(fp)
            r = dict(task="finetune", src=src, tgt=tgt, proto=proto,
                     alarm=np.mean([a for a in alarms if a]) if any(alarms)
                     else np.nan,
                     lead=np.mean(leads) if leads else np.nan,
                     fp=np.mean(fps) if fps else np.nan,
                     pos_prob=np.mean([x[0] for x in insamp]),
                     neg_prob=np.mean([x[1] for x in insamp]))
            rows.append(r)
            print(f"  [{proto:6s}] 报警 {r['alarm']:.2f}s "
                  f"(GT {gt_t:.2f}, lead {r['lead'] if np.isfinite(r['lead'])
                  else float('nan'):+.2f}s), FP_pre {r['fp']*100:.0f}%, "
                  f"in-sample P(pos)={r['pos_prob']:.3f} P(neg)={r['neg_prob']:.3f}")

    # ============ B. TUDa 跨速从头训练 ============
    print("\n" + "=" * 76)
    print("B. TUDa 跨速从头训练 (源转速 stall run 全部窗 -> 目标转速)")
    for src, tgt in (("N65", "N100"), ("N100", "N65")):
        p_s, tw_s, y_s, gt_s = tuda_windows(src)
        p_t, tw_t, y_t, gt_t = tuda_windows(tgt)
        Xall = np.stack([p_s[int(i * HOP):int(i * HOP) + WIN]
                         for i in range(len(tw_s))])[:, :, None]
        Xall = Xall.astype(np.float32)
        gmu, gsd = Xall.mean(), Xall.std() + 1e-9
        print(f"\n源 {src}: 全部窗 {len(tw_s)} (pos {int(y_s.sum())}), "
              f"全局 mu={gmu:.4f} sd={gsd:.4f}; 目标 {tgt}: GT {gt_t:.2f}s")
        for proto in ("global", "window"):
            leads, fps, alarms = [], [], []
            for seed in SEEDS:
                torch.manual_seed(2026 + seed)
                rs = np.random.default_rng(seed)
                perm = rs.permutation(len(tw_s))
                Xs_n = (Xall - gmu) / gsd if proto == "global" \
                    else norm_protocol(Xall, "window")
                model = CNN1D()
                model = train(model, Xs_n, y_s, lr=1e-3, epochs=30,
                              batch=8, pos_weight=True)
                alarm, lead, fp = eval_stream(model, p_t, tw_t, gt_t,
                                              proto, gmu, gsd)
                alarms.append(alarm)
                if lead is not None:
                    leads.append(lead)
                if fp is not None:
                    fps.append(fp)
            r = dict(task="scratch", src=src, tgt=tgt, proto=proto,
                     alarm=np.mean([a for a in alarms if a]) if any(alarms)
                     else np.nan,
                     lead=np.mean(leads) if leads else np.nan,
                     fp=np.mean(fps) if fps else np.nan,
                     pos_prob=np.nan, neg_prob=np.nan)
            rows.append(r)
            print(f"  [{proto:6s}] 报警 {r['alarm']:.2f}s "
                  f"(GT {gt_t:.2f}, lead {r['lead'] if np.isfinite(r['lead'])
                  else float('nan'):+.2f}s), FP_pre {r['fp']*100:.0f}%")

    out_csv = os.path.join(ROOT, "results", "dl_norm_ablation.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["task", "src", "tgt", "proto", "alarm_s", "lead_s",
                    "fp_pre", "insample_pos_prob", "insample_neg_prob"])
        for r in rows:
            w.writerow([r["task"], r["src"], r["tgt"], r["proto"],
                        f"{r['alarm']:.3f}" if np.isfinite(r["alarm"]) else "None",
                        f"{r['lead']:.3f}" if np.isfinite(r["lead"]) else "None",
                        f"{r['fp']:.3f}" if np.isfinite(r["fp"]) else "",
                        f"{r['pos_prob']:.3f}" if np.isfinite(r["pos_prob"]) else "",
                        f"{r['neg_prob']:.3f}" if np.isfinite(r["neg_prob"]) else ""])
    print(f"\nCSV 已保存: {out_csv}")


if __name__ == "__main__":
    main()
