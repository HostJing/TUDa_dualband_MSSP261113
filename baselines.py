# -*- coding: utf-8 -*-
"""
CNN / LSTM 深度学习对比基线 (torch CPU)

任务设定 (与经典检测器同一评测协议):
  - 输入: 滑动窗口 p(t), 0.25 s @ 1 kHz (250 点), 窗口标签 = 窗口末端的 ground truth 状态
      0=stable, 1=pre-stall(先兆), 2=stall/surge(失稳发展)
  - 按工况 (episode) 划分训练/测试集, 避免窗口级泄漏
  - 评测: 测试工况上的窗口分类指标 + 事件级预警 (P(2) 持续超阈值 -> 报警),
      与经典检测器的 检出率/提前量/误报率 对齐比较

模型:
  - 1D-CNN:  Conv1d(1->16,k16,s4)+ReLU -> Conv1d(16->32,k8,s2)+ReLU -> GAP -> FC
  - LSTM:    2 层 skip 平均下采样 (250->125) -> LSTM(1->32) -> 末隐状态 -> FC
训练: Adam, 交叉熵 (类权重), 少量 epoch, CPU 即可
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

WIN_S = 0.25          # 窗口时长
FS_DS = 1000          # 窗口内降采样率 (原始 20 kHz -> 1 kHz)
STRIDE_S = 0.05       # 窗口滑动步长
SEED = 11

# ---------------------------------------------------------------------------
# 窗口数据集构建
# ---------------------------------------------------------------------------
def build_windows(idx):
    """返回 X (N,250) float32, y (N,), ep_id (N,) 供按工况切分"""
    win_len = int(WIN_S * FS_DS)          # 250
    stride = int(STRIDE_S * FS_DS)        # 50
    ds = 20                                # 20 kHz -> 1 kHz (FFT 低通抗混叠)
    X, y, ep = [], [], []
    for e, (_, r) in enumerate(idx.iterrows()):
        d = np.load(os.path.join(DATA, r["file"]), allow_pickle=True)
        t, p, lab = d["t"], d["p"], d["labels"]
        # FFT 低通 400 Hz 抗混叠后抽取
        P = np.fft.rfft(p)
        f = np.fft.rfftfreq(len(p), d=1.0 / float(d["fs"]))
        P[f > 400] = 0
        p_ds = np.fft.irfft(P, n=len(p))[::ds]
        lab_ds = lab[::ds]
        n_win = (len(p_ds) - win_len) // stride + 1
        for w in range(n_win):
            i0 = w * stride
            X.append(p_ds[i0:i0 + win_len])
            # 3 类: 0=stable, 1=pre-stall, 2=stall/surge(失稳发展, 合并标签 2/3)
            y.append(min(int(lab_ds[i0 + win_len - 1]), 2))
            ep.append(e)
    return (np.array(X, dtype=np.float32), np.array(y, dtype=np.int64),
            np.array(ep, dtype=np.int64))


def split_episodes(ep, y, seed=SEED, test_frac=0.3):
    """按 episode 分层 (按是否含失稳) 划分"""
    rng = np.random.default_rng(seed)
    eps = np.unique(ep)
    has_inst = {e: int((y[ep == e] > 0).any()) for e in eps}
    te = []
    for flag in (0, 1):
        pool = [e for e in eps if has_inst[e] == flag]
        rng.shuffle(pool)
        te += pool[:max(1, int(round(len(pool) * test_frac)))]
    te = set(te)
    tr = [e for e in eps if e not in te]
    return tr, sorted(te)


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
class CNN1D(nn.Module):
    def __init__(self, n_class=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 16, stride=4), nn.ReLU(),
            nn.Conv1d(16, 32, 8, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(32, n_class))

    def forward(self, x):
        return self.net(x)


class LSTMClf(nn.Module):
    def __init__(self, hidden=32, n_class=3):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, n_class)

    def forward(self, x):
        # (N,1,250) -> (N,125,1) skip 平均下采样
        z = x.squeeze(1)
        z = 0.5 * (z[:, ::2] + z[:, 1::2])
        z = z.unsqueeze(-1)
        out, _ = self.lstm(z)
        return self.fc(out[:, -1])


# ---------------------------------------------------------------------------
# 训练与评测
# ---------------------------------------------------------------------------
def train_model(model, Xtr, ytr, epochs=12, batch=64, lr=1e-3):
    rng = np.random.default_rng(SEED)
    counts = np.bincount(ytr, minlength=3).astype(float)
    w = torch.tensor(counts.sum() / np.maximum(counts, 1), dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss(weight=w)
    n = len(Xtr)
    for ep in range(epochs):
        perm = rng.permutation(n)
        tot, correct, lsum = 0, 0, 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = torch.from_numpy(Xtr[idx]).unsqueeze(1)
            yb = torch.from_numpy(ytr[idx])
            opt.zero_grad()
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
            lsum += float(loss) * len(idx)
            correct += int((out.argmax(1) == yb).sum())
            tot += len(idx)
        print(f"  epoch {ep+1:>2}: loss={lsum/tot:.4f} acc={correct/tot:.3f}")
    return model


def predict_proba(model, X, batch=512):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).unsqueeze(1)
            outs.append(torch.softmax(model(xb), dim=1).numpy())
    return np.concatenate(outs)


def evaluate_windows(ytrue, ypred):
    """逐类 precision/recall (宏平均由调用方汇总)"""
    res = {}
    for c, name in enumerate(["stable", "pre-stall", "instability"]):
        tp = int(((ytrue == c) & (ypred == c)).sum())
        fp = int(((ytrue != c) & (ypred == c)).sum())
        fn = int(((ytrue == c) & (ypred != c)).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        res[name] = dict(precision=prec, recall=rec, support=tp + fn)
    return res


def episode_early_warning(idx, ep_test, X, ep, proba, thr=0.6, n_sustain=3,
                          stride_s=STRIDE_S):
    """事件级预警评测: P(instability) 连续 n_sustain 窗 > thr -> 报警"""
    recs = []
    for e in ep_test:
        row = idx.iloc[e]
        m = ep == e
        pg = proba[m][:, 2] + proba[m][:, 1]  # 先兆或失稳
        alarm_idx = None
        cnt = 0
        for i, v in enumerate(pg):
            cnt = cnt + 1 if v > thr else 0
            if cnt >= n_sustain:
                alarm_idx = i
                break
        cand = [v for v in (row.get("t_surge", np.nan), row.get("t_stall", np.nan))
                if np.isfinite(v)]
        onset = min(cand) if cand else np.nan
        # 窗口末端时刻 ≈ 窗口起点 + WIN_S (窗口按 0.05 s 步进对齐到信号)
        alarm_t = None if alarm_idx is None else alarm_idx * stride_s + WIN_S
        if onset != onset:
            status = "FA" if alarm_t is not None else "ok"
        elif alarm_t is None:
            status = "none"
        else:
            status = "early" if alarm_t < onset else "late"
        lead = (onset - alarm_t) if (alarm_t is not None and onset == onset) else np.nan
        recs.append(dict(family=row["family"], B=row["B"], status=status, lead=lead))
    return pd.DataFrame(recs)


def main():
    torch.manual_seed(SEED)
    t0 = time.time()
    idx = pd.read_csv(os.path.join(DATA, "index.csv"))
    print("构建窗口数据集 ...")
    X, y, ep = build_windows(idx)
    tr_eps, te_eps = split_episodes(ep, y)
    mtr = np.isin(ep, tr_eps)
    mte = np.isin(ep, te_eps)
    # 标准化 (训练集统计)
    mu, sd = X[mtr].mean(), X[mtr].std() + 1e-9
    Xn = (X - mu) / sd
    Xtr, ytr = Xn[mtr], y[mtr]
    Xte, yte = Xn[mte], y[mte]
    print(f"窗口: 训练 {len(ytr)} (来自 {len(tr_eps)} 工况) / 测试 {len(yte)} "
          f"({len(te_eps)} 工况);  类分布 train={np.bincount(ytr)}, test={np.bincount(yte)}")

    results = {}
    for name, model in [("1D-CNN", CNN1D()), ("LSTM", LSTMClf())]:
        print(f"\n=== 训练 {name} ===")
        train_model(model, Xtr, ytr)
        pred = predict_proba(model, Xte).argmax(1)
        wres = evaluate_windows(yte, pred)
        acc = float((pred == yte).mean())
        results[name] = dict(windows=wres, acc=acc)
        print(f"{name} 窗口准确率 {acc:.3f}; " +
              "; ".join(f"{k}: P={v['precision']:.2f}/R={v['recall']:.2f}"
                        for k, v in wres.items()))
        # 事件级预警
        proba = predict_proba(model, Xte)
        ev = episode_early_warning(idx, te_eps, Xte, ep[mte], proba)
        inst = ev[ev.status.isin(["early", "late", "none"])]
        e = ev[ev.status == "early"]
        print(f"事件级: early {len(e)}/{len(inst)} ({100*len(e)/max(len(inst),1):.0f}%), "
              f"late {(ev.status=='late').sum()}, FAR {(ev.status=='FA').sum()}/"
              f"{(ev.status=='ok').sum()+(ev.status=='FA').sum()}")
        if len(e):
            print(f"  提前量中位 {1000*e['lead'].median():.0f} ms")
        ev.to_csv(os.path.join(RESULTS, f"dl_{name.replace('-','').lower()}_episodes.csv"),
                  index=False)
        results[name]["events"] = ev

    # 保存汇总
    summary = {k: dict(acc=v["acc"],
                       windows={kk: {kkk: (round(vvv, 4) if isinstance(vvv, float) else vvv)
                                     for kkk, vvv in vv.items()}
                                for kk, vv in v["windows"].items()})
               for k, v in results.items()}
    import json
    with open(os.path.join(RESULTS, "dl_baselines_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n总耗时 {time.time()-t0:.0f}s; 汇总 -> results/dl_baselines_summary.json")


if __name__ == "__main__":
    main()
