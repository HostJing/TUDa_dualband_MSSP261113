# -*- coding: utf-8 -*-
"""
CNN / LSTM 深度学习对比基线 —— 与双频带归一化检测器对比

任务定义 (预失速预警, 与检测器报警语义对齐):
  输入: 机匣压力窗 p[t-w : t] (0.5s @ 1kHz, 降采样)
  标签: 未来 T_lead=1.0s 内是否发生失稳事件 (t_pre/t_surge 进入窗口)
  输出: 二分类概率; 推理时以概率 > 0.5 首次越顶作为 "DL 报警时刻"

模型 (torch, CPU):
  - CNN1D: 3 x (Conv1d + ReLU + MaxPool) + GlobalAvgPool + FC
  - LSTM:  2 层 LSTM(64) 取末隐状态 + FC

数据:
  训练/验证: MG 数据集 episode 级 8:2 分割 (data/*.npz, 66 例)
  测试: held-out episodes + TUDa 校准版 npz + TUDa 实测 stall run
  (跨台架测试体现泛化差异: DL 需重训 vs 归一化检测器零重整定)

对比指标:
  - 窗级 PR-AUC / F1
  - 事件检出提前量 (首个报警窗 相对 事件时刻)
  - 误报率 (stable episodes 上正窗占比)
"""
import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
XI_PER_SEC = 50.0
FS_DS = 1000.0          # 降采样目标
WIN_S = 0.5             # 输入窗
HOP_S = 0.25
T_LEAD = 1.0            # 预警提前窗口
EPOCHS = 12
BATCH = 64
LR = 1e-3
SEED = 2026


# ---------------- 数据准备 ----------------
def load_episodes():
    """读取 MG 数据集 episodes -> list[dict(p (T,), fs, f_rotor, t_event_s, label)]"""
    out = []
    for f in sorted(glob.glob(os.path.join(DATA_DIR, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        ev = z["event"].item() if z["event"].shape == () else z["event"]
        t_ev = None
        for key in ("t_pre", "t_surge"):
            v = float(ev[key]) if ev[key] is not None else np.nan
            if np.isfinite(v):
                t_ev = min(t_ev, v) if t_ev is not None else v
        out.append(dict(
            p=z["p"].astype(np.float32),
            fs=float(z["fs"]), f_rotor=float(z["f_rotor"]),
            t_event=(t_ev / XI_PER_SEC) if t_ev is not None else None,
            label=str(ev["label"]),
            name=os.path.basename(f),
        ))
    return out


def decimate_sig(p, fs, fs_new):
    factor = int(round(fs / fs_new))
    # 简易抗混叠: 因子长度滑动平均 (等价低通) + 抽取
    k = np.ones(factor) / factor
    return np.convolve(p, k, mode="valid")[::factor]


def make_windows(ep, label_offset_known=True):
    """episode -> (X, y, t_win) 窗级样本"""
    p = decimate_sig(ep["p"], ep["fs"], FS_DS)
    n = len(p)
    win = int(WIN_S * FS_DS)
    hop = int(HOP_S * FS_DS)
    X, y, tw = [], [], []
    t_ev = ep["t_event"]
    for i in range(0, n - win, hop):
        t_end = (i + win) / FS_DS               # 窗末时刻
        X.append(p[i:i + win])
        if t_ev is None:
            y.append(0)
        else:
            # 正类: 窗末已进入预警窗口 (事件前 T_LEAD) 或事件已发生
            y.append(1 if t_end >= t_ev - T_LEAD else 0)
        tw.append(t_end)
    return np.stack(X)[:, :, None], np.array(y, dtype=np.float32), np.array(tw)


# ---------------- 模型 ----------------
class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 15, stride=4), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 9, stride=2), nn.ReLU(), nn.MaxPool1d(3),
            nn.Conv1d(64, 128, 5, stride=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        h = self.net(x).squeeze(-1)
        return self.fc(h).squeeze(-1)


class LSTMNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.ds = nn.AvgPool1d(5)          # 500 -> 100 步
        self.lstm = nn.LSTM(1, hidden, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.ds(x).squeeze(1).unsqueeze(-1)   # (B,T/5,1), batch_first
        out, _ = self.lstm(h)
        return self.fc(out[:, -1, :]).squeeze(-1)


def train_model(model, Xtr, ytr, name):
    torch.manual_seed(SEED)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    # 类别不均衡加权
    pos = max(ytr.sum(), 1.0)
    w = torch.tensor([len(ytr) / (2.0 * pos)])
    lossf = nn.BCEWithLogitsLoss(pos_weight=w)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32).transpose(1, 2)   # (B,1,T)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    n = len(ytr)
    for ep in range(EPOCHS):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            out = model(Xtr_t[idx])
            loss = lossf(out, ytr_t[idx])
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        print(f"  [{name}] epoch {ep+1}/{EPOCHS}: loss={tot/n:.4f}")
    return model


def predict(model, X, kind):
    model.eval()
    with torch.no_grad():
        # X 可能是 float64 (归一化后) -> 统一转 float32 与模型权重一致
        Xt = torch.tensor(X, dtype=torch.float32).transpose(1, 2)
        return torch.sigmoid(model(Xt)).numpy()


# ---------------- 事件级评估 ----------------
def event_metrics(model, kind, eps):
    """每个 episode: 首个报警窗时刻 vs 事件时刻 -> 提前量; stable -> 误报"""
    out = []
    for ep in eps:
        X, y, tw = make_windows(ep)
        if len(X) == 0:
            continue
        prob = predict(model, X, kind)
        hit = prob > 0.5
        alarm_t = tw[np.argmax(hit)] if hit.any() else None
        rec = dict(name=ep["name"], label=ep["label"], t_event=ep["t_event"],
                   alarm_t=alarm_t,
                   lead=(ep["t_event"] - alarm_t) if (alarm_t is not None and
                         ep["t_event"] is not None) else None,
                   fp_rate=float(np.mean(hit)) if ep["t_event"] is None else None)
        out.append(rec)
    return out


def summarize(recs, tag):
    ev = [r for r in recs if r["t_event"] is not None]
    st = [r for r in recs if r["t_event"] is None]
    det = [r for r in ev if r["alarm_t"] is not None]
    leads = [r["lead"] for r in det]
    print(f"  [{tag}] 事件 {len(ev)}, 检出 {len(det)} "
          f"({100*len(det)/max(len(ev),1):.0f}%), "
          f"提前量 median={np.median(leads):.2f}s" if leads else
          f"  [{tag}] 事件 {len(ev)}, 检出 {len(det)}")
    if st:
        print(f"        stable 误报率: 均值 {np.mean([r['fp_rate'] for r in st])*100:.1f}%"
              f" (每 episode 正窗占比)")
    return det, st


def main():
    torch.set_num_threads(max(os.cpu_count() - 1, 1))
    print("=" * 76)
    print("CNN / LSTM 对比基线 (任务: 未来 1s 内失稳预警)")
    print("=" * 76)

    eps = load_episodes()
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(eps))
    n_te = max(len(eps) // 5, 4)
    te_idx = set(idx[:n_te].tolist())
    train_eps = [e for i, e in enumerate(eps) if i not in te_idx]
    test_eps = [e for i, e in enumerate(eps) if i in te_idx]
    print(f"数据: {len(eps)} episodes -> train {len(train_eps)} / test {len(test_eps)}")

    Xtr, ytr = [], []
    for ep in train_eps:
        X, y, _ = make_windows(ep)
        Xtr.append(X)
        ytr.append(y)
    Xtr = np.concatenate(Xtr)
    ytr = np.concatenate(ytr)
    print(f"训练窗: {len(Xtr)} (正类 {int(ytr.sum())}, {100*ytr.mean():.0f}%)")
    # 归一化 (窗级标准化)
    mu, sd = Xtr.mean(), Xtr.std() + 1e-9
    Xtr_n = ((Xtr - mu) / sd).astype(np.float32)

    results = {}
    for name, model in [("CNN", CNN1D()), ("LSTM", LSTMNet())]:
        print(f"\n--- 训练 {name} ---")
        train_model(model, Xtr_n, ytr, name)
        results[name] = model

        # held-out MG episodes
        recs = event_metrics(model, name.lower(),
                             [{**e, "p": (e["p"] - mu) / sd} for e in test_eps])
        summarize(recs, f"{name} MG held-out")

        # TUDa 校准版 (跨参数化)
        z = np.load(os.path.join(ROOT, "results", "tuda_calib_dataset.npz"),
                    allow_pickle=True)
        calib_eps = []
        for i in range(int(z["n_cases"])):
            p = z[f"case{i}_p"]
            t = z[f"case{i}_t"]
            # 校准版事件时刻: 用 J 阈值估计 (J>0.1)
            J = z[f"case{i}_J"]
            ii = np.argmax(J > 0.1)
            t_ev = float(t[ii]) if J.max() > 0.1 else None
            calib_eps.append(dict(p=p.astype(np.float32), fs=1.0/(t[1]-t[0]),
                                  t_event=t_ev, label="calib", name=f"calib{i}"))
        recs_c = event_metrics(model, name.lower(),
                               [{**e, "p": (e["p"] - mu) / sd} for e in calib_eps])
        summarize(recs_c, f"{name} TUDa-calib")

        # TUDa 实测 stall run (跨台架)
        import scipy.io as sio
        real_eps = []
        for fn in ["Stall_Speed100.16_Kulite_data.mat", "Stall_Speed65.03_Kulite_data.mat"]:
            fpath = os.path.join(ROOT, "tuda_data", "open_source", fn)
            if not os.path.exists(fpath):
                continue
            m = sio.loadmat(fpath)
            s = m["kulite_data"][0, 0]
            kd = np.asarray(s["kudatabase"], dtype=np.float32)[:, 0]
            sp = float(np.asarray(s["speed"]).ravel()[0])
            f_rot = sp / 100 * 20000 / 60
            # 实测事件时刻: 失速带 RMS 跳变 (与 GT 管线一致) —— 简化用
            # pressure_fluctuation_criterion 的结果由调用方另行对比
            from tuda_adapter import decimate as tdec
            p_ds, fs_ds = tdec(kd, 500000.0, FS_DS)
            real_eps.append(dict(p=p_ds.astype(np.float32), fs=fs_ds,
                                 t_event=None, label="TUDa-real",
                                 name=fn.replace("_Kulite_data.mat", "")))
        recs_r = event_metrics(model, name.lower(),
                               [{**e, "p": (e["p"] - mu) / sd} for e in real_eps])
        summarize(recs_r, f"{name} TUDa-real (无事件标签, 仅报误报率)")

    # 保存模型与归一化参数
    torch.save({"cnn": results["CNN"].state_dict(),
                "lstm": results["LSTM"].state_dict(),
                "mu": mu, "sd": sd},
               os.path.join(ROOT, "results", "dl_baseline.pt"))
    print("\n模型已保存: results/dl_baseline.pt")


if __name__ == "__main__":
    main()
