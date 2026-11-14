# -*- coding: utf-8 -*-
"""
喘振/失稳数据集批量生成器
- 基于 Moore-Greitzer 模型瞬态节流过程生成多工况轨迹
- 合成机匣壁面动态压力传感器信号 (含噪声/电磁干扰脉冲/燃油阶跃扰动)
- 按 ground truth (J, Psi 振荡) 生成逐样本标签:
    0=stable, 1=pre-stall(先兆), 2=rotating stall, 3=surge
输出: data/episode_*.npz + data/index.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import mg_model as mg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
os.makedirs(DATA, exist_ok=True)

# ---------------- 物理映射参数 ----------------
XI_END = 800.0          # 无量纲仿真时长 (节流斜坡 ~21 个喘振周期, 漂移与失稳频带分离)
DT = 5e-4               # 积分步长
J0 = 1e-4               # 初始扰动 (环境扰动水平)
A_PARAM = 0.5           # 叶排滞后参数 (增大失速增长率 rho)
XI_PER_SEC = 100.0      # 无量纲时间 -> 物理时间 映射
FS = 20000.0            # 采样率 Hz
F_ROTOR = 400.0         # 转子基频 Hz (示意值)
STALL_FRAC = 0.45       # 失速团传播速度 (转子转速占比)
K_P = 10000.0           # 压力系数 -> Pa
A_RS = 350.0            # 失速团扰动幅值 Pa (J=1 时)
NOISE_STD = 5.0         # 传感器噪声 std Pa

BASE = {**mg.DEFAULT_PARAMS, "a": A_PARAM}
DS = 10  # 信号抽取因子: 物理 fs = XI_PER_SEC/DT/DS = 200 kHz/10 = 20 kHz


# ---------------- 工况设计 ----------------
def build_scenarios(seed=2026):
    """构造多类工况: surge / rotating_stall / stable / stable+interference"""
    rng = np.random.default_rng(seed)
    scen = []

    # 1) 喘振工况 (含混合失稳): B 中大, 缓慢关小节气门越过边界
    for i in range(26):
        B = rng.choice([1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3])
        gT_end = rng.uniform(0.48, 0.53)
        ramp_end = rng.uniform(200, 550)
        scen.append(dict(family="surge", B=B, gammaT_start=0.68,
                         gammaT_end=gT_end, ramp_end=ramp_end))

    # 2) 旋转失速工况: B 小
    for i in range(18):
        B = rng.uniform(0.35, 0.7)
        gT_end = rng.uniform(0.49, 0.53)
        ramp_end = rng.uniform(200, 550)
        scen.append(dict(family="rotating_stall", B=B, gammaT_start=0.68,
                         gammaT_end=gT_end, ramp_end=ramp_end))

    # 3) 稳定工况 (部分含燃油阶跃/干扰): 节气门不过界, 同样缓慢下降
    for i in range(22):
        B = rng.uniform(0.6, 2.0)
        gT_end = rng.uniform(0.63, 0.67)
        interference = i % 2 == 1          # 一半含燃油阶跃
        scen.append(dict(family="stable", B=B, gammaT_start=0.68,
                         gammaT_end=gT_end, ramp_end=400.0,
                         fuel_step=interference))
    return scen, rng


def make_sched_vec(scen):
    """按工况向量化节气门时间表 (线性关小 + 可选燃油阶跃)"""
    n = len(scen)

    def sched(xi, g0):
        out = np.empty(n)
        for i, s in enumerate(scen):
            gs, ge, re_ = s["gammaT_start"], s["gammaT_end"], s["ramp_end"]
            if xi >= re_:
                g = ge
            elif xi >= 10.0:
                f = (xi - 10.0) / max(re_ - 10.0, 1e-9)
                g = gs + f * (ge - gs)
            else:
                g = gs
            st = s.get("step_time")
            if st is not None and xi >= st:
                g += s.get("step_delta", 0.0)
            out[i] = g
        return out
    return sched


def add_fuel_steps(scen, rng):
    """给 stable 工况添加燃油阶跃 (模拟加速/减速机动扰动, 不引发失稳)"""
    for s in scen:
        if s.get("fuel_step"):
            s["step_time"] = rng.uniform(50, 700)
            s["step_delta"] = rng.choice([-1, 1]) * rng.uniform(0.02, 0.045)
    return scen


# ---------------- 主流程 ----------------
def main():
    scen, rng0 = build_scenarios()
    rng0 = add_fuel_steps(scen, rng0)
    n = len(scen)
    print(f"共 {n} 个工况, 开始向量化积分 (dt={DT}, xi_end={XI_END}) ...")

    params_list = [{**BASE, "B": s["B"]} for s in scen]
    sched = make_sched_vec(scen)
    xi, Y, meta = mg.integrate(params_list, sched, xi_end=XI_END, dt=DT, J0=J0)
    events = mg.event_ground_truth(xi, Y, meta, params_list)
    print("积分完成, 合成传感器信号 ...")

    T = len(xi)
    fs_ratio = FS / XI_PER_SEC          # 每个无量纲时间步的采样点数
    t_phys = xi / XI_PER_SEC
    idx_records = []
    rng_sig = np.random.default_rng(7)

    for c, s in enumerate(scen):
        ev = events[c]
        sig = mg.synthesize_signal(
            xi, Y[:, :, c:c + 1], meta, rng_sig,
            xi_per_sec=XI_PER_SEC, f_rotor=F_ROTOR, stall_frac=STALL_FRAC,
            K_p=K_P, A_rs=A_RS, noise_std=NOISE_STD)

        # 抽取到 20 kHz (真实物理采样率 = 1/(DT/XI_PER_SEC) = 200 kHz)
        sl = slice(None, None, DS)
        t_phys_ds = sig["t"][sl]
        p_ds = sig["p"][sl, 0]
        Psi_ds, Phi_ds, J_ds = sig["Psi"][sl, 0], sig["Phi"][sl, 0], sig["J"][sl, 0]
        fs_true = FS  # 抽取后 fs = 200kHz/10 = 20 kHz, 校验:
        assert abs((1.0 / (DT / XI_PER_SEC)) / DS - fs_true) < 1.0

        # 逐样本标签 (按物理时间, 与抽取后网格对齐)
        t_pre = ev["t_pre"] / XI_PER_SEC if not np.isnan(ev["t_pre"]) else np.inf
        t_stall = ev["t_stall"] / XI_PER_SEC if not np.isnan(ev["t_stall"]) else np.inf
        t_surge = ev["t_surge"] / XI_PER_SEC if not np.isnan(ev["t_surge"]) else np.inf
        T_ds = len(t_phys_ds)
        labels = np.zeros(T_ds, dtype=np.int8)
        labels[t_phys_ds >= min(t_pre, t_stall, t_surge)] = 1          # 先兆
        labels[t_phys_ds >= min(t_stall, t_surge)] = 2                 # 失速发展
        labels[t_phys_ds >= t_surge] = 3                               # 喘振

        fname = f"episode_{c:03d}_{s['family']}.npz"
        np.savez_compressed(
            os.path.join(DATA, fname),
            t=t_phys_ds, p=p_ds, Psi=Psi_ds, Phi=Phi_ds, J=J_ds,
            labels=labels, event=dict(ev), scenario=dict(
                (k, v) for k, v in s.items() if isinstance(v, (int, float, str, bool))),
            fs=fs_true, f_rotor=F_ROTOR, stall_frac=STALL_FRAC)
        idx_records.append(dict(
            file=fname, family=s["family"], case=c, B=s["B"],
            gammaT_end=s["gammaT_end"], fuel_step=bool(s.get("fuel_step", False)),
            label=ev["label"],
            t_pre=t_pre if np.isfinite(t_pre) else np.nan,
            t_stall=t_stall if np.isfinite(t_stall) else np.nan,
            t_surge=t_surge if np.isfinite(t_surge) else np.nan,
        ))
        print(f"  [{c+1:>2}/{n}] {s['family']:>15} B={s['B']:.2f} label={ev['label']}")

    df = pd.DataFrame(idx_records)
    df.to_csv(os.path.join(DATA, "index.csv"), index=False)
    print("\n=== 汇总 ===")
    print(df.groupby("family")["label"].value_counts().unstack(fill_value=0))
    print(f"\n数据集保存于 {DATA} (index.csv + {n} 个 npz)")


if __name__ == "__main__":
    main()
