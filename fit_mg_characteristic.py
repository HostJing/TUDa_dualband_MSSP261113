# -*- coding: utf-8 -*-
"""
TUDa 实测稳态特性线 -> MG 模型参数拟合

数据来源: tuda_data/open_source/characteristics.csv (data_submission_template.zip)
  - 单条 100% 校正转速线 (20044.68 rpm), 14 个工作点
  - 列: Corrected_speed(RPM), Corrected_mass_flow(kg/s), pt_MA_15(Pa), Tt_MA_15(K),
        pt_MA_30(Pa), Tt_MA_30(K), ptr_MA_15_30, Ttr_MA_15_30, Isentropic_Efficiency_15_30
  - 另有 3 列未命名附加数据 (疑似另一测量截面组合), 本脚本不使用

流程:
  1. load_measured_lines(): 读取 100% 校正转速的总压比-流量数据
  2. convert_to_mg_plane(): 换算到 MG (Phi, Psi) 平面
  3. fit_cubic(): 两层最小二乘拟合 psi_c(Phi) = psi_c0 + H*(1 + 1.5x - 0.5x^3)
  4. sanity: 拟合线失速裕度 vs 实测失速裕度闭环对比

换算约定 (见 docs/TUDa_边界条件核对.md):
  Phi = c_x / U_tip,  Psi = dp0 / (rho * U_tip^2)
  进口密度 rho 用实测 pt_MA_15 / Tt_MA_15 计算 (非 ISA 常数)
"""
import csv
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "tuda_data", "open_source", "characteristics.csv")

# 物理常数与 TUDa 公开设计参数 (见 docs/TUDa_边界条件核对.md)
GAMMA = 1.4
R_AIR = 287.05
CP = 1004.5
P0_ISA = 101325.0
U_TIP = 398.0        # m/s, 校正叶尖速度 (N=20000 rpm)
RT = 0.190           # m
RH_RT = 0.51
A_ANN = np.pi * RT ** 2 * (1 - RH_RT ** 2)   # 环形通道面积 [m^2]


def load_measured_lines(path=CSV_PATH, speed_key=20044.68):
    """读取指定校正转速的实测特性线.

    返回 dict: mdot_corr[kg/s], pi_total[-], pt_in[Pa], Tt_in[K]
    """
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        for r in reader:
            if not r or not r[0].strip():
                continue
            rpm = float(r[0])
            if abs(rpm - speed_key) > 1.0:
                continue
            rows.append(dict(
                mdot=float(r[1]),
                pt_in=float(r[2]), Tt_in=float(r[3]),
                pt_out=float(r[4]), Tt_out=float(r[5]),
                pi=float(r[6]),
                eff=float(r[8]),
            ))
    if not rows:
        raise RuntimeError(f"未找到校正转速 {speed_key} rpm 的数据行")
    rows.sort(key=lambda d: d["mdot"])   # 流量升序 (NS -> 堵)
    return dict(
        mdot_corr=np.array([d["mdot"] for d in rows]),
        pi_total=np.array([d["pi"] for d in rows]),
        eff=np.array([d["eff"] for d in rows]),
        pt_in=rows[0]["pt_in"], Tt_in=rows[0]["Tt_in"],
    )


def inlet_density(pt_in, Tt_in):
    """实测进口总参数 -> 密度 (近似: 总参数直接代入理想气体, Ma 低时误差小)"""
    return pt_in / (R_AIR * Tt_in)


def convert_to_mg_plane(mdot_corr, pi_total, rho_in):
    """工程量 -> MG 无量纲平面: Phi = c_x/U_tip, Psi = dp0/(rho*U_tip^2)"""
    mdot = np.asarray(mdot_corr, float)
    pi = np.asarray(pi_total, float)
    c_x = mdot / (rho_in * A_ANN)
    dp0 = P0_ISA * (pi - 1.0)          # 进口总压取 ISA 标准值 (相对总压比定义)
    Phi = c_x / U_TIP
    Psi = dp0 / (rho_in * U_TIP ** 2)
    return Phi, Psi


def psi_cubic(Phi, psi_c0, H, W):
    x = Phi / W - 1.0
    return psi_c0 + H * (1.0 + 1.5 * x - 0.5 * x ** 3)


def fit_cubic(Phi, Psi, W_grid=None, branch="descending"):
    """两层拟合: 外层扫 W, 内层对 (psi_c0, H) 线性最小二乘.

    branch="descending" (默认): 实测特性线在峰值右侧的下降段 (负斜率,
    流量减小->压升增大), 要求 MG 峰值 2W <= min(Phi), 即 W <= Phi_min/2.
    这样 14 点全部落在 x > 1 的工作区。

    ⚠ 可辨识性警告: psi_c0/H/W 强相关 —— 单支转速线 + 窄 Φ 区间时,
    mse 对 W 的平坦度需显式报告 (诊断输出), 参数只能作"等效 MG 参数"用。
    """
    Phi = np.asarray(Phi, float)
    Psi = np.asarray(Psi, float)
    if W_grid is None:
        W_max = Phi.min() / 2.0          # 峰值必须在数据左侧 (下降段)
        W_grid = np.linspace(0.10, W_max, 80)
    results = []
    for W in W_grid:
        x = Phi / W - 1.0
        A = np.column_stack([np.ones_like(x), 1.0 + 1.5 * x - 0.5 * x ** 3])
        coef, *_ = np.linalg.lstsq(A, Psi, rcond=None)
        mse = float(np.mean((A @ coef - Psi) ** 2))
        results.append(dict(W=float(W), psi_c0=float(coef[0]), H=float(coef[1]),
                            mse=mse))
    best = min(results, key=lambda d: d["mse"])
    return best, results


def fit_cubic_constrained(Phi, Psi, W):
    """固定 W 后对 (psi_c0, H) 做非负最小二乘 (psi_c0 >= 0, H >= 0).

    单支近峰区数据下 psi_c0 与 H 强相关, 无约束解可能给出非物理的
    psi_c0 < 0; 此处用手写两步回退 (等价 scipy lsq_linear 两变量情形):
      1) 无约束 lstsq; 若 psi_c0 >= 0 且 H >= 0 则返回;
      2) 否则固定 psi_c0 = 0, 对 H 一维解析: H = max(0, <g,Psi>/<g,g>).
    """
    x = np.asarray(Phi, float) / W - 1.0
    g = 1.0 + 1.5 * x - 0.5 * x ** 3
    A = np.column_stack([np.ones_like(x), g])
    coef, *_ = np.linalg.lstsq(A, Psi, rcond=None)
    if coef[0] >= 0 and coef[1] >= 0:
        psi_c0, H = float(coef[0]), float(coef[1])
    else:
        psi_c0 = 0.0
        H = float(max(0.0, g @ Psi / (g @ g)))
    Psi_fit = psi_c0 + H * g
    mse = float(np.mean((Psi_fit - Psi) ** 2))
    return dict(W=float(W), psi_c0=psi_c0, H=H, mse=mse)


def stall_margins(mdot, pi, Phi_fit_peak):
    """失速裕度对比: 实测 (相对压比峰值点) vs 拟合特性线.

    SM_flow = (mdot_ref - mdot_NS) / mdot_ref
    实测 ref = pi 峰值点; 拟合 ref = Phi_peak = 2W 处 (对应实测峰值点).
    """
    i_ns = 0                      # mdot 升序 -> 第一点为 NS (最小流量)
    i_peak = int(np.argmax(pi))
    sm_meas = (mdot[i_peak] - mdot[i_ns]) / mdot[i_peak]
    Phi_ns = None                 # 由调用方填充
    return sm_meas, i_ns, i_peak


if __name__ == "__main__":
    print("=" * 72)
    print("TUDa-GLR-OpenStage 实测特性线 -> MG 参数拟合")
    print("=" * 72)

    data = load_measured_lines()
    mdot, pi, eff = data["mdot_corr"], data["pi_total"], data["eff"]
    rho_in = inlet_density(data["pt_in"], data["Tt_in"])
    print(f"\n[1] 实测数据: {len(mdot)} 点, N_corr = 20044.68 rpm (100%)")
    print(f"    进口: pt={data['pt_in']:.0f} Pa, Tt={data['Tt_in']:.2f} K "
          f"-> rho_in = {rho_in:.4f} kg/m^3 (ISA: 1.225)")
    print(f"    ṁ_corr: {mdot.min():.3f} .. {mdot.max():.3f} kg/s")
    print(f"    π_tt  : {pi.min():.4f} .. {pi.max():.4f}")
    print(f"    η_is  : {eff.min():.4f} .. {eff.max():.4f}")

    i_peak = int(np.argmax(pi))
    i_ns = 0
    print(f"\n    π 峰值点: ṁ={mdot[i_peak]:.3f} kg/s, π={pi[i_peak]:.4f}")
    print(f"    NS 点   : ṁ={mdot[i_ns]:.3f} kg/s, π={pi[i_ns]:.4f}")
    sm_meas = (mdot[i_peak] - mdot[i_ns]) / mdot[i_peak]
    print(f"    实测流量失速裕度 (相对 π 峰值): {sm_meas*100:.2f}%")

    Phi, Psi = convert_to_mg_plane(mdot, pi, rho_in)
    print(f"\n[2] MG 平面: Phi = {Phi.min():.4f} .. {Phi.max():.4f}, "
          f"Psi = {Psi.min():.4f} .. {Psi.max():.4f}")

    best, results = fit_cubic(Phi, Psi)
    print(f"\n[3] 立方特性线拟合 (下降段约束: 2W <= Phi_min = {Phi.min():.4f}):")
    print(f"    psi_c0 = {best['psi_c0']:.4f}, H = {best['H']:.4f}, "
          f"W = {best['W']:.4f}, MSE = {best['mse']:.3e}")
    print(f"    拟合峰值点 Phi_peak = 2W = {2*best['W']:.4f}, "
          f"Psi_peak = psi_c0 + 2H = {best['psi_c0'] + 2*best['H']:.4f}")
    print(f"    实测峰值点 Phi = {Phi[i_peak]:.4f}, Psi = {Psi[i_peak]:.4f}")
    # 拟合曲线在工作区端点的复现质量
    for tag, i in [("NS", i_ns), ("峰值", i_peak), ("堵侧", len(Phi) - 1)]:
        fit_v = psi_cubic(Phi[i], best["psi_c0"], best["H"], best["W"])
        print(f"    {tag}: Phi={Phi[i]:.4f} 实测 Ψ={Psi[i]:.4f} "
              f"拟合 Ψ={fit_v:.4f} (Δ={fit_v - Psi[i]:+.4f})")

    # W 敏感性 (可辨识性诊断): mse 在最优 W 附近的平坦程度
    print(f"\n[4] W 敏感性 (可辨识性诊断):")
    ws = np.array([r["W"] for r in results])
    w_lo, w_hi = ws.min(), ws.max()
    for w_probe in [w_lo + 0.02, w_lo + 0.04,
                    (w_lo + w_hi) / 2, w_hi - 0.02, w_hi]:
        j = int(np.argmin(np.abs(ws - w_probe)))
        r = results[j]
        print(f"    W={r['W']:.3f}: psi_c0={r['psi_c0']:.4f}, H={r['H']:.4f}, "
              f"MSE={r['mse']:.3e} (x{r['mse']/best['mse']:.2f})")

    # [5b] 物理约束版 (psi_c0 >= 0): W 仍取实测峰值定位
    best_c = fit_cubic_constrained(Phi, Psi, best["W"])
    print(f"\n[5b] 物理约束拟合 (psi_c0 >= 0, 同一 W = {best_c['W']:.4f}):")
    print(f"    psi_c0 = {best_c['psi_c0']:.4f}, H = {best_c['H']:.4f}, "
          f"MSE = {best_c['mse']:.3e} (自由版 x{best_c['mse']/best['mse']:.2f})")
    for tag, i in [("NS", i_ns), ("峰值", i_peak), ("堵侧", len(Phi) - 1)]:
        fit_v = psi_cubic(Phi[i], best_c["psi_c0"], best_c["H"], best_c["W"])
        print(f"    {tag}: 实测 Ψ={Psi[i]:.4f} 拟合 Ψ={fit_v:.4f} "
              f"(Δ={fit_v - Psi[i]:+.4f})")

    # [5c] 拟合线失速裕度闭环: NS 点相对拟合峰值的流量裕度
    Phi_ns = Phi[i_ns]
    Phi_peak_fit = 2 * best["W"]
    sm_fit = (Phi_ns - Phi_peak_fit) / Phi_ns
    print(f"\n[5] 失速裕度闭环: 实测 {sm_meas*100:.2f}% vs 拟合线 {sm_fit*100:.2f}% "
          f"(Δ = {(sm_fit-sm_meas)*100:+.2f} pp)")
    print(f"    (约束拟合下 2W <= Phi_NS 由构造保证, 差值反映峰值点定位)")

    # [6] 特性线对比图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_png = os.path.join(ROOT, "results", "tuda_mg_characteristic_fit.png")
    Phi_dense = np.linspace(0.10, 0.55, 300)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, params, label in [
        (axes[0], best, "free fit (psi_c0 can go negative)"),
        (axes[1], best_c, "physically constrained (psi_c0 >= 0)"),
    ]:
        ax.plot(Phi_dense, psi_cubic(Phi_dense, params["psi_c0"], params["H"],
                                     params["W"]),
                "-", color="#3b74c4", lw=1.8,
                label=f"MG cubic (W={params['W']:.3f})")
        ax.plot(Phi, Psi, "o", color="#c43b3b", ms=6, label="TUDa measured (100% Nc)")
        ax.axvline(Phi[i_ns], color="gray", ls=":", lw=1, label="NS point")
        ax.axvline(2 * params["W"], color="#3b74c4", ls="--", lw=1, label="MG peak (2W)")
        ax.set_xlabel("Phi = c_x / U_tip")
        ax.set_ylabel("Psi = dp0 / (rho U_tip^2)")
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"\n[7] 对比图已保存: {out_png}")

    # [8] 结论提示
    print(f"""
[6] 使用说明:
    - 本拟合为"等效 MG 参数"(单条转速线 + psi_c0/H/W 强相关), 用于:
      a) 将 TUDa 工况映射进 MG 仿真: Phi0 取设计/待研究点, B 用 U_tip/a 前估计;
      b) 检测器归一化常数复核 (STALL_NORM 等以 Phi/Psi 标定).
    - 若后续拿到非定常壁压数据测得真实失速点流量, 用 NS 实测 Φ 重校 W.
    - 压升尺度注意: 本脚本 Psi 用 U_tip=398 m/s; 文档中的 psi=0.40 用
      U_mean≈300 m/s (平均半径), 两者差 (398/300)^2 ≈ 1.76 倍, 属定义差异.
""")
