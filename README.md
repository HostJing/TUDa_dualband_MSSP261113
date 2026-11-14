# surge-detection

**Rotor-frequency-normalized dual-band detection of rotating-stall inception:
zero-retuning cross-speed validation on an open transonic compressor dataset**

Code and reproducibility pipeline for the manuscript submitted to
*Mechanical Systems and Signal Processing* (MSSP). The detector scales all
band limits, window lengths and baseline intervals with the rotor frequency,
so a detector commissioned on one machine transfers to another speed line
without re-tuning; thresholds are calibrated from 26 steady operating points
of the open TUDa-GLR-OpenStage transonic compressor dataset via a
peaks-over-threshold fit.

**Paper outputs at a glance**

| Validation | Result | Script |
|---|---|---|
| Zero-retuning cross-speed alarms (K=1.6) | lead 0.29 s at both speeds, 3/26 steady FAs | `src/test_tuda_stall.py` |
| POT/GPD calibrated threshold | K=2.195: 0/26 false alarms, alarms within 0.13 s of GT | `src/auto_threshold.py` |
| Precursor measurement | 0.72 s / 0.92 s before reference onset | `src/precursor_tf.py` |
| Online causal operation | 23 ms / 0 ms vs offline, <3% of one CPU core | `src/online_eval.py` |
| CNN/LSTM baselines | AP ≈ 0.78 in distribution, no transfer across speeds | `src/dl_baseline.py`, `src/dl_cross_domain.py` |
| Closed-loop avoidance | lead ≥ 0.55 s required; measured 0.72–0.92 s | `src/closed_loop.py`, `src/modal_closed_loop.py` |
| Paper figures | regenerate all 10 figures + graphical abstract | `src/make_paper_figures.py`, `src/make_graphical_abstract.py` |

**Quick start**

```bash
pip install -r requirements.txt
python src/generate_dataset.py      # build the synthetic MG dataset -> data/ (~2-3 min)
python src/run_all.py               # reproduce all paper results (~1 h CPU)
python src/make_paper_figures.py    # regenerate the paper figures -> paper/figures/
```

**External data.** The two TUDa stall runs and 26 steady operating points
(~36 GB) are **not** included in this repository. They are available through the official GPPS portal
(https://gpps.global/tu-darmstadt/; Klausmann et al., 2022,
DOI: 10.33737/jgpps/156120) -- see
`docs/TUDa_OpenStage_获取指南.md` for the download/申请 path and expected
directory layout (`tuda_data/open_source/`, `tuda_data/steady/`).

---

# 航空发动机压气机喘振/失速预检测 —— 仿真数据集与验证管线

## 背景

公开的航空发动机整机喘振数据集几乎不存在，本项目提供两条腿：

1. **Moore-Greitzer 非线性模型自仿真数据集**（本仓库生成）：覆盖稳定 / 预失速先兆 / 旋转失速 / 喘振 / 燃油阶跃干扰五类场景，用于算法开发与阈值整定；
2. **TUDa-GLR-OpenStage 真实失速起始壁压数据**（外部申请）：用于最终验证，申请路径见 `docs/TUDa_OpenStage_获取指南.md`。

## 目录结构

```
surge_project/
├── src/
│   ├── mg_model.py          # Moore-Greitzer 三状态模型 + RK4 + 信号综合 + 事件真值
│   ├── generate_dataset.py  # 批量生成 66 工况数据集 -> data/
│   ├── calibrate.py         # 四类典型工况动力学校准图
│   ├── detector.py          # 双频带自适应阈值预失速检测器（仅用 p(t)，转频归一化）
│   ├── array_processing.py  # 周向阵列：多通道合成/传播速度(非均匀布置)/相干性闸门
│   ├── array_demo.py        # 阵列演示图生成脚本
│   ├── validate_array.py    # 阵列闸门批量验证 -> array_gate_report.md
│   ├── tuda_adapter.py      # TUDa 真实数据接入 adapter（读取/切片/管线对接）
│   ├── test_adapter_pipeline.py  # adapter 管线端到端测试
│   ├── test_tuda_stall.py   # TUDa stall run 双转速检测验证 (核心实测验证)
│   ├── false_alarm_eval.py  # 26 稳态点虚警率评估 (P0-1)
│   ├── precursor_tf.py      # 实测前兆时频搜索: 分带 z-score + 跨通道相干 (P0-2)
│   ├── online_eval.py       # 因果在线化评测: 滚动缓冲 + 延迟预算 (P0-3)
│   ├── dl_baseline.py       # CNN/LSTM 基线 (MG 同分布 + 三级跨域测试)
│   ├── dl_cross_domain.py   # few-shot 跨域微调 + PR-AUC + 经典基线 (P0-4)
│   ├── auto_threshold.py    # POT/GPD 自动阈值 (26 稳态点标定, P1-5)
│   ├── generate_dataset_tuda.py # TUDa 校准版 MG 数据集 (实测特性线+文献 B)
│   ├── fit_mg_characteristic.py # 实测特性线 -> MG 参数定量拟合
│   ├── closed_loop.py       # 检测->避喘闭环仿真 (负结果归因)
│   ├── run_all.py           # 一键复现全部论文结果
│   ├── detector_online.py   # 在线流式版（因果 IIR，逐块推入）
│   ├── ablation_sensors.py  # 传感器数量/位置消融 -> sensor_ablation.md
│   ├── type_discrimination.py # 失稳类型早期区分 -> type_discrimination.md
│   ├── robustness.py        # SNR/增益失配/进气畸变鲁棒性 -> robustness.md
│   ├── baselines.py         # CNN/LSTM 对比基线（torch CPU）
│   └── validate.py          # 检出率/预警提前量/误报率评估 + 报告生成
├── data/                    # episode_*.npz + index.csv（生成产物）
├── results/                 # 校准图/示例图/validation_report.md
└── docs/TUDa_OpenStage_获取指南.md
```

## 模型（Moore & Greitzer 1986, 三状态 Galerkin）

```
ψc(Φ) = ψc0 + H[1 + 1.5x − 0.5x³],  x = Φ/W − 1
dΦ/dξ = (1/lc)[ψc(Φ) − Ψ − (3/4)H·x·J]
dΨ/dξ = (Φ − γT√Ψ)/(4B²lc)
dJ/dξ = J·ρ·(1 − x² − J/4),  ρ = 3aH/((1+ma)W)
```

参数：`ψc0=0.30, H=0.18, W=0.25, lc=4.0, a=0.5, m=1.75`。

- **B 参数决定失稳类型**：小 B → 旋转失速；大 B → 深喘振（倒流极限环）；中等 B → 混合失稳（模态波先兆 → 喘振，即"RS 是喘振前兆"）
- 传感器信号综合：`p(t) = Kp·Ψ(t) + Ars·√J·cos(2π·0.45·N·t) + 噪声 + 电磁干扰脉冲`
- 燃油阶跃干扰：稳定工况上叠加节气门开度阶跃（模拟加速/减速机动），考察误报

## ground truth 标签

| 标签 | 判据 | 含义 |
|---|---|---|
| 0 stable | — | 稳定 |
| 1 pre-stall | J > 1e-3 | 失速先兆（模态波增长） |
| 2 stall | J > 0.1 | 旋转失速发展 |
| 3 surge | Ψ 滑动 std > 0.25H | 喘振（压力大幅振荡/倒流） |

## 使用

```bash
python src/calibrate.py        # 四类工况动力学校准
python src/generate_dataset.py # 生成数据集（约 2-3 分钟）
python src/validate.py         # 运行检测器评估，输出 results/validation_report.md
```

依赖：numpy, scipy, matplotlib, torch, scikit-learn（见 requirements.txt）。

## TUDa 实测验证与论文冲刺（2026-11-13 更新）

数据全部到手后的核心验证结果（P0/P1 计划见 `docs/Q1论文补充研究计划.md`）：

| 验证 | 结果 |
|---|---|
| 双转速 stall run 零重整定 | 融合报警一致提前包络 GT 0.29s（N100 2.387s / N65 4.780s） |
| 实测前兆时频搜索（P0-2） | 前兆提前 **0.72s / 0.92s**（分带 z-score），N65 失速带 zpeak 55.7；跨通道相干 → 相干扰动结构 |
| 稳态虚警（P0-1, 26 点） | 单通道检测器 **0/26**；融合 3/26（均远离 NS 点）→ POT/GPD 自动阈值（P1-5）后 **0/26** 且双转速报警保持早于 GT |
| 在线化（P0-3） | 双转速在线 vs 离线偏差 23ms / 0ms（PASS），单核计算占比 <1% |
| DL 对照（P0-4） | 同分布 AP=0.775/0.757；跨转速 few-shot（≤50 标注窗，已用尽单 run 全部正窗）无法恢复——恒报警 FP 100% |
| 经典基线（稳态标定绝对阈值） | lead +0.03s / +0.38s，零学习即有效 |

关键脚本：`test_tuda_stall.py` / `false_alarm_eval.py` / `precursor_tf.py` /
`online_eval.py` / `dl_cross_domain.py` / `auto_threshold.py`；一键复现
`python src/run_all.py`。物理标度论证（失速带 0.25–0.625×f_rotor 依据失速团
传播速度文献范围、喘振带 0.001–0.03×f_rotor 依据 Helmholtz 频率与 B 的关系
f_H = U/(4πB·L_c)）见 `docs/检测频带物理标度论证.md`。

## 数据文件格式（episode_*.npz）

| 字段 | 说明 |
|---|---|
| t, p | 物理时间 [s]，机匣壁面动态压力 [Pa]（20 kHz） |
| Psi, Phi, J | 模型状态轨迹（ground truth，检测器不得使用） |
| labels | 逐样本标签 0/1/2/3 |
| event | 事件时刻（无量纲）：t_pre / t_stall / t_surge |
| fs, f_rotor, stall_frac | 采样率与信号合成参数 |

## 三种方法对比（同一数据集、同一事件级评测协议）

| 方法 | 预警成功率 | 提前量中位数 | 误报率 | 因果/可在线 |
|---|---|---|---|---|
| 离线双频带检测器 (`detector.py`) | **88.9%** (40/45) | **2794 ms** | 4.8% (1/21) | 否（全段 FFT） |
| 在线流式检测器 (`detector_online.py`) | **73.3%** (33/45) | 145–184 ms | **0%** (0/21) | 是（因果 IIR，逐块推入） |
| 1D-CNN 窗口分类 (`baselines.py`) | 36% (5/14*) | 85 ms | 0% (0/6*) | 可在线（0.25 s 窗） |
| LSTM 窗口分类 (`baselines.py`) | 0% (0/14*) | — | 0% (0/6*) | 可在线 |

\* DL 基线按工况划分的训练/测试集（测试集仅 14 失稳 + 6 稳定 episode）。

**关键结论**：在有限数据条件下，物理启发的频带能量检测器显著优于朴素深度学习基线——DL 在先兆类（训练窗口仅 59 个）上 precision 只有 0.33/0，事件级预警大量迟报；这与文献中"喘振预警需要物理特征先验，纯端到端 DL 需要大数据量"的观点一致。DL 基线的价值在于提供对比下限与未来（TUDa 真实数据到手后）扩展的起点。

## 在线流式检测器 (`detector_online.py`)

- **因果 IIR 滤波链**：手写 RBJ 双二阶系数（无 scipy 依赖）——1 Hz 高通去趋势 + 失速带 100–250 Hz 两级带通 + 喘振带 1–12 Hz 带通；逐 10 ms 块推入，单 episode 处理 ~0.1 s
- **工程要点**（调试中踩过的坑，均有数值验证）：
  - 直流阶跃振铃：用首样本预置 HP 状态可消除，但级联各级输入不同、需整体热身（`run_episode` 内置 0.3 s 恒定热身段）
  - 漂移与失稳频带重叠：要求节气门下降过程 ≫ 喘振周期（数据集已改为 21 个喘振周期的慢节流）；在线用慢速自适应基线（τ=2 s，下限锁定）替代离线 FFT 的频点清零
  - 喘振带判据用多数表决（3/5 块）而非连续判据，避免低频振荡波谷清零计数
- 预警提前量 145–184 ms（离线版 2.8 s 的差距来自：因果滤波群延迟 + 自适应基线吸收缓慢增长段）

## 深度学习基线 (`baselines.py`, torch CPU)

- 窗口分类任务：0.25 s @ 1 kHz 窗口 → {stable, pre-stall, stall/surge}，按 episode 划分防泄漏
- 1D-CNN：Conv1d(16, k16, s4) → Conv1d(32, k8, s2) → GAP → FC；窗口准确率 96.8%
- LSTM：skip 平均下采样 → LSTM(32) → 末隐状态 → FC；窗口准确率 96.2%
- 类权重平衡 + Adam，12 epochs，CPU 训练 ~1 min
- 局限（诚实标注）：先兆类训练窗口仅 59 个（J 指数增长使先兆相位极短）；测试 episode 少；未做特征工程输入，仅原始波形

## 转频归一化（换台架零整定）

两个检测器均提供 `from_rotor(fs, f_rotor)` 工厂：所有频带以转频归一化（失速带 0.25–0.625×f_rotor，喘振带 0.001–0.03×f_rotor），窗口/时间常数以转子转数归一化（如失速带 RMS 窗 20 转）。设计常数依据本仿真数据集整定（f_rotor=400 Hz 时与历史绝对参数完全一致，回归指标不变）；换到任何转速/采样率的台架只需改 `fs`/`f_rotor`，已在 25.6 kHz/600 Hz 模拟流上验证等比缩放。

## 周向阵列处理 (`array_processing.py`)

真实 WPT 测量（含 TUDa）沿机匣周向布置多传感器。旋转失速团在相邻通道间有确定相位差（行波），噪声/EMI/漂移不相干——两个物理事实支撑两个新能力：

| 能力 | 方法 | 仿真验证结果 |
|---|---|---|
| 失速团传播速度估计 | 失速带内相邻通道时移互相关 → β = Δθ/(360°·f_rotor·Δt)，**支持任意非均匀布置** | β 中位 0.446（真值 0.45）；扩散布置 4–8 通道误差均 <0.01 |
| 空间相干性闸门 | Welch 相干谱（子段去趋势+Hann 抗共模漂移泄漏）+ 持续超阈游程分段确认 | **旋转失速型失稳 30/30 确认**；失速工况相干性 0.93 vs 干扰工况 0.13 |

- **能力边界（诚实标注）**：喘振/真实压力瞬变是零阶平面波（各通道零相差），空间相干性原理上无法区分"真喘振"与"真实但非喘振的压力瞬变"（如燃油阶跃）；B≥1.9 纯深喘振 dive 前 J≈0、无相干结构，闸门不制造虚假检测
- 闸门还**暴露了原始单通道指标的虚高**：surge 家族约 7 个 "early" 实为 0.055 s 噪声误触发碰巧早于真实起始，已被闸门诚实拒绝
- 工程细节：kPa 级共模漂移的 FFT 旁瓣在带内频点完全相干，会人为抬高相干性——子段线性去趋势 + Hann 窗是必须的（已内置）
- 批量验证：`python src/validate_array.py` → `results/array_gate_report.md`；演示图：`python src/array_demo.py`

## TUDa 数据接入 (`tuda_adapter.py`)

数据到手后的流程已打通（`python src/test_adapter_pipeline.py` 端到端测试 PASS，模拟 25.6 kHz / f_rotor=600 Hz 数据流）：`read_raw_*()`（格式按 datasheet 实现桩函数）→ `resample_uniform()` → `WPTLayout`（非均匀传感器布置元数据）→ `build_episodes()` 切片 → `sanity_check()` 五项数据体检 → `run_detection()` 自动按真实 f_rotor 缩放频带 → `pressure_fluctuation_criterion()` 失速时刻后备估计。

## 方法研究（论文素材）

三项研究均有可复现脚本、报告与图，位于 `results/`：

**1) 传感器数量/位置消融**（`ablation_sensors.py` → `sensor_ablation.md/png`）
- 相干性闸门在 K=2 即完全有效，确认率/误报率与通道数基本无关（利用行波空间相位结构，2 个相邻传感器已可判别）
- β 速度估计：亚样本抛物线插值后等间隔 K=3 最优（误差 0.0009）；聚束布置误差增大 ~3 倍（小间隔角→小滞后，对噪声更敏感）
- 工程结论：周向 2–4 个等间隔传感器即可获得全部收益

**2) 失稳类型早期区分**（`type_discrimination.py` → `type_discrimination.md/png`）
- 关键设计：报警在失稳起始前 ~2.8 s 触发，瞬时特征无区分力 → 用报警以来**累积最大值**（失速团一旦出现相干性即跃升）
- 规则决策（无训练参数）：f3（失速带相干性）>0.4 → stall；否则 f2（喘振带能量）>4 → surge
- 结果：旋转失速型 **30/30 正确**，已判定样本准确率 97%（覆盖率 80%，未判定均为纯深喘振——物理上 dive 前无特征）
- 处置联动：stall→允许短时观察；surge→立即收油门；undecided→继续监视

**3) 噪声与退化鲁棒性**（`robustness.py` → `robustness.md/png`）
- SNR 扫描：旋转失速带检测到 SNR≈7 dB 仍 100% 检出；**发现并如实报告了高 SNR 病理**——噪声 <5 Pa 时基线缩到噪声底，慢漂移必然穿越相对阈值（FAR=100%），缓解方案为在线版自适应基线
- 通道增益失配 0.3–1.0：确认率/误报率/速度估计**完全不变**（归一化相干函数中静态增益相消）——闸门对灵敏度老化天然免疫
- 进气畸变（MG 模型注入特性线亏缺+成核扰动）：失稳起始随畸变幅值单调提前（0.05H 时提前 ~70 ξ），检测器全部 6/6 early——畸变诱发更早先兆，预警窗口反而增大

## 诚实的局限性

- 本数据集是**模型仿真**数据：真实跨音速压气机的激波/叶尖泄漏流耦合失稳机制远比 MG 模型复杂；
- MG 模型的"预失速"只有模态波型；突尖波（spike）型先兆未建模；
- 深喘振 episode（大 B）模态波被抑制、几乎无先兆窗口——这是真实物理也是检测的难点；
- TUDa 真实数据需签协议申请，到手后按 `docs/` 指南接入验证。
