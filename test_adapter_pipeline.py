# -*- coding: utf-8 -*-
"""
TUDa adapter 管线端到端测试

用仿真数据模拟 TUDa 数据流 (不同采样率/转频), 验证:
  重采样 -> 切片 -> sanity_check -> 离线/在线检测 -> GT 后备判据
真实数据到手后, 只需实现 read_raw_*() 即可复用全部下游代码。
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from tuda_adapter import (build_episodes, resample_uniform, sanity_check,
                          run_detection, pressure_fluctuation_criterion,
                          WPTLayout)
from array_processing import synthesize_array

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

# 模拟 TUDa 台架参数 (真实值以 datasheet/测量日志为准)
FS_TUDA = 25600.0        # 假设采样率
F_ROTOR_TUDA = 600.0     # 假设物理转速对应转频
EPISODE_FILE = "episode_026_rotating_stall.npz"


def main():
    d = np.load(os.path.join(DATA, EPISODE_FILE), allow_pickle=True)
    arr = synthesize_array(d, n_sensors=8, seed=7)

    # 1) 模拟台架采样率不同 -> 重采样
    t2, P2 = resample_uniform(arr["t"], arr["P"], FS_TUDA)
    print(f"[1] 重采样: {arr['fs']:.0f} Hz -> {FS_TUDA:.0f} Hz, "
          f"{P2.shape[0]} 样本 x {P2.shape[1]} 通道")

    # 2) 传感器布置元数据 (非均匀扩散安装示例)
    layout = WPTLayout(
        n_sensors=8,
        angles_deg=np.sort(np.random.default_rng(3).uniform(0, 360, 8)),
        axial_pos=[0.0] * 8)
    print(f"[2] WPTLayout: 8 通道, 周向角 "
          f"{np.round(layout.angles_deg, 1).tolist()} deg, "
          f"相邻间隔角 {np.round(np.rad2deg(layout.spacing_matrix()), 1).tolist()}")

    # 3) 切片
    eps = build_episodes(t2, P2, fs=FS_TUDA, f_rotor=F_ROTOR_TUDA,
                         seg_s=4.0, hop_s=2.0)
    print(f"[3] 切片: {len(eps)} episodes, 每段 {len(eps[0]['t'])} 样本")

    # 4) 数据体检 + 检测
    ok = True
    for i, ep in enumerate(eps):
        checks = sanity_check(ep)
        ok &= checks
        res_off = run_detection(ep, online=False, k_stall=4, k_surge=5)
        res_onl = run_detection(ep, online=True, k_stall=4, k_surge=5)
        t_est = pressure_fluctuation_criterion(ep)
        print(f"    ep{i}: offline alarm={res_off['alarm_time']} "
              f"({res_off['alarm_type']}), online alarm={res_onl['alarm_time']}, "
              f"GT估计={t_est and round(t_est, 3)}")

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
