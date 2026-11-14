# -*- coding: utf-8 -*-
"""
一键复现全部论文结果 (顺序执行, 总计 ~15 min, CPU 即可)

步骤对应论文骨架:
  1. MG 仿真数据集 (若 data/ 为空) + 检测验证
  2. TUDa 校准版数据集 + 离线检测复跑
  3. TUDa stall run 双转速实测检测 (核心)
  4. P0-1 稳态点虚警评估
  5. P0-2 前兆时频搜索
  6. P0-3 在线化评测
  7. P0-4 DL 基线 + few-shot 跨域
  8. P1-5 POT/GPD 自动阈值
用法: python src/run_all.py [--quick]   # quick 跳过步骤 1 数据生成 (已存在时)
"""
import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = [
    ("MG 检测验证", "validate.py"),
    ("TUDa 校准版数据集检测", "generate_dataset_tuda.py"),
    ("TUDa stall run 双转速检测", "test_tuda_stall.py"),
    ("P0-1 稳态虚警评估", "false_alarm_eval.py"),
    ("P0-2 前兆时频搜索", "precursor_tf.py"),
    ("P0-3 在线化评测", "online_eval.py"),
    ("P0-4 DL 基线", "dl_baseline.py"),
    ("P0-4 DL few-shot 跨域", "dl_cross_domain.py"),
    ("P1-5 POT/GPD 自动阈值", "auto_threshold.py"),
    ("闭环避喘仿真 (讨论用)", "closed_loop.py"),
]


def main():
    quick = "--quick" in sys.argv
    for tag, script in STEPS:
        if quick and script in ("generate_dataset_tuda.py", "closed_loop.py"):
            print(f"[skip] {tag} (quick)")
            continue
        print(f"\n{'=' * 72}\n>>> {tag} ({script})\n{'=' * 72}")
        r = subprocess.run([sys.executable, os.path.join(HERE, script)])
        if r.returncode != 0:
            print(f"[FAIL] {tag} (exit {r.returncode})")
            sys.exit(1)
    print(f"\n{'=' * 72}\n全部步骤完成 -> results/\n{'=' * 72}")


if __name__ == "__main__":
    main()
