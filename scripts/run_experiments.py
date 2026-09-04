"""实验编排脚本：把课程设计需要的全部实验组织成可复现的命令序列。

所有实验通过各训练脚本的命令行参数定义，产物按
``results/<family>/<model>/<exp_name>/`` 契约自动落盘；本脚本只负责
按计划顺序执行、跳过已完成的实验、打印进度。

用法：
    python scripts/run_experiments.py --group <组名>          # 跑某一组（自动跳过已完成）
    python scripts/run_experiments.py --group all             # 跑全部组
    python scripts/run_experiments.py --group all --dry_run   # 只预览命令，不执行
    python scripts/run_experiments.py --group base --force    # 强制重跑（忽略已完成）

实验组一览（耗时为本机 RTX 4060 量级）：
    base        三模型正式结果（stratified 9:1 划分）                 约 40 分钟
                textcnn/base、bilstm/base、bert/base
    nn_compare  BiLSTM 手写实现 vs nn.LSTM 库实现（同构对照）          约 1 分钟
                bilstm/base_nn
    grouped     句级分组划分对照（无泄漏指标，防泄漏分析用）           约 10 分钟
                textcnn/base_grouped、bilstm/base_grouped
    context     BERT 句对上下文融合 (完整句子, 短语)（自主改进实验）   约 20 分钟
                bert/ctx

从零复现全部结果的完整流水线：
    # ① 统一划分（首次运行一次即可，data/split/ 生成名单缓存）
    python src/common/split.py
    # ② 全部训练实验（约 70 分钟；BERT 需先设 HF 镜像，设一次当窗口有效）
    $env:HF_ENDPOINT = "https://hf-mirror.com"
    python scripts/run_experiments.py --group all
    # ③ 三模型概率平均集成（约 10 秒）
    python scripts/ensemble.py
    # ④ 汇总表 comparison.csv + 全部报告图
    python scripts/make_figures.py

说明：
- 断点续跑：每个实验完成后落盘 metrics.json，重新执行同一命令会自动跳过
  已完成部分；中途 Ctrl+C 中断后，直接重跑同一条命令即可从断点继续。
- 单独调试某个模型时可直接运行训练脚本，各训练脚本的 docstring 里有
  该模型的完整参数示例（如 train_bilstm.py / train_bert.py）。
- 队友的经典模型（src/models/classical/）无编排组，单独运行，例如：
  python src/models/classical/train_linear_svc.py --exp_name base
  结果同样按契约落盘，会被 make_figures 自动并入汇总表和图表。
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (family, model, exp_name, 训练脚本相对路径 + 参数)
GROUPS: dict[str, list[tuple[str, str, str, str]]] = {
    "base": [
        ("dl", "textcnn", "base", "src/models/deep_learning/train_textcnn.py --exp_name base"),
        ("dl", "bilstm", "base", "src/models/deep_learning/train_bilstm.py --exp_name base --batch_size 128"),
        ("dl", "bert", "base", "src/models/deep_learning/train_bert.py --exp_name base"),
    ],
    "nn_compare": [
        ("dl", "bilstm", "base_nn", "src/models/deep_learning/train_bilstm.py --exp_name base_nn --impl nn --batch_size 128"),
    ],
    "grouped": [
        # 防泄漏对照。BERT 版已裁撤（省 25 分钟），两项已完成的数据足够支撑泄漏分析
        ("dl", "textcnn", "base_grouped", "src/models/deep_learning/train_textcnn.py --exp_name base_grouped --mode grouped"),
        ("dl", "bilstm", "base_grouped", "src/models/deep_learning/train_bilstm.py --exp_name base_grouped --mode grouped"),
    ],
    "context": [
        # 上下文融合（自主改进）：BERT 句对输入 (完整句子, 短语)，交叉注意力
        # 做词级跨段交互。TextCNN/BiLSTM 已各有两组对照，融合实验聚焦 BERT。
        ("dl", "bert", "ctx", "src/models/deep_learning/train_bert.py --exp_name ctx --use_context --max_len 96 --batch_size 48 --epochs 2"),
    ],
}


def metrics_path(family: str, model: str, exp_name: str) -> Path:
    return ROOT / "results" / family / model / exp_name / "metrics.json"


def run_group(group: str, dry_run: bool, force: bool) -> None:
    experiments = GROUPS[group]
    print(f"\n===== 实验组 [{group}]：共 {len(experiments)} 个实验 =====")
    for i, (family, model, exp_name, cmd) in enumerate(experiments, 1):
        done = metrics_path(family, model, exp_name).exists()
        status = "已完成，跳过" if (done and not force) else ("将执行" if not done else "强制重跑")
        print(f"\n[{i}/{len(experiments)}] {family}/{model}/{exp_name}  ->  {status}")
        print(f"  命令: {cmd}")
        if dry_run:
            continue
        if done and not force:
            continue

        t0 = time.time()
        # Windows 无法直接执行 .py，统一用当前解释器启动训练脚本
        result = subprocess.run([sys.executable, *shlex.split(cmd)], cwd=ROOT)
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f"\n[中止] 实验 {exp_name} 失败（exit={result.returncode}），剩余实验未执行。")
            sys.exit(result.returncode)
        print(f"[完成] {exp_name}  耗时 {elapsed / 60:.1f} 分钟")

    print(f"\n===== 实验组 [{group}] 结束 =====")
    print("汇总与出图: python scripts/make_figures.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment groups defined for the course design.")
    parser.add_argument("--group", type=str, required=True,
                        choices=(*GROUPS.keys(), "all"))
    parser.add_argument("--dry_run", action="store_true", help="只打印命令不执行")
    parser.add_argument("--force", action="store_true", help="忽略已完成状态强制重跑")
    args = parser.parse_args()

    groups = list(GROUPS.keys()) if args.group == "all" else [args.group]
    for group in groups:
        run_group(group, dry_run=args.dry_run, force=args.force)


if __name__ == "__main__":
    main()
