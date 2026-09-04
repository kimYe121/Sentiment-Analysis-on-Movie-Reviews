"""实验编排脚本：把课程设计需要的全部实验组织成可复现的命令序列。

所有实验通过各训练脚本的命令行参数定义，产物按
``results/<family>/<model>/<exp_name>/`` 契约自动落盘；本脚本只负责
按计划顺序执行、跳过已完成的实验、打印进度。

用法：
    python scripts/run_experiments.py --group context          # 上下文融合实验（自主改进）
    python scripts/run_experiments.py --group base             # 三模型正式结果
    python scripts/run_experiments.py --group all --dry_run    # 预览全部命令

实验组说明：
- base:       三个模型的正式结果（stratified 划分）
- nn_compare: BiLSTM 手写实现 vs nn.LSTM 库实现，同一结构只换循环核心
- grouped:    TextCNN / BiLSTM 在句级分组划分下的无泄漏指标
- context:    上下文融合（自主改进）：短语 + 所在完整句子双输入
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
        # 上下文融合（自主改进）：短语 + 所在完整句子。针对短语脱离语境
        # 无法判断的问题，预期显著提升，是有提升效果的改进证据。
        ("dl", "bilstm", "ctx", "src/models/deep_learning/train_bilstm.py --exp_name ctx --use_context --max_len 32 --ctx_max_len 48 --epochs 12 --patience 4"),
        ("dl", "bert", "ctx", "src/models/deep_learning/train_bert.py --exp_name ctx --use_context --max_len 80 --batch_size 24 --epochs 2"),
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
        result = subprocess.run(shlex.split(cmd), cwd=ROOT)
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f"\n[中止] 实验 {exp_name} 失败（exit={result.returncode}），剩余实验未执行。")
            sys.exit(result.returncode)
        print(f"[完成] {exp_name}  耗时 {elapsed / 60:.1f} 分钟")

    print(f"\n===== 实验组 [{group}] 结束 =====")
    print("汇总: python src/evaluation/aggregator.py")


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
