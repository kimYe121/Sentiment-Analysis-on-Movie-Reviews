"""实验编排脚本：把课程设计需要的全部实验组织成可复现的命令序列。

所有实验通过各训练脚本的命令行参数定义，产物按
``results/<family>/<model>/<exp_name>/`` 契约自动落盘；本脚本只负责
按计划顺序执行、跳过已完成的实验、打印进度。

用法：
    python scripts/run_experiments.py --group nn_compare          # 手写 vs 库对照
    python scripts/run_experiments.py --group grouped             # 分组划分对照（无泄漏）
    python scripts/run_experiments.py --group ablation            # 参数消融矩阵（课设硬性要求）
    python scripts/run_experiments.py --group ablation --dry_run  # 只打印将执行的命令
    python scripts/run_experiments.py --group all                 # 全部（含 base，从零复现用）

实验组说明：
- base:       三个模型的正式结果（stratified 划分）
- nn_compare: BiLSTM 手写实现 vs nn.LSTM 库实现，同一结构只换循环核心
- grouped:    三模型在句级分组划分下的无泄漏指标
- ablation:   每个模型固定其他参数、每次只改一个参数，与 base 对比即为该参数的影响
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
        ("dl", "textcnn", "base_grouped", "src/models/deep_learning/train_textcnn.py --exp_name base_grouped --mode grouped"),
        ("dl", "bilstm", "base_grouped", "src/models/deep_learning/train_bilstm.py --exp_name base_grouped --mode grouped"),
        ("dl", "bert", "base_grouped", "src/models/deep_learning/train_bert.py --exp_name base_grouped --mode grouped"),
    ],
    "ablation": [
        # ---- TextCNN：控制 batch/epochs 与 base 一致，每次只改一个参数 ----
        ("dl", "textcnn", "filters64", "src/models/deep_learning/train_textcnn.py --exp_name filters64 --num_filters 64"),
        ("dl", "textcnn", "filters256", "src/models/deep_learning/train_textcnn.py --exp_name filters256 --num_filters 256"),
        ("dl", "textcnn", "dropout03", "src/models/deep_learning/train_textcnn.py --exp_name dropout03 --dropout 0.3"),
        ("dl", "textcnn", "dropout07", "src/models/deep_learning/train_textcnn.py --exp_name dropout07 --dropout 0.7"),
        ("dl", "textcnn", "kernels2345", "src/models/deep_learning/train_textcnn.py --exp_name kernels2345 --kernel_sizes 2,3,4,5"),
        ("dl", "textcnn", "wd001", "src/models/deep_learning/train_textcnn.py --exp_name wd001 --weight_decay 0.01"),
        # ---- BiLSTM：batch 128 与优化后默认一致 ----
        ("dl", "bilstm", "hidden64", "src/models/deep_learning/train_bilstm.py --exp_name hidden64 --hidden_size 64"),
        ("dl", "bilstm", "hidden256", "src/models/deep_learning/train_bilstm.py --exp_name hidden256 --hidden_size 256"),
        ("dl", "bilstm", "pool_mean", "src/models/deep_learning/train_bilstm.py --exp_name pool_mean --pooling mean"),
        ("dl", "bilstm", "pool_last", "src/models/deep_learning/train_bilstm.py --exp_name pool_last --pooling last"),
        ("dl", "bilstm", "layers2", "src/models/deep_learning/train_bilstm.py --exp_name layers2 --num_layers 2"),
        ("dl", "bilstm", "unidirectional", "src/models/deep_learning/train_bilstm.py --exp_name unidirectional --bidirectional 0"),
        # ---- BERT：2 轮即可（base 的过拟合拐点在第 2 轮）----
        ("dl", "bert", "lr3e5", "src/models/deep_learning/train_bert.py --exp_name lr3e5 --lr 3e-5 --epochs 2"),
        ("dl", "bert", "lr5e5", "src/models/deep_learning/train_bert.py --exp_name lr5e5 --lr 5e-5 --epochs 2"),
        ("dl", "bert", "maxlen32", "src/models/deep_learning/train_bert.py --exp_name maxlen32 --max_len 32 --epochs 2"),
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
