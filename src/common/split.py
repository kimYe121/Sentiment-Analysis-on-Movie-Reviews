"""统一的数据划分模块：全项目所有模型共用同一份验证集，保证横向可比。

提供两种划分方式：

- ``stratified``：按 Sentiment 分层随机抽样。这是主线指标，便于与常见公开
  结果对比；但同一句子的短语之间高度重叠，按短语随机划分会把同一句子的
  子短语同时放进训练集与验证集，存在标签泄漏，指标会偏高。
- ``grouped``：按 SentenceId 分组划分，训练集与验证集的句子完全不重叠，
  是无泄漏的诚实指标，报告中用于论证两种划分下的指标差异。

划分结果落盘到 ``data/split/``（git 不追踪）。训练脚本统一通过
``ensure_split`` 获取划分：文件已存在则直接读取，不存在则按当前种子生成
并落盘。由于生成过程完全由 seed 决定，任何人重新运行都会得到同一份划分。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.utils import SPLIT_DIR, TRAIN_PATH, ensure_dirs

SUPPORTED_MODES = ("stratified", "grouped")


def _stratified_val_ids(df: pd.DataFrame, val_ratio: float, seed: int) -> np.ndarray:
    """按标签分层随机抽取验证集 PhraseId。"""
    rng = np.random.default_rng(seed)
    val_ids: list[np.ndarray] = []
    for _, group in df.groupby("Sentiment"):
        ids = group["PhraseId"].to_numpy(copy=True)
        rng.shuffle(ids)
        n_val = max(1, int(round(len(ids) * val_ratio)))
        val_ids.append(ids[:n_val])
    return np.concatenate(val_ids)


def _grouped_val_ids(df: pd.DataFrame, val_ratio: float, seed: int) -> np.ndarray:
    """按句子分组抽取验证集 PhraseId，保证句子不跨集合。"""
    rng = np.random.default_rng(seed)
    sentence_ids = np.array(sorted(df["SentenceId"].unique()))
    rng.shuffle(sentence_ids)
    n_val = max(1, int(round(len(sentence_ids) * val_ratio)))
    val_sentences = set(sentence_ids[:n_val].tolist())
    return df.loc[df["SentenceId"].isin(val_sentences), "PhraseId"].to_numpy()


def split_dataframe(df: pd.DataFrame, val_ids: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按验证集 PhraseId 把数据切成 (train_part, val_part)。"""
    val_id_set = set(np.asarray(val_ids).tolist())
    mask = df["PhraseId"].isin(val_id_set)
    train_part = df.loc[~mask].reset_index(drop=True)
    val_part = df.loc[mask].reset_index(drop=True)
    return train_part, val_part


def _split_stem(mode: str, val_ratio: float, seed: int) -> str:
    """划分文件名的稳定主干，编码 mode/val_ratio/seed。

    把 val_ratio / seed 纳入文件名，避免改动参数后静默复用旧缓存
    （此前只按 mode 命名，换 val_ratio 或 seed 会读到旧的 0.1 划分）。
    """
    ratio = f"{val_ratio:g}"  # 0.1 -> "0.1"，去除浮点噪声
    return f"{mode}_r{ratio}_s{seed}"


def ensure_split(
    df: pd.DataFrame,
    mode: str = "stratified",
    val_ratio: float = 0.1,
    seed: int = 42,
    split_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """获取统一的训练/验证划分（按 mode/val_ratio/seed 命名缓存，不存在则生成并落盘）。"""
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"不支持的划分方式: {mode}，可选 {SUPPORTED_MODES}")

    ensure_dirs()
    split_dir = Path(split_dir) if split_dir is not None else SPLIT_DIR
    split_dir.mkdir(parents=True, exist_ok=True)
    stem = _split_stem(mode, val_ratio, seed)
    id_path = split_dir / f"val_phrase_ids_{stem}.csv"
    meta_path = split_dir / f"split_meta_{stem}.json"

    if id_path.exists():
        val_ids = pd.read_csv(id_path)["PhraseId"].to_numpy()
    else:
        if mode == "stratified":
            val_ids = _stratified_val_ids(df, val_ratio, seed)
        else:
            val_ids = _grouped_val_ids(df, val_ratio, seed)
        pd.DataFrame({"PhraseId": sorted(val_ids.tolist())}).to_csv(id_path, index=False)

        _, val_part = split_dataframe(df, val_ids)
        meta = {
            "mode": mode,
            "val_ratio": val_ratio,
            "seed": seed,
            "train_size": int(len(df) - len(val_part)),
            "val_size": int(len(val_part)),
            "val_sentence_count": int(val_part["SentenceId"].nunique()),
            "label_distribution": {
                str(k): int(v) for k, v in val_part["Sentiment"].value_counts().sort_index().items()
            },
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return split_dataframe(df, val_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成并检查统一数据划分。")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, default="all", choices=("all",) + SUPPORTED_MODES)
    args = parser.parse_args()

    train_df = pd.read_csv(TRAIN_PATH, sep="\t")
    modes = SUPPORTED_MODES if args.mode == "all" else (args.mode,)
    for mode in modes:
        train_part, val_part = ensure_split(train_df, mode=mode, val_ratio=args.val_ratio, seed=args.seed)
        print(f"[{mode}] train={len(train_part)}  val={len(val_part)}  "
              f"val句子数={val_part['SentenceId'].nunique()}")
        print(f"[{mode}] 验证集标签分布: {val_part['Sentiment'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
