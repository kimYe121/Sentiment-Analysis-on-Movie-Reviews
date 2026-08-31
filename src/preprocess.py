from __future__ import annotations

import re
from typing import Iterable

import pandas as pd


def clean_text(text: str) -> str:
    """进行基础文本清洗。"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_sentence_context(train_df: pd.DataFrame) -> pd.DataFrame:
    """根据 SentenceId 聚合句子级上下文信息。"""
    sentence_info = (
        train_df.groupby("SentenceId", as_index=False)
        .agg(
            sentence_text=("Phrase", lambda x: " ".join(str(v) for v in x if str(v).strip())),
            sentence_length=("Phrase", "count"),
        )
    )
    sentence_info.rename(columns={"sentence_text": "sentence_context"}, inplace=True)
    return train_df.merge(sentence_info, on="SentenceId", how="left")


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """对文本列进行清洗，并补充上下文语句特征。"""
    out = df.copy()
    out["Phrase"] = out["Phrase"].apply(clean_text)
    if "SentenceId" in out.columns:
        out["sentence_context"] = out.groupby("SentenceId")["Phrase"].transform(
            lambda s: " ".join(str(v) for v in s if str(v).strip())
        )
    return out


def combine_phrase_and_sentence(df: pd.DataFrame) -> pd.DataFrame:
    """生成短语和句子拼接后的文本字段。"""
    out = df.copy()
    out["phrase_sentence_text"] = out["Phrase"].fillna("") + " " + out.get("sentence_context", "").fillna("")
    return out


def extract_text_features(df: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """构造最终用于建模的文本字段。"""
    if columns is None:
        columns = ["Phrase", "sentence_context", "phrase_sentence_text"]

    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].fillna("").astype(str)
    return out
