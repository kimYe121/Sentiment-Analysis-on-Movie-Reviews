from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess import prepare_dataframe
from src.utils import DATA_DIR, ROOT_DIR, set_seed


MODEL_MAP = {
    "lr": LogisticRegression(max_iter=2000, multi_class="auto", solver="liblinear"),
    "nb": MultinomialNB(),
    "svc": LinearSVC(),
    "rf": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train classical text classifiers for movie review sentiment classification.")
    parser.add_argument("--model", choices=["lr", "nb", "svc", "rf"], default="lr")
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--train_path", type=str, default=str(DATA_DIR / "train.tsv" / "train.tsv"))
    return parser.parse_args()


def build_pipeline(model_name: str):
    """构造 TF-IDF + 分类器管道。"""
    model = MODEL_MAP[model_name]
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, strip_accents="unicode", sublinear_tf=True)),
        ("model", model),
    ])


def main() -> None:
    args = parse_args()
    set_seed(args.random_state)

    train_df = pd.read_csv(args.train_path, sep='\t')
    train_df = prepare_dataframe(train_df)

    X = train_df["Phrase"].fillna("").astype(str)
    y = train_df["Sentiment"].astype(int)

    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.random_state)
    pipeline = build_pipeline(args.model)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=1)

    print(f"Model={args.model}, CV Accuracy={scores.mean():.4f} ± {scores.std():.4f}")
    pipeline.fit(X, y)
    pred = pipeline.predict(X)
    print(classification_report(y, pred, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y, pred))


if __name__ == "__main__":
    main()
