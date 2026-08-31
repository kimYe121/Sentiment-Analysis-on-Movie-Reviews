# Sentiment Analysis on Movie Reviews

## 项目概述

本项目基于 Kaggle 竞赛 "Sentiment Analysis on Movie Reviews"，目标是对 Rotten Tomatoes 影评短语进行 5 级细粒度情感分类。

标签定义：

- 0: negative
- 1: somewhat negative
- 2: neutral
- 3: somewhat positive
- 4: positive

## 数据说明

- 训练集：data/train.tsv/train.tsv
- 测试集：data/test.tsv/test.tsv
- 主要字段：PhraseId、SentenceId、Phrase、Sentiment（训练集）
- 数据规模：约 156,060 条训练样本，约 66,292 条测试样本

说明：
- 训练集中标签为 0~4，属于多分类任务。
- 短语文本通常较短，且依赖整句上下文；SentenceId 可用于构造句子级上下文特征。

## 目录结构

```text
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── data/
│   ├── train.tsv/
│   │   └── train.tsv
│   └── test.tsv/
│       └── test.tsv
├── src/
│   ├── __init__.py
│   ├── utils.py
│   ├── preprocess.py
│   ├── features.py
│   ├── train_classical.py
│   ├── train_textcnn.py
│   ├── train_bilstm.py
│   ├── train_bert.py
│   └── evaluate.py
├── results/
├── notebooks/
├── report/
└── .venv/   # 由 uv 创建，通常不纳入版本控制
```

## 模型路线

### 经典机器学习

1. TF-IDF + Logistic Regression
2. TF-IDF + Multinomial Naive Bayes
3. TF-IDF + Linear SVC
4. TF-IDF + Random Forest

### 深度学习

5. TextCNN
6. BiLSTM
7. BERT Fine-tuning

## 评价指标

- Accuracy
- Macro F1
- Weighted F1
- Per-class Precision / Recall
- Confusion matrix
- ROC-AUC (One-vs-Rest)

## 运行方式（uv）

```bash
# 1) 创建虚拟环境
uv venv

# 2) 激活环境
# Windows PowerShell
.venv\Scripts\Activate.ps1

# 3) 安装依赖
uv pip install -r requirements.txt

# 4) 运行经典模型训练
python src/train_classical.py --model lr --cv 3 --random_state 42

# 5) 运行 TextCNN
python src/train_textcnn.py --epochs 5 --batch_size 64

# 6) 运行 BiLSTM
python src/train_bilstm.py --epochs 5 --batch_size 64

# 7) 运行 BERT
python src/train_bert.py --epochs 2 --batch_size 32 --max_samples 10000
```

## 说明

- 所有模型应使用相同数据划分和相同评估口径，确保公平比较。
- 本目录仅作为项目框架，后续将逐步补充完整训练逻辑、评估脚本与实验报告。
- 参考方法：
  - TextCNN: Kim, 2014, EMNLP
  - LSTM: Hochreiter & Schmidhuber, 1997, Neural Computation
  - BERT: Devlin et al., 2019, NAACL

## 后续完善方向

- 构造 `Phrase` 与 `Sentence` 两级特征
- 统一 train/validation 分割
- 输出 `results/comparison.csv`
- 对不同参数组合进行对照实验
- 生成混淆矩阵和 ROC 图
