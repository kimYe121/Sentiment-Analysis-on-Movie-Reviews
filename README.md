# Sentiment Analysis on Movie Reviews

## 项目简介

本项目是一个面向课程设计的电影评论情感分析实验仓库，任务来源于 Kaggle 的 `Sentiment Analysis on Movie Reviews` 数据集，研究目标是对 Rotten Tomatoes 电影评论短语进行五分类情感识别。

情感标签定义如下：

- `0`：negative
- `1`：somewhat negative
- `2`：neutral
- `3`：somewhat positive
- `4`：positive

当前仓库已完成项目结构整理，重点是为后续的模型训练、实验对比、结果分析和课程报告撰写提供清晰的组织基础。

## 课题目标

- 完成电影评论短语的细粒度情感分类任务。
- 对比传统机器学习方法与深度学习方法的建模效果。
- 统一实验目录、评估方式和结果输出，便于课程设计展示与复现。

## 数据说明

默认数据目录为 `data/`，代码中约定的数据路径如下：

- 训练集：`data/train.tsv`
- 测试集：`data/test.tsv`

数据集常见字段包括：

- `PhraseId`
- `SentenceId`
- `Phrase`
- `Sentiment`（仅训练集包含）

其中 `SentenceId` 可用于构造句子级上下文特征，适合在短语级情感分类任务中补充语义信息。

## 项目结构

```text
SentimentAnalysisOnMovieReviews/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── data/                         # 原始数据目录，默认不纳入版本控制
│   ├── .gitkeep
│   ├── train.tsv
│   └── test.tsv
├── notebooks/                    # 实验记录、可视化分析 Notebook
│   └── .gitkeep
├── report/                       # 课程报告、答辩材料
│   └── .gitkeep
├── results/                      # 预测结果、图表、评估输出
│   └── .gitkeep
└── src/
    ├── __init__.py
    ├── common/                   # 通用工具与数据预处理
    │   ├── __init__.py
    │   ├── preprocess.py
    │   └── utils.py
    ├── evaluation/               # 统一评估与可视化入口
    │   ├── __init__.py
    │   └── evaluate.py
    └── models/                   # 按模型类型组织训练脚本
        ├── __init__.py
        ├── classical/            # 传统机器学习模型
        │   ├── __init__.py
        │   ├── train_linear_svc.py
        │   ├── train_logistic_regression.py
        │   ├── train_multinomial_nb.py
        │   └── train_random_forest.py
        └── deep_learning/        # 深度学习模型
            ├── __init__.py
            ├── train_bert.py
            ├── train_bilstm.py
            └── train_textcnn.py
```

## 目录设计说明

- `src/common/`：放置文本清洗、特征准备、随机种子、数据路径等公共逻辑。
- `src/evaluation/`：统一管理评估指标和图表输出，避免评估逻辑分散。
- `src/models/classical/`：集中管理传统机器学习实验脚本，便于横向比较不同方法。
- `src/models/deep_learning/`：集中管理深度学习模型脚本，便于后续扩展训练与推理流程。
- `results/`、`report/`、`notebooks/`：分别对应实验结果、课程文档和过程分析，适合课设提交与归档。

## 模型路线

本项目当前规划的实验路线如下：

### 传统机器学习

- TF-IDF + Logistic Regression
- TF-IDF + Multinomial Naive Bayes
- TF-IDF + Linear SVC
- TF-IDF + Random Forest

### 深度学习

- TextCNN
- BiLSTM
- BERT Fine-tuning

## 评估指标

项目中的统一评估脚本围绕以下指标进行设计：

- Accuracy
- Macro F1
- Weighted F1
- Per-class Precision / Recall
- Confusion Matrix

## 环境配置

### 方式一：使用 uv

```bash
uv venv
.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

### 方式二：使用 pip

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 使用说明

### 1. 准备数据

请先将数据集按如下形式放入 `data/`：

```text
data/
├── train.tsv
└── test.tsv
```

### 2. 运行示例

当前仓库中，`BERT` 脚本保留了相对完整的参数入口，其余训练脚本主要作为后续实验扩展的结构占位。

运行 `BERT` 脚本示例：

```bash
python src/models/deep_learning/train_bert.py --epochs 2 --batch_size 32 --max_samples 10000
```

运行统一评估脚本示例：

```bash
python src/evaluation/evaluate.py --pred_path results/pred.csv --label_path results/label.csv --output_dir results
```

## 当前完成情况

- 已完成基础项目骨架整理，适合继续开展课程设计实验。
- 已拆分通用模块、模型模块和评估模块，目录职责清晰。
- 已保留传统机器学习与深度学习两条实验路线的脚本入口。
- 已提供文本预处理、数据读取、随机种子控制和统一评估脚本。

## 后续可完善方向

- 增加统一的训练入口与参数管理。
- 补充 `TextCNN`、`BiLSTM` 和传统机器学习模型的完整训练逻辑。
- 增加实验结果汇总表，例如 `results/comparison.csv`。
- 完善课程报告，包括方法设计、实验设置、结果分析与结论。
- 增加更多可视化内容，如标签分布图、ROC 曲线和误差分析。

## 参考方向

- TextCNN: Yoon Kim, 2014, EMNLP
- LSTM: Hochreiter and Schmidhuber, 1997
- BERT: Devlin et al., 2019, NAACL

