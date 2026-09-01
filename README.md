# Sentiment Analysis on Movie Reviews

## 项目简介

本项目是一个面向课程设计的电影评论情感分析实验仓库，任务来源于 Kaggle 的 `Sentiment Analysis on Movie Reviews` 数据集，研究目标是对 Rotten Tomatoes 电影评论短语进行五分类情感识别。

情感标签定义如下：

- `0`：negative
- `1`：somewhat negative
- `2`：neutral
- `3`：somewhat positive
- `4`：positive

## 课题目标

- 完成电影评论短语的细粒度情感分类任务。
- 对比传统机器学习方法与深度学习方法的建模效果。
- 统一实验目录、评估方式和结果输出，便于课程设计展示与复现。

## 数据说明

数据目录为 `data/`（已加入 `.gitignore`，不入库）：

- 训练集：`data/train.tsv`（156,060 条短语，8,544 个句子）
- 测试集：`data/test.tsv`（66,292 条短语，无标签，Kaggle 提交格式）
- 统一划分：`data/split/`（由 `src/common/split.py` 生成，确定性可复现）

字段包括 `PhraseId`、`SentenceId`、`Phrase`、`Sentiment`（仅训练集）。

### 关于数据划分（重要）

测试集没有标签，所有模型对比都在从训练集切出的验证集上进行。同一句子的短语之间高度重叠，按短语随机划分会把同一句子的子短语同时放进训练集与验证集（标签泄漏，指标偏高）。因此项目提供两种统一划分：

- `stratified`（默认）：按标签分层随机 9:1 划分，主线指标，便于与常见公开结果对比；
- `grouped`：按 `SentenceId` 分组 9:1 划分，训练/验证句子完全不重叠，是无泄漏的诚实指标，报告中用于对比论证。

所有训练脚本通过 `--mode` 参数选择，划分文件全项目共享，保证不同模型横向可比。

## 项目结构

```text
SentimentAnalysisOnMovieReviews/
├── README.md
├── pyproject.toml
├── requirements.txt
├── data/                         # 数据与划分文件（不入库）
├── notebooks/                    # 实验记录、可视化分析 Notebook
├── report/                       # 课程报告、答辩材料（不入库）
├── results/                      # 实验产物与汇总图表（不入库）
└── src/
    ├── common/
    │   ├── utils.py              # 路径常量、随机种子、数据读取
    │   ├── preprocess.py         # 文本清洗与句子上下文特征
    │   ├── split.py              # 统一训练/验证划分（stratified / grouped）
    │   ├── experiment.py         # 实验产物落盘契约（ExperimentLogger）
    │   ├── dl_data.py            # 手写词表、编码、批迭代器（深度学习用）
    │   └── dl_train.py           # 手写训练循环、梯度裁剪、早停
    ├── evaluation/
    │   ├── evaluate.py           # 统一指标计算与混淆矩阵图
    │   └── aggregator.py         # 汇总所有实验，生成对比表和对比图
    └── models/
        ├── classical/            # 传统机器学习（经典 ML 组负责）
        │   ├── train_logistic_regression.py   # 参考实现/模板
        │   ├── train_multinomial_nb.py
        │   ├── train_linear_svc.py
        │   └── train_random_forest.py
        └── deep_learning/        # 深度学习（手写实现为主）
            ├── layers.py         # 手写基础层（嵌入/线性/dropout/交叉熵）+ 一致性验证
            ├── textcnn.py        # 手写 TextCNN（含手写卷积）
            ├── bilstm.py         # 手写 LSTM 单元、双向展开、注意力池化
            ├── train_textcnn.py
            ├── train_bilstm.py
            └── train_bert.py     # BERT 微调（手写训练循环与调度器）
```

## 实验产物契约（所有训练脚本必须遵守）

每个实验在 `results/<family>/<model>/<exp_name>/` 下产出固定文件，`aggregator.py` 据此自动汇总：

| 文件 | 内容 |
|---|---|
| `config.json` | 全部超参、种子、库版本、数据规模 |
| `metrics.json` | accuracy / macro F1 / weighted F1 / 各类别 P、R、F1 |
| `history.csv` | 逐轮 train_loss、val_loss、val_acc、val_macro_f1 |
| `pred_val.csv` / `label_val.csv` | 验证集预测与真实标签 |
| `submission.csv` | Kaggle 测试集提交文件 |

汇总产物（`aggregator.py` 自动生成）：

- `results/comparison.csv`：全部实验指标汇总表
- `results/comparison_metrics.png`：各实验 Accuracy / Macro F1 对比柱状图
- `results/training_curves.png`：验证损失与精度训练曲线
- `results/confusion_matrices.png`：各实验混淆矩阵拼图

**原则：训练脚本只负责训练与落盘；汇总工具只读文件、不参与训练。**

## 快速开始

### 1. 环境配置

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

国内网络下载 HuggingFace 模型建议设置镜像：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

### 2. 生成统一划分（首次运行一次即可）

```bash
python src/common/split.py
```

### 3. 训练深度学习模型

```bash
# TextCNN（全手写，含手写卷积，启动时打印与 nn.Conv1d 的一致性验证）
python src/models/deep_learning/train_textcnn.py --exp_name base

# BiLSTM（手写 LSTM 单元 + 注意力池化）
python src/models/deep_learning/train_bilstm.py --exp_name base

# BERT 微调（模型权重来自 HuggingFace，训练循环手写）
python src/models/deep_learning/train_bert.py --exp_name base

# 快速调试：加 --max_samples 20000 --epochs 2
```

### 4. 训练传统机器学习模型

```bash
python src/models/classical/train_logistic_regression.py --exp_name base
# 其余经典模型请参照 train_logistic_regression.py 模板实现
```

### 5. 汇总与可视化

```bash
python src/evaluation/aggregator.py
```

### 6. 单个实验的评估与混淆矩阵

```bash
python src/evaluation/evaluate.py --pred_path results/dl/textcnn/base/pred_val.csv --label_path results/dl/textcnn/base/label_val.csv --output_dir results/dl/textcnn/base
```

## 模型路线

### 传统机器学习（经典 ML 组）

- TF-IDF + Logistic Regression（已提供参考实现）
- TF-IDF + Multinomial Naive Bayes
- TF-IDF + Linear SVC
- TF-IDF + Random Forest

### 深度学习（手写实现说明）

| 模型 | 实现方式 | 复现文献 |
|---|---|---|
| TextCNN | 全流程手写：词表、批处理、卷积（unfold+einsum）、训练循环；启动时与 `nn.Conv1d` 做数值一致性验证 | Kim, 2014, EMNLP |
| BiLSTM | 手写 LSTM 单元与双向时间步展开，attention/mean/last 三种池化；与 `nn.LSTM` 做数值一致性验证 | Hochreiter & Schmidhuber, 1997；Zhou et al., 2016, ACL |
| BERT | 模型权重与 tokenizer 来自 HuggingFace（复现发表方法对照项），训练循环、warmup+线性衰减调度、混合精度均为手写，不使用 Trainer | Devlin et al., 2019, NAACL |

## 评估指标

- Accuracy
- Macro F1
- Weighted F1
- Per-class Precision / Recall / F1
- Confusion Matrix

注意：验证集标签分布不均衡（neutral 约占 51%），macro F1 通常明显低于 accuracy，报告中两者都应呈现。

## 当前完成情况

- 已完成统一数据划分模块（stratified / grouped 两种，全项目共享同一份验证集）。
- 已建立实验产物落盘契约与结果自动汇总工具（comparison.csv + 对比图表）。
- 已完成深度学习三个模型：TextCNN / BiLSTM 全手写实现，BERT 手写训练循环微调。
- 已提供经典机器学习参考实现模板（TF-IDF + Logistic Regression）。
- 已提供文本预处理、随机种子控制和统一评估脚本。

## 后续可完善方向

- 经典 ML 组：参照模板补齐 Multinomial NB / Linear SVC / Random Forest 及其参数对比实验。
- 深度学习组：按 `results/dl/<model>/<exp_name>/` 命名约定跑参数消融（filters / kernels / dropout；hidden / pooling / 层数；BERT 的 lr / max_len / epochs）。
- 句子上下文特征融合改进实验（`preprocess.py` 已提供 `sentence_context` 字段）。
- 完善课程报告：方法设计、实验设置、结果对比分析（可直接引用 comparison.csv 与汇总图）、误差分析。

## 参考方向

- TextCNN: Kim Y. Convolutional Neural Networks for Sentence Classification[C]//EMNLP 2014: 1746-1751.
- LSTM: Hochreiter S, Schmidhuber J. Long Short-Term Memory[J]. Neural Computation, 1997, 9(8): 1735-1780.
- Attention BiLSTM: Zhou P, et al. Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification[C]//ACL 2016: 207-212.
- BERT: Devlin J, et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding[C]//NAACL-HLT 2019: 4171-4186.
- 数据集: Socher R, et al. Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank[C]//EMNLP 2013: 1631-1642.
