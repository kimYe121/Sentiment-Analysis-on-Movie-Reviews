# Sentiment Analysis on Movie Reviews

## 项目简介

本项目是面向课程设计的电影评论情感分析实验仓库，任务来源于 Kaggle 的 `Sentiment Analysis on Movie Reviews` 数据集，对 Rotten Tomatoes 电影评论短语进行五分类情感识别（`0` negative ~ `4` positive，neutral 约占 51%，类别不均衡）。

---

## 一、工作设计与分工（先读这一节）

### 1.1 总体流程

```
data/train.tsv ──▶ 统一划分 src/common/split.py（stratified / grouped 二选一）
                        │
                        ▼
        训练脚本 train_textcnn / train_bilstm / train_bert（一次调用 = 一个实验）
                        │  按"实验产物契约"落盘 results/<family>/<model>/<exp_name>/
                        │  （config.json / metrics.json / history.csv / 预测与提交文件）
                        ▲
编排脚本 scripts/run_experiments.py ── 按 GROUPS 清单批量、顺序调用训练脚本
                        │
                        ▼
        aggregator.py 汇总全部实验 ──▶ comparison.csv + 对比柱状图/训练曲线/混淆矩阵
                        │                                    │
                        ▼                                    ▼
        submission.csv ──▶ Kaggle 官方分数          报告直接引用的表格与图
```

### 1.2 分工

| 成员 | 负责内容 |
|---|---|
| 深度学习组（本人） | 三个深度模型的实现（手写为主）、全部公共基础设施（划分/契约/汇总）、深度学习侧的全部实验与消融 |
| 经典 ML 组（队友） | TF-IDF + 朴素贝叶斯 / Linear SVC / 随机森林，**参照 `train_logistic_regression.py` 模板实现**，走同一划分与同一落盘契约 |

### 1.3 必须分清的四组概念

| 概念 | 含义 |
|---|---|
| 训练脚本 vs 编排脚本 | 训练脚本**一次只跑一个实验**；编排脚本只按清单**批量调用**训练脚本，自身不做任何训练。没有自动触发机制，脚本跑完清单即退出 |
| stratified vs grouped | stratified（按短语分层随机 9:1）= 主表指标，与常见公开做法可比但含短语重叠泄漏；grouped（按句子分组 9:1）= 训练/验证句子零重叠的无泄漏指标，**和 Kaggle 榜可比**。两者都报，落差即泄漏效应的定量证据 |
| test.tsv 的用途 | 无标签，本地算不了指标；唯一用途是每个实验生成的 `submission.csv` 传 Kaggle 换官方分数 |
| 手写实现 vs `--impl nn` | **正式结果全部由手写模型产出**；`--impl nn` 只是对照实验（同结构只换循环核心），用于证明手写实现正确（精度一致）并量化与 cuDNN 的速度差距（约 19 倍） |

### 1.4 深度学习部分的"设计工作"构成

1. **手写组件层**（`layers.py`/`textcnn.py`/`bilstm.py`）：嵌入查表、线性层、dropout、softmax 交叉熵、一维卷积（unfold+einsum）、LSTM 单元与双向展开、注意力池化——全部不调用 nn.* 现成模块，每个组件附与库实现的**数值一致性验证**（训练启动时自动打印，最大误差 3.7e-09 ~ 3.6e-07）；
2. **手写管线与训练循环**（`dl_data.py`/`dl_train.py`）：词表、批迭代器（不用 DataLoader）、前向/反向/梯度裁剪/早停/最优权重保留；
3. **方法学设计**：双划分防泄漏体系、实验产物契约、结果自动汇总——保证"对比实验"这一课设要求成立；
4. **自主改进**：BiLSTM 效率优化（输入投影外提 + 批内动态长度，155s→45s/轮，3.4×，数学等价）；
5. **BERT 对照项**：模型权重来自 HuggingFace（复现 Devlin et al. 2019 微调协议），训练循环/调度器/混合精度手写。

---

## 二、实验计划总表（本设计的全部实验）

| 实验组 | 做什么 | 目的 | 状态 |
|---|---|---|---|
| base（主实验） | TextCNN / BiLSTM / BERT 各跑一次 | 三模型正式成绩与完整指标 → 报告主对比表 | ✅ 0.6723 / 0.6756 / 0.7044 |
| nn_compare（库对照） | BiLSTM `--impl nn`，跑一次 | 证明手写实现正确（精度接近）并量化与 cuDNN 的速度差（约 19 倍） | ✅ 0.6738 |
| grouped（防泄漏对照） | TextCNN / BiLSTM 以 `--mode grouped` 重跑 | 量化两种划分的指标落差（泄漏效应） | ✅ 0.6085 / 0.6272 |
| ablation（消融） | 每模型改一参数重跑（脚本就绪） | 参数敏感性对比 | ❌ 按进度决定取消；配置对比由"旧设计 vs 新设计"（dropout 0.5→0.7、+AdamW/标签平滑/余弦退火的前后成绩）与算法间/划分/实现对比承担，需要时可 `--group ablation` 一条命令补跑 |
| 可选 | 预训练词向量初始化 / 句子上下文融合 | 特征增强（加分项，暂缓） | ⬜ 可选 |

消融清单（每次只改一个参数，与同模型 base 对比即得该参数的影响）：

- TextCNN：`--num_filters 256`（模型容量）、`--dropout 0.7`（正则化强度）
- BiLSTM：`--hidden_size 256`（模型容量）、`--pooling mean`（池化方式，对照 attention）
- BERT：`--max_len 32`（输入长度配置，约 15 分钟；为满足"每个算法都有参数对比"，如需进一步省时可删除）

---

## 三、课题目标

- 完成电影评论短语的细粒度情感分类任务。
- 对比传统机器学习方法与深度学习方法的建模效果。
- 统一实验目录、评估方式和结果输出，便于课程设计展示与复现。

## 四、数据说明

数据目录为 `data/`（已加入 `.gitignore`，不入库）：

- 训练集：`data/train.tsv`（156,060 条短语，8,544 个句子）
- 测试集：`data/test.tsv`（66,292 条短语，无标签，Kaggle 提交格式）
- 统一划分：`data/split/`（由 `src/common/split.py` 生成，确定性可复现）

字段包括 `PhraseId`、`SentenceId`、`Phrase`、`Sentiment`（仅训练集）。同一句子的短语高度重叠，划分不当会造成标签泄漏——这是双划分设计的动机（见 1.3）。

## 五、项目结构

```text
SentimentAnalysisOnMovieReviews/
├── README.md
├── pyproject.toml / requirements.txt
├── scripts/
│   └── run_experiments.py        # 实验编排（四组实验清单）
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
    │   ├── dl_data.py            # 手写词表、编码、批迭代器（含批内动态长度）
    │   └── dl_train.py           # 手写训练循环、梯度裁剪、早停
    ├── evaluation/
    │   ├── evaluate.py           # 统一指标计算与混淆矩阵图
    │   └── aggregator.py         # 汇总所有实验，生成对比表和对比图
    └── models/
        ├── classical/            # 传统机器学习（经典 ML 组负责）
        │   ├── train_logistic_regression.py   # 参考实现/模板
        │   ├── train_multinomial_nb.py / train_linear_svc.py / train_random_forest.py
        └── deep_learning/
            ├── layers.py         # 手写基础层 + 一致性验证
            ├── textcnn.py        # 手写 TextCNN（手写卷积）
            ├── bilstm.py         # 手写 LSTM 单元/双向展开/注意力池化 + nn.LSTM 对照封装
            ├── train_textcnn.py / train_bilstm.py / train_bert.py
```

## 六、实验产物契约（所有训练脚本必须遵守）

每个实验在 `results/<family>/<model>/<exp_name>/` 下产出固定文件，`aggregator.py` 据此自动汇总：

| 文件 | 内容 |
|---|---|
| `config.json` | 全部超参、种子、库版本、数据规模 |
| `metrics.json` | accuracy / macro F1 / weighted F1 / 各类别 P、R、F1 |
| `history.csv` | 逐轮 train_loss、val_loss、val_acc、val_macro_f1 |
| `pred_val.csv` / `label_val.csv` | 验证集预测与真实标签 |
| `submission.csv` | Kaggle 测试集提交文件 |

汇总产物（`aggregator.py` 自动生成）：`results/comparison.csv`、`comparison_metrics.png`（指标柱状图）、`training_curves.png`（训练曲线）、`confusion_matrices.png`（混淆矩阵拼图）。

**原则：训练脚本只负责训练与落盘；汇总工具只读文件、不参与训练。**

### 可视化产物清单（统一由 `scripts/make_figures.py` 生成）

| 可视化 | 支撑报告章节 |
|---|---|
| `results/eda_label_distribution.png` | 数据分析：类别不均衡（中性 51%） |
| `results/eda_length_distribution.png` | 数据分析：句长分布 → max_len 截断决策 |
| `results/model_comparison.png` | 模型对比：accuracy / macro F1 + 多数类基线 |
| `results/training_curves.png` | 训练过程与过拟合治理分析 |
| `results/per_class_f1.png` | 类别不均衡：各类别 F1 对比（两端类别短板） |
| `results/error_structure.png` | 误差分析：错误集中于相邻情感等级（序数性质） |
| `results/attention_heatmap_*.png` | 模型设计亮点：注意力权重的可解释性 |
| `results/comparison.csv` | 全部实验的数值汇总表（`src/evaluation/aggregator.py` 生成） |

## 七、快速开始

### 1. 环境配置

方式一（推荐）：uv

```bash
uv venv
.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt

# 免激活运行
uv run python src/models/deep_learning/train_textcnn.py --exp_name demo
```

方式二：pip

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

国内网络下载 HuggingFace 模型建议设置镜像（PowerShell）：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

### 2. 生成统一划分（首次运行一次即可）

```bash
python src/common/split.py
```

### 3. 跑实验与出图：三个入口脚本

```bash
# ① 训练实验（唯一训练入口；实验组见脚本 GROUPS：base / nn_compare / grouped / context）
python scripts/run_experiments.py --group context --dry_run   # 预览命令
python scripts/run_experiments.py --group context             # 正式执行（自动跳过已完成）

# ② 报告图表（唯一出图入口：EDA、主对比、训练曲线、各类别F1、误差结构、注意力热力图）
python scripts/make_figures.py

# ③ 概率集成（读取各实验保存的 probs_val.npy 做平均，生成 ensemble 实验）
python scripts/ensemble.py
```

也可以绕过编排脚本手动跑单个实验（调试用）：

```bash
python src/models/deep_learning/train_textcnn.py --exp_name base
python src/models/deep_learning/train_bilstm.py --exp_name ctx --use_context
$env:HF_ENDPOINT="https://hf-mirror.com"; python src/models/deep_learning/train_bert.py --exp_name base
```

### 4. 训练传统机器学习模型（经典 ML 组）

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

## 八、模型路线与手写实现说明

### 传统机器学习（经典 ML 组）

- TF-IDF + Logistic Regression（已提供参考实现）
- TF-IDF + Multinomial Naive Bayes / Linear SVC / Random Forest

### 深度学习（手写实现说明）

| 模型 | 实现方式 | 复现文献 |
|---|---|---|
| TextCNN | 全流程手写：词表、批处理、卷积（unfold+einsum）、训练循环 | Kim, 2014, EMNLP |
| BiLSTM | 手写 LSTM 单元与双向时间步展开，attention/mean/last 三种池化；含批内动态长度与输入投影外提两项效率优化 | Hochreiter & Schmidhuber, 1997；Zhou et al., 2016, ACL |
| BERT | 权重与 tokenizer 来自 HuggingFace（复现发表方法对照项），训练循环、warmup+线性衰减调度、fp16 手写，不用 Trainer | Devlin et al., 2019, NAACL |

## 九、评估指标

- Accuracy / Macro F1 / Weighted F1 / Per-class Precision·Recall·F1 / Confusion Matrix

注意：验证集标签分布不均衡（neutral 约 51%），macro F1 通常明显低于 accuracy，报告中两者都应呈现；accuracy 与 macro F1 的差距本身就是类别不均衡的证据。

## 十、当前结果速览与进度

**主对比表**（stratified 验证集 15,606 条；多数类基线 accuracy = 0.512）：

| 模型 | accuracy | macro F1 | weighted F1 | 验证见顶轮次 |
|---|---|---|---|---|
| TextCNN（手写） | 0.6723 | 0.5588 | 0.6649 | 第 9 轮（共练 15 轮） |
| BiLSTM + Attention（手写） | 0.6756 | 0.5759 | 0.6690 | 第 6 轮（共练 11 轮） |
| **BERT（微调）** | **0.7044** | **0.6296** | **0.7045** | 第 2 轮（微调协议 2~4 轮） |
| BiLSTM（nn.LSTM 对照） | 0.6738 | 0.5666 | 0.6681 | 与手写版等价，快约 19 倍 |
| TextCNN（grouped 划分） | 0.6085 | 0.4294 | 0.5857 | — |
| BiLSTM（grouped 划分） | 0.6272 | 0.4746 | 0.6077 | — |

**结果解读要点**：① 针对早期版本的过拟合（第 3 轮即见顶），模型设计升级为 dropout 0.7/0.6 + AdamW 权重衰减 + 标签平滑 0.1 + 余弦学习率退火，验证指标见顶轮次推迟到第 9/6 轮、训练曲线显著变长，精度基本持平（±0.3pt）——该数据集 15.6 万条短语仅来自 8.5 千个句子，有效多样性低，饱和后继续训练只会过拟合，早停保留的是最优轮权重；② macro F1 系统性低于 accuracy，源自类别不均衡（neutral 占 51%）；③ 分层与分组划分的落差（TextCNN 6.4pt、BiLSTM 4.8pt）量化了短语重叠泄漏的影响；④ 手写与 nn.LSTM 精度等价而速度差约 19 倍，差距源于框架级算子融合而非算法本身。

进度：

- ✅ 公共基础设施：统一划分、实验契约、汇总工具、评估脚本、可视化体系
- ✅ 深度学习三模型实现（手写为主）+ 数值一致性验证
- ✅ BiLSTM 效率优化（3.4×）与手写 vs 库对照实验
- ✅ base / nn_compare / grouped 实验全部完成（上表），过拟合治理后的最终版
- ✅ 可视化：EDA 图、指标柱状图、训练曲线、混淆矩阵、注意力热力图（均已按最终权重重新生成）
- ❌ 独立消融矩阵按进度决定取消（配置对比由新旧设计前后成绩与算法间/划分/实现对比承担）
- ⬜ 课程报告；可选：预训练词向量初始化、句子上下文融合

## 十一、参考方向

- Kim Y. Convolutional Neural Networks for Sentence Classification[C]//EMNLP 2014: 1746-1751.
- Hochreiter S, Schmidhuber J. Long Short-Term Memory[J]. Neural Computation, 1997, 9(8): 1735-1780.
- Zhou P, et al. Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification[C]//ACL 2016: 207-212.
- Devlin J, et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding[C]//NAACL-HLT 2019: 4171-4186.
- Socher R, et al. Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank[C]//EMNLP 2013: 1631-1642.
