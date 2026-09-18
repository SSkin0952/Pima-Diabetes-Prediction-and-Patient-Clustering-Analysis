# Pima Diabetes Prediction and Patient Clustering Analysis

[English](#english) | [中文](#中文)

---

<a id="english"></a>

## English

This project performs diabetes prediction and patient subtype discovery on the **Pima Indians Diabetes Dataset** using Logistic Regression, Gradient Boosting, Random Forest, MLP, bootstrap inference, KMeans, and hierarchical clustering.

### Sample Outputs

#### ROC Curve Comparison

![ROC Curve](output/figures/02_roc_curve_comparison.png)

#### Cluster Visualization

![Clusters](output/figures/09_kmeans_clusters_pca.png)

### Results

Model performance comparison:

| Model | Accuracy | ROC-AUC |
|---|---:|---:|
| Logistic Regression | 0.701 | 0.808 |
| Gradient Boosting | 0.766 | 0.833 |
| Random Forest | 0.734 | 0.814 |
| MLP Classifier | 0.747 | 0.817 |

Key findings:

- Gradient Boosting achieved the highest accuracy (0.766) and ROC-AUC (0.833).
- Glucose was identified as the most influential predictor of diabetes risk.
- BMI and Age also demonstrated strong predictive importance.
- Clustering analysis identified patient subgroups with different diabetes prevalence rates.

### Project Overview

The goal is to predict diabetes outcomes and explore possible patient subtypes based on medical features.

The project includes:

- data cleaning and missing-value imputation
- feature standardization
- classification models:
  - Linear Regression baseline
  - Logistic Regression
  - Gradient Boosting
  - Random Forest
  - MLP Classifier
- model evaluation:
  - Accuracy
  - Classification report
  - Confusion matrix
  - ROC curve and AUC
  - Calibration curve
- bootstrap confidence intervals for logistic regression coefficients
- unsupervised clustering:
  - KMeans clustering
  - Hierarchical clustering
  - PCA visualization
  - Cluster outcome distribution

### Dataset

Dataset: **Pima Indians Diabetes Database**

The script downloads the dataset automatically using `kagglehub`.

### Requirements

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
tqdm
kagglehub
```

### Project Structure

```text
Pima Diabetes Analysis/
├── Diabetes analyse.py
├── README.md
└── output/
    ├── figures/
    ├── hierarchical_cluster_summary.csv
    ├── kmeans_cluster_summary.csv
    ├── logistic_coef_bootstrap_summary.csv
    ├── model_auc_comparison.csv
    ├── model_metrics_summary.csv
    ├── pima_clusters.csv
    └── results_summary.txt
```

### How to Run

#### 1. Install Dependencies

Create a Python virtual environment (recommended):

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

#### 2. Run the Analysis

Execute the main script:

```bash
python "Diabetes analyse.py"
```

The script will automatically:

- download the Pima Diabetes dataset
- perform data preprocessing and imputation
- train multiple machine-learning models
- evaluate classification performance
- perform clustering analysis
- generate figures and summary files

### Outputs

All generated files are saved inside the `output/` directory.

#### Figures

The following visualizations are generated:

- Confusion Matrix Comparison
- ROC Curve Comparison
- Individual ROC Curves
- AUC Comparison Bar Chart
- Predicted Probability Distributions
- Calibration Curves
- Correlation Heatmap
- KMeans Elbow Plot
- KMeans Silhouette Score Plot
- PCA Cluster Visualization
- Cluster Centroid Heatmap
- Hierarchical Clustering Dendrogram
- Hierarchical Cluster PCA Visualization

#### CSV Results

The script exports several result files, including:

- `logistic_coef_bootstrap_summary.csv`: logistic regression coefficient means, 95% bootstrap confidence intervals, and feature-importance rankings
- `pima_clusters.csv`: original patient records with KMeans and hierarchical cluster assignments

### Machine-Learning Models

| Model | Description |
|---|---|
| Linear Regression | Baseline binary prediction model |
| Logistic Regression | Main interpretable classification model |
| Gradient Boosting | Boosted tree ensemble model |
| Random Forest | Bagging ensemble model |
| MLP Classifier | Feed-forward neural network |

Performance metrics include Accuracy, Precision, Recall, F1 Score, ROC-AUC, and Confusion Matrix.

### Clustering Analysis

Two unsupervised learning methods are used:

#### KMeans Clustering

- Elbow Method
- Silhouette Analysis
- Cluster Profiling
- Outcome Distribution Analysis

#### Hierarchical Clustering

- Ward Linkage
- Dendrogram Visualization
- Agglomerative Clustering
- Cluster Agreement Analysis (Adjusted Rand Index)

### Technologies

Python, Pandas, NumPy, Scikit-Learn, SciPy, Matplotlib, Seaborn, KaggleHub, and TQDM.

### Project Highlights

- end-to-end diabetes prediction pipeline
- comparison of linear, tree-based, and neural-network models
- bootstrap confidence-interval estimation
- patient subtype discovery through clustering
- automated figure generation
- fully executable standalone Python implementation

### Future Improvements

- XGBoost and LightGBM models
- hyperparameter optimization
- SHAP-based model interpretation
- cross-validation framework
- class-imbalance handling
- clinical risk-scoring system

---

<a id="中文"></a>

## 中文

本项目基于 **Pima Indians Diabetes Dataset（皮马印第安人糖尿病数据集）**，使用逻辑回归、梯度提升、随机森林、多层感知机、Bootstrap 推断、KMeans 和层次聚类，实现糖尿病预测与患者亚型探索。

### 示例输出

#### ROC 曲线对比

![ROC曲线](output/figures/02_roc_curve_comparison.png)

#### 聚类可视化

![聚类结果](output/figures/09_kmeans_clusters_pca.png)

### 实验结果

模型性能对比：

| 模型 | 准确率 | ROC-AUC |
|---|---:|---:|
| 逻辑回归 | 0.701 | 0.808 |
| 梯度提升 | 0.766 | 0.833 |
| 随机森林 | 0.734 | 0.814 |
| MLP 分类器 | 0.747 | 0.817 |

主要结论：

- 梯度提升模型取得最高准确率（0.766）和最高 ROC-AUC（0.833）。
- 血糖（Glucose）是影响糖尿病风险最显著的预测特征。
- BMI 和年龄（Age）同样具有较强的预测重要性。
- 聚类分析识别出了糖尿病患病率不同的患者群体。

### 项目概述

本项目旨在根据医学特征预测糖尿病结果，并探索潜在的患者亚型。

项目内容包括：

- 数据清洗与缺失值填补
- 特征标准化
- 分类模型：
  - 线性回归基线模型
  - 逻辑回归
  - 梯度提升
  - 随机森林
  - MLP 分类器
- 模型评估：
  - 准确率
  - 分类报告
  - 混淆矩阵
  - ROC 曲线与 AUC
  - 校准曲线
- 逻辑回归系数的 Bootstrap 置信区间
- 无监督聚类：
  - KMeans 聚类
  - 层次聚类
  - PCA 可视化
  - 各聚类结果的患病分布

### 数据集

数据集：**Pima Indians Diabetes Database**

程序通过 `kagglehub` 自动下载该数据集。

### 环境依赖

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
tqdm
kagglehub
```

### 项目结构

```text
Pima Diabetes Analysis/
├── Diabetes analyse.py
├── README.md
└── output/
    ├── figures/
    ├── hierarchical_cluster_summary.csv
    ├── kmeans_cluster_summary.csv
    ├── logistic_coef_bootstrap_summary.csv
    ├── model_auc_comparison.csv
    ├── model_metrics_summary.csv
    ├── pima_clusters.csv
    └── results_summary.txt
```

### 运行方法

#### 1. 安装依赖

建议先创建 Python 虚拟环境：

```bash
python -m venv .venv
```

在 Windows 中激活虚拟环境：

```powershell
.venv\Scripts\activate
```

安装所需依赖：

```bash
pip install -r requirements.txt
```

#### 2. 运行分析程序

执行主程序：

```bash
python "Diabetes analyse.py"
```

程序将自动完成：

- 下载 Pima 糖尿病数据集
- 数据预处理与缺失值填补
- 训练多个机器学习模型
- 评估分类性能
- 执行聚类分析
- 生成图表和汇总文件

### 输出结果

所有生成文件均保存在 `output/` 目录中。

#### 图表

程序将生成以下可视化结果：

- 混淆矩阵对比
- ROC 曲线对比
- 各模型 ROC 曲线
- AUC 对比柱状图
- 预测概率分布
- 校准曲线
- 相关性热力图
- KMeans 肘部法则图
- KMeans 轮廓系数图
- PCA 聚类可视化
- 聚类中心热力图
- 层次聚类树状图
- 层次聚类 PCA 可视化

#### CSV 结果

程序将导出多个结果文件，其中包括：

- `logistic_coef_bootstrap_summary.csv`：逻辑回归系数均值、95% Bootstrap 置信区间及特征重要性排名
- `pima_clusters.csv`：包含 KMeans 与层次聚类标签的原始患者记录

### 机器学习模型

| 模型 | 说明 |
|---|---|
| 线性回归 | 二分类预测基线模型 |
| 逻辑回归 | 主要的可解释分类模型 |
| 梯度提升 | 基于提升方法的树集成模型 |
| 随机森林 | 基于 Bagging 的集成模型 |
| MLP 分类器 | 前馈神经网络 |

评估指标包括准确率、精确率、召回率、F1 分数、ROC-AUC 和混淆矩阵。

### 聚类分析

本项目采用两种无监督学习方法：

#### KMeans 聚类

- 肘部法则
- 轮廓系数分析
- 聚类特征分析
- 结果分布分析

#### 层次聚类

- Ward 连接法
- 树状图可视化
- 凝聚层次聚类
- 聚类一致性分析（调整兰德指数）

### 使用技术

Python、Pandas、NumPy、Scikit-Learn、SciPy、Matplotlib、Seaborn、KaggleHub 和 TQDM。

### 项目亮点

- 端到端糖尿病预测流程
- 对比线性模型、树模型和神经网络模型
- 使用 Bootstrap 估计置信区间
- 通过聚类发现患者亚型
- 自动生成可视化图表
- 可独立执行的完整 Python 实现

### 后续改进

- 引入 XGBoost 和 LightGBM
- 超参数优化
- 基于 SHAP 的模型解释
- 交叉验证框架
- 类别不平衡处理
- 临床风险评分系统

---

## Author / 作者

Wei Sun  
M.S. Mechanical Engineering, Columbia University  
Expected Graduation / 预计毕业时间：December 2026
