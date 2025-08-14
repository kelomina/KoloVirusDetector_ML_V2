# 神枢 - Axon V2

基于机器学习的恶意软件检测和分类系统，使用LightGBM和深度学习技术实现恶意软件的检测和家族分类。

## 项目概述

神枢 - Axon V2 是一个基于机器学习的恶意软件检测系统，能够：

1. 检测文件是否为恶意软件（二分类）
2. 对恶意软件进行家族分类（多分类）

该系统使用多种技术组合来实现高精度的恶意软件检测和分类，包括：
- 基于字节序列的统计特征提取
- PE文件结构分析
- LightGBM机器学习模型
- HDBSCAN聚类算法进行恶意软件家族发现

## 目录结构

```
KoloVirusDetectorML/
├── benign_samples/           # 正常软件样本
├── malicious_samples/        # 恶意软件样本
├── data/                     # 处理后的数据集
├── saved_models/             # 保存的模型文件
├── hdbscan_cluster_results/  # HDBSCAN聚类结果
├── feature_extractor_enhanced.py  # 特征提取模块
├── pretrain.py               # 预训练模块
├── finetune.py               # 微调模块
├── scanner.py                # 扫描器模块
├── extracted_features.pkl    # 提取的特征文件
└── model_evaluation.png      # 模型评估结果图
```

## 功能模块

### 1. 特征提取 (feature_extractor_enhanced.py)

该模块负责从可执行文件中提取特征，包括：
- 字节序列统计特征
- 文件熵值计算
- PE文件结构信息
- 文件属性信息

### 2. 预训练 (pretrain.py)

实现恶意软件的二分类检测功能：
- 使用LightGBM训练二分类模型
- 区分正常软件和恶意软件
- 提供模型评估和验证功能

### 3. 微调 (finetune.py)

实现恶意软件家族分类功能：
- 使用HDBSCAN聚类算法发现恶意软件家族
- 对恶意软件进行细粒度分类
- 可视化聚类结果

### 4. 扫描器 (scanner.py)

提供命令行接口用于扫描文件：
- 单个文件扫描
- 目录批量扫描
- 输出详细的扫描报告

## 安装依赖

```bash
pip install numpy pandas scikit-learn lightgbm matplotlib seaborn tqdm pefile torch fast-hdbscan
```

## 使用方法

### 1. 训练二分类模型

```bash
python pretrain.py --data_dir data/processed_lightgbm --metadata_file data/metadata.json
```

### 2. 进行恶意软件家族聚类

```bash
python finetune.py --features_file extracted_features.pkl --output_dir hdbscan_cluster_results
```

### 3. 扫描文件

```bash
# 扫描单个文件
python scanner.py --model saved_models/lightgbm_model.txt --file /path/to/file.exe

# 扫描整个目录
python scanner.py --model saved_models/lightgbm_model.txt --dir /path/to/directory
```

## 模型性能

模型性能评估结果可参考 `model_evaluation.png` 文件，其中包括：
- 准确率指标
- 混淆矩阵
- ROC曲线

## 数据集

项目使用以下数据集进行训练和测试：
- 正常软件样本：`benign_samples/` 目录
- 恶意软件样本：`malicious_samples/` 目录

处理后的数据存储在 `data/processed_lightgbm/` 目录中。

## 聚类结果

HDBSCAN聚类结果保存在 `hdbscan_cluster_results/` 目录中，包括：
- 聚类标签文件
- 聚类可视化图表
- 家族名称映射

## 项目特点

1. **高准确性**：结合多种特征和先进的机器学习算法
2. **高效性**：优化的特征提取和预测过程
3. **可扩展性**：模块化设计，易于添加新功能
4. **可视化**：提供丰富的结果可视化功能

## 贡献

欢迎提交Issue和Pull Request来改进项目。

## 许可证

本项目采用Apache-2.0许可证，详情请见LICENSE文件。





# Shenshu - Axon V2

A machine learning-based malware detection and classification system that leverages LightGBM and deep learning techniques to detect malware and classify it into families.

## Project Overview

**Shenshu - Axon V2** is a machine learning-powered malware detection system capable of:

1. Detecting whether a file is malicious (binary classification)  
2. Classifying malware into specific families (multi-class classification)

The system combines multiple advanced techniques to achieve high-precision malware detection and classification, including:
- Statistical feature extraction from byte sequences
- PE file structure analysis
- LightGBM machine learning models
- HDBSCAN clustering algorithm for malware family discovery

## Directory Structure

```
KoloVirusDetectorML/
├── benign_samples/           # Benign software samples
├── malicious_samples/        # Malware samples
├── data/                     # Processed datasets
├── saved_models/             # Saved model files
├── hdbscan_cluster_results/  # HDBSCAN clustering results
├── feature_extractor_enhanced.py  # Feature extraction module
├── pretrain.py               # Pre-training module
├── finetune.py               # Fine-tuning module
├── scanner.py                # Scanner module
├── extracted_features.pkl    # Extracted features file
└── model_evaluation.png      # Model evaluation result chart
```

## Functional Modules

### 1. Feature Extraction (feature_extractor_enhanced.py)

This module extracts features from executable files, including:
- Statistical features from byte sequences
- File entropy calculation
- PE file structural information
- File attribute metadata

### 2. Pre-training (pretrain.py)

Implements binary malware detection:
- Trains a binary classification model using LightGBM
- Distinguishes between benign and malicious software
- Provides model evaluation and validation capabilities

### 3. Fine-tuning (finetune.py)

Implements malware family classification:
- Uses the HDBSCAN clustering algorithm to discover malware families
- Performs fine-grained classification of malware samples
- Visualizes clustering results

### 4. Scanner (scanner.py)

Provides a command-line interface for scanning files:
- Single file scanning
- Batch scanning of directories
- Outputs detailed scan reports

## Installation Dependencies

```bash
pip install numpy pandas scikit-learn lightgbm matplotlib seaborn tqdm pefile torch fast-hdbscan
```

## Usage Instructions

### 1. Train the Binary Classification Model

```bash
python pretrain.py --data_dir data/processed_lightgbm --metadata_file data/metadata.json
```

### 2. Perform Malware Family Clustering

```bash
python finetune.py --features_file extracted_features.pkl --output_dir hdbscan_cluster_results
```

### 3. Scan Files

```bash
# Scan a single file
python scanner.py --model saved_models/lightgbm_model.txt --file /path/to/file.exe

# Scan an entire directory
python scanner.py --model saved_models/lightgbm_model.txt --dir /path/to/directory
```

## Model Performance

Refer to `model_evaluation.png` for model evaluation results, which include:
- Accuracy metrics
- Confusion matrix
- ROC curve

## Dataset

The project uses the following datasets for training and testing:
- Benign software samples: located in `benign_samples/`
- Malware samples: located in `malicious_samples/`

Processed data is stored in the `data/processed_lightgbm/` directory.

## Clustering Results

HDBSCAN clustering results are saved in the `hdbscan_cluster_results/` directory, including:
- Cluster label files
- Clustering visualization charts
- Malware family name mappings

## Key Features

1. **High Accuracy**: Combines multiple features and advanced machine learning algorithms  
2. **Efficiency**: Optimized feature extraction and prediction pipeline  
3. **Scalability**: Modular design that supports easy extension and integration of new features  
4. **Visualization**: Rich visualization tools for analysis and reporting

## Contributions

We welcome contributions via Issues and Pull Requests to improve the project.

## License

This project is licensed under the Apache-2.0 License. See the LICENSE file for details.
