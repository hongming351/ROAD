# 交通流分析项目 (ROAD)

## 项目概述

这是一个综合性的交通流分析项目，包含多个模块用于分析自动驾驶车辆对交通网络的影响。

## 项目结构

```
ROAD/
├── data/                               # 原始数据
│   └── 2017_MCM_Problem_C_Data.csv     # 华盛顿州交通数据
├── src/                                # 源代码
│   ├── lwr/                           # LWR宏观交通流模型
│   │   ├── lwr_analysis.py            # LWR模型分析
│   │   ├── congestion_classifier.py     # 拥堵分类器
│   │   ├── data_analysis.py           # 数据分析
│   │   └── critical_bottlenecks.csv   # 瓶颈路段数据
│   ├── cellular_automaton/            # 元胞自动机微观模型
│   │   ├── CA.py                      # 元胞自动机核心模型
│   │   ├── bayesian_optimization.py   # 贝叶斯优化参数校准
│   │   ├── integrate_ml2.py           # ML2集成模块
│   │   ├── extract_simulation_data.py # 仿真数据提取
│   │   └── run_pelt_with_ca_data.py   # PELT变点检测
│   └── network_equilibrium/           # 网络均衡模型
│       ├── network_equilibrium_model.py # 网络均衡模型
│       └── python od_estimation.py    # OD需求估计
├── results/                            # 分析结果
│   ├── raw/                           # 原始分析结果
│   ├── processed/                     # 处理后的结果
│   └── visualizations/                # 可视化图表
├── scripts/                           # 运行脚本
│   ├── run_lwr_analysis.py            # LWR分析脚本
│   ├── run_cellular_automaton.py      # 元胞自动机脚本
│   ├── run_network_equilibrium.py     # 网络均衡脚本
│   └── run_all_models.py              # 全部模型运行脚本
├── docs/                              # 文档
│   ├── README.md                      # 项目说明文档
│   ├── quick_start.md                 # 快速开始指南
│   ├── model_documentation.md         # 模型文档
│   └── analysis_summary.md            # 分析总结报告
├── tests/                             # 测试文件
├── requirements.txt                   # Python依赖包
├── road_data_analysis_report.md       # 数据分析报告
└── .gitignore                         # Git忽略文件
```

## 主要模块

### 1. LWR宏观交通流模型 (LWR/)

- **lwr_analysis.py**: 基于LWR模型的宏观交通流分析
- **congestion_classifier.py**: 使用随机森林的拥堵分类器
- **data_analysis.py**: 交通数据基础分析

### 2. 元胞自动机微观模型 (元胞自动机模型/)

- **CA.py**: 两类车辆（人类/自动驾驶）的元胞自动机模型
- **bayesian_optimization.py**: 贝叶斯优化参数校准
- **integrate_ml2.py**: 集成贝叶斯优化的完整模型
- **extract_simulation_data.py**: 仿真数据提取和处理
- **run_pelt_with_ca_data.py**: PELT变点检测分析

### 3. 网络均衡模型 (网络均衡模型/)

- **network_equilibrium_model.py**: 修复的物理正确网络均衡模型
- 支持弹性需求和自动驾驶影响分析

### 4. OD需求分析 (results/)

- OD矩阵生成和分析
- 交通需求分布可视化

## 安装依赖

```bash
pip install -r requirements.txt
```

## 主要功能

1. **交通数据分析**: 分析华盛顿州高速公路交通数据
2. **拥堵识别**: 使用机器学习识别拥堵路段
3. **自动驾驶影响**: 分析自动驾驶车辆对交通流的影响
4. **参数优化**: 使用贝叶斯优化校准模型参数
5. **网络均衡**: 求解交通网络均衡状态
6. **变点检测**: 识别交通模式变化点

## 运行示例

### 运行LWR分析

```python
python LWR/lwr_analysis.py
```

### 运行元胞自动机仿真

```python
python 元胞自动机模型/CA.py
```

### 运行贝叶斯优化

```python
python 元胞自动机模型/bayesian_optimization.py
```

### 运行网络均衡分析

```python
python 网络均衡模型/network_equilibrium_model.py
```

## 输出文件

项目生成多种格式的输出文件：

- **JSON**: 结构化分析结果
- **CSV**: 数值数据表格
- **PNG**: 可视化图表
- **Pickle**: 训练好的机器学习模型

## 技术栈

- **Python**: 主要编程语言
- **Pandas/Numpy**: 数据处理
- **Matplotlib/Seaborn**: 数据可视化
- **Scikit-learn**: 机器学习
- **NetworkX**: 网络分析
- **Bayesian-Optimization**: 贝叶斯优化

## 项目特点

1. **多尺度建模**: 从微观到宏观的多层次交通分析
2. **机器学习集成**: 使用ML技术优化和预测
3. **真实数据驱动**: 基于华盛顿州实际交通数据
4. **可视化丰富**: 提供多种图表展示分析结果
5. **模块化设计**: 各模块可独立运行和测试

## 许可证

本项目采用 MIT 许可证。
