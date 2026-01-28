# 快速开始指南

## 环境准备

### Python环境

推荐使用Python 3.8或更高版本。

### 安装依赖

```bash
pip install -r requirements.txt
```

### 依赖包说明

- **numpy**: 数值计算
- **pandas**: 数据处理
- **matplotlib**: 数据可视化
- **scikit-learn**: 机器学习
- **networkx**: 网络分析
- **bayesian-optimization**: 贝叶斯优化

## 快速运行

### 1. 运行LWR宏观分析

```bash
python scripts/run_lwr_analysis.py
```

**输出文件**:

- `src/lwr/lwr_traffic_analysis.png` - 宏观交通流分析图
- `src/lwr/critical_bottlenecks.csv` - 瓶颈路段数据

### 2. 运行元胞自动机仿真

```bash
python scripts/run_cellular_automaton.py
```

**输出文件**:

- `src/cellular_automaton/road_snapshot_real.png` - 道路快照
- `src/cellular_automaton/change_points_analysis.png` - 变点分析

### 3. 运行贝叶斯优化

```bash
python scripts/run_cellular_automaton.py
# 选择选项 2
```

**输出文件**:

- `src/cellular_automaton/bayesian_optimization_results.png` - 优化过程图
- `src/cellular_automaton/optimized_parameters.json` - 最优参数

### 4. 运行网络均衡分析

```bash
python scripts/run_network_equilibrium.py
```

**输出文件**:

- `src/network_equilibrium/network_equilibrium_results.json` - 网络均衡结果
- `src/network_equilibrium/network_analysis_simple.png` - 网络分析图

## 完整流程

### 一键运行所有分析

```bash
# 运行贝叶斯优化（参数校准）
python 元胞自动机模型/bayesian_optimization.py

# 运行集成模型
python 元胞自动机模型/integrate_ml2.py

# 运行网络均衡分析
python 网络均衡模型/network_equilibrium_model.py

# 运行LWR分析
python LWR/lwr_analysis.py
```

## 数据文件说明

### 输入数据

- `2017_MCM_Problem_C_Data.csv` - 华盛顿州高速公路交通数据

### 输出数据类型

- **JSON**: 结构化分析结果
- **CSV**: 数值数据表格
- **PNG**: 可视化图表
- **Pickle**: 训练好的机器学习模型

## 常见问题

### Q: 运行时出现内存不足错误

A: 减少仿真规模，修改代码中的参数：

```python
# 在CA.py中减少道路长度
road_length = 500  # 从1000减少到500
```

### Q: 缺少依赖包

A: 安装缺失的包：

```bash
pip install missing_package_name
```

### Q: 运行时间过长

A:

1. 减少仿真时间步数
2. 使用更小的道路规模
3. 启用并行计算（如果支持）

### Q: 结果不准确

A:

1. 检查输入数据格式
2. 调整模型参数
3. 增加仿真时间

## 性能优化建议

### 1. 内存优化

```python
# 及时释放大对象
del large_array
import gc
gc.collect()
```

### 2. 计算优化

```python
# 使用向量化操作
import numpy as np
# 而不是循环
for i in range(n):
    result[i] = calculation(data[i])
# 使用
result = np.vectorize(calculation)(data)
```

### 3. 并行计算

```python
from multiprocessing import Pool

def parallel_function(args):
    # 你的计算函数
    pass

# 使用并行计算
with Pool() as pool:
    results = pool.map(parallel_function, arg_list)
```

## 结果解读

### 交通流特性图

- **横轴**: 交通密度
- **纵轴**: 交通流量
- **峰值**: 最大通行能力

### 瓶颈识别

- **高拥堵率**: 瓶颈路段
- **低速度**: 拥堵区域
- **高密度**: 流量饱和

### 自动驾驶影响

- **AV渗透率**: 自动驾驶车辆占比
- **网络性能**: 总体交通效率
- **最佳比例**: 最优AV渗透率

## 下一步

1. **深入学习**: 阅读详细的模型文档
2. **参数调优**: 根据具体场景调整参数
3. **扩展应用**: 应用到其他交通网络
4. **算法改进**: 尝试更先进的算法

## 联系支持

如有问题，请查看：

- `docs/model_documentation.md` - 详细模型文档
- `docs/analysis_summary.md` - 分析总结报告
- 项目README.md - 项目概述
