# 模型文档

## 1. LWR宏观交通流模型

### 模型原理
LWR (Lighthill-Whitham-Richards) 模型是宏观交通流理论的基础模型，基于连续性方程描述交通流的宏观特性。

### 数学模型
```
∂ρ/∂t + ∂q/∂x = 0
q = ρ * v(ρ)
```

其中：
- ρ: 交通密度 (veh/km)
- q: 交通流量 (veh/h)
- v: 交通速度 (km/h)
- t: 时间
- x: 空间位置

### 实现文件
- `LWR/lwr_analysis.py` - LWR模型实现
- `LWR/data_analysis.py` - 数据分析和预处理
- `LWR/congestion_classifier.py` - 拥堵分类器

### 主要功能
1. **交通流特性分析**: 分析密度-流量关系
2. **拥堵识别**: 使用机器学习识别拥堵状态
3. **瓶颈检测**: 识别交通网络中的瓶颈路段
4. **可视化**: 生成交通流特性图表

### 输出结果
- `LWR/lwr_traffic_analysis.png` - 宏观交通流分析图
- `LWR/critical_bottlenecks.csv` - 关键瓶颈路段数据
- `LWR/congestion_classifier_model.pkl` - 训练好的拥堵分类器

## 2. 元胞自动机微观模型

### 模型原理
元胞自动机模型模拟单个车辆的行为，考虑两类车辆：人类驾驶车辆(HV)和自动驾驶车辆(AV)。

### 模型规则
1. **加速**: 车辆加速到最大速度
2. **减速**: 根据前车距离减速
3. **随机减速**: 以一定概率随机减速（HV概率更高）
4. **位置更新**: 根据速度更新位置

### 实现文件
- `元胞自动机模型/CA.py` - 元胞自动机核心实现
- `元胞自动机模型/bayesian_optimization.py` - 参数优化
- `元胞自动机模型/integrate_ml2.py` - ML2集成模型

### 参数说明
```python
# 车辆参数
max_speed = 5  # 最大速度 (cells/time step)
safety_distance = 2  # 安全距离 (cells)
random_slow_down_hv = 0.3  # HV随机减速概率
random_slow_down_av = 0.1  # AV随机减速概率

# 网络参数
road_length = 1000  # 道路长度 (cells)
num_lanes = 3  # 车道数
```

### 主要功能
1. **微观仿真**: 模拟单个车辆行为
2. **AV影响分析**: 分析自动驾驶对交通流的影响
3. **参数优化**: 使用贝叶斯优化校准参数
4. **数据提取**: 提取仿真数据用于分析

### 输出结果
- `change_points_analysis.png` - 变点分析图
- `元胞自动机模型/road_snapshot_real.png` - 道路快照
- `simulation_results_optimized.csv` - 优化仿真结果

## 3. 网络均衡模型

### 模型原理
基于Wardrop用户均衡原理，假设所有用户都选择最短路径，达到网络均衡状态。

### 数学模型
```
min ∑∫₀^x c_a(ω)dω
s.t. ∑f_p^rs = d^rs
     f_p^rs ≥ 0
```

其中：
- c_a: 链路a的阻抗函数
- f_p^rs: OD对(r,s)在路径p上的流量
- d^rs: OD对(r,s)的需求量

### 实现文件
- `网络均衡模型/network_equilibrium_model.py` - 网络均衡模型
- `网络均衡模型/python od_estimation.py` - OD需求估计

### 主要功能
1. **流量分配**: 将OD需求分配到网络路径
2. **弹性需求**: 考虑需求对阻抗的弹性
3. **AV影响**: 分析自动驾驶对网络性能的影响
4. **瓶颈识别**: 识别网络中的瓶颈路段

### 输出结果
- `网络均衡_results.json` - 网络均衡结果
- `network_analysis_simple.png` - 网络分析图
- `fixed_physical_analysis.png` - 物理正确性分析

## 4. 机器学习模型

### 拥堵分类器
使用随机森林算法识别交通拥堵状态。

#### 特征
- 交通流量
- 交通密度
- 平均速度
- 速度标准差
- 流量变化率

#### 实现文件
- `LWR/congestion_classifier.py` - 拥堵分类器实现

#### 输出结果
- `congestion_classifier_model.pkl` - 训练好的模型
- `congestion_classifier_scaler.pkl` - 特征缩放器
- `congestion_classifier_results.png` - 分类结果图

### 贝叶斯优化
使用贝叶斯优化算法校准元胞自动机模型参数。

#### 优化目标
最小化仿真结果与真实数据的差异。

#### 实现文件
- `元胞自动机模型/bayesian_optimization.py` - 贝叶斯优化实现

#### 输出结果
- `bayesian_optimization_results.png` - 优化过程图
- `optimized_parameters.json` - 最优参数
- `optimized_road_snapshot.png` - 优化后道路快照

## 5. 数据处理流程

### 数据预处理
1. **数据清洗**: 处理缺失值和异常值
2. **数据转换**: 标准化和归一化
3. **特征工程**: 提取有用的特征

### 实现文件
- `LWR/data_analysis.py` - 数据分析和预处理

### 数据格式
- **输入数据**: `2017_MCM_Problem_C_Data.csv`
- **输出数据**: 各种CSV、JSON和PNG格式的分析结果

## 6. 模型集成

### 多模型协作
项目采用多模型协作的方式，从不同尺度分析交通流：

1. **微观层面**: 元胞自动机模拟个体行为
2. **宏观层面**: LWR模型分析整体特性
3. **网络层面**: 均衡模型研究网络分配

### 数据流
```
原始数据 → 数据预处理 → 多模型分析 → 结果集成 → 可视化
```

### 集成文件
- `元胞自动机模型/integrate_ml2.py` - ML2集成模型
- `road_data_analysis_report.md` - 综合分析报告

## 7. 使用指南

### 运行单个模型
```bash
# 运行LWR分析
python LWR/lwr_analysis.py

# 运行元胞自动机仿真
python 元胞自动机模型/CA.py

# 运行网络均衡分析
python 网络均衡模型/network_equilibrium_model.py
```

### 运行完整流程
```bash
# 运行贝叶斯优化
python 元胞自动机模型/bayesian_optimization.py

# 运行集成模型
python 元胞自动机模型/integrate_ml2.py
```

### 依赖安装
```bash
pip install -r requirements.txt
```

## 8. 性能评估

### 评估指标
1. **准确性**: 模型预测与真实数据的吻合度
2. **效率**: 模型运行时间和资源消耗
3. **鲁棒性**: 模型对参数变化的敏感性

### 评估方法
- 交叉验证
- 敏感性分析
- 对比实验

## 9. 扩展建议

### 模型扩展
1. **增加车辆类型**: 考虑更多类型的交通参与者
2. **复杂网络**: 扩展到更复杂的交通网络
3. **动态需求**: 考虑时变的交通需求

### 算法改进
1. **深度学习**: 使用神经网络改进预测精度
2. **强化学习**: 优化交通控制策略
3. **多目标优化**: 考虑多个优化目标

## 10. 故障排除

### 常见问题
1. **内存不足**: 减少仿真规模或使用更高效的算法
2. **运行时间过长**: 优化代码或使用并行计算
3. **结果不准确**: 检查参数设置和数据质量

### 调试建议
1. **分步调试**: 逐个模块测试
2. **日志记录**: 添加详细的日志信息
3. **可视化**: 使用图表检查中间结果