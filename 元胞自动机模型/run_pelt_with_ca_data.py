import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from scipy import signal
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class SimplePELTDetector:
    """
    简化版PELT变点检测器
    """
    
    def __init__(self, min_distance=2, penalty=3.0, smooth_window=5):
        """
        初始化检测器
        
        参数:
        min_distance: 最小变点间距
        penalty: 惩罚参数（控制变点数量）
        smooth_window: 平滑窗口大小
        """
        self.min_distance = min_distance
        self.penalty = penalty
        self.smooth_window = smooth_window
        
        # 确保平滑窗口是奇数且大于多项式阶数
        if self.smooth_window % 2 == 0:
            self.smooth_window += 1
        if self.smooth_window < 5:  # 多项式阶数是3，所以窗口至少为5
            self.smooth_window = 5
    
    def detect_change_points_alternative(self, signal_data):
        """
        替代PELT算法：使用梯度变化检测变点
        """
        # 确保信号长度足够
        if len(signal_data) < 10:
            print(f"信号长度不足 ({len(signal_data)})，无法检测变点")
            return []
        
        # 平滑信号（Savitzky-Golay滤波器）
        window_size = min(self.smooth_window, len(signal_data))
        if window_size < 5:  # 确保窗口足够大
            window_size = min(5, len(signal_data))
        
        # 确保窗口是奇数
        if window_size % 2 == 0:
            window_size -= 1
        
        # 确保多项式阶数小于窗口大小
        polyorder = min(3, window_size - 1)
        
        try:
            smoothed = signal.savgol_filter(signal_data, window_size, polyorder)
        except:
            # 如果平滑失败，使用移动平均
            smoothed = np.convolve(signal_data, np.ones(window_size)/window_size, mode='same')
        
        # 计算一阶导数（梯度）
        gradient = np.gradient(smoothed)
        
        # 计算梯度绝对值的移动平均
        abs_gradient = np.abs(gradient)
        gradient_ma = np.convolve(abs_gradient, np.ones(window_size)/window_size, mode='same')
        
        # 检测变点（梯度变化显著的点）
        change_points = []
        threshold = np.mean(gradient_ma) + self.penalty * np.std(gradient_ma)
        
        for i in range(window_size, len(gradient_ma) - window_size):
            if gradient_ma[i] > threshold:
                # 检查是否与之前的变点太近
                if not change_points or (i - change_points[-1]) >= self.min_distance:
                    change_points.append(i)
        
        return change_points
    
    def detect_change_points_simple(self, signal_data):
        """
        简单变点检测：基于均值变化
        """
        n = len(signal_data)
        if n < 10:
            return []
        
        change_points = []
        min_segment = max(3, self.min_distance)
        
        # 使用滑动窗口检测均值变化
        for i in range(min_segment, n - min_segment):
            left_mean = np.mean(signal_data[:i])
            right_mean = np.mean(signal_data[i:])
            
            # 计算均值差异的统计显著性
            left_std = np.std(signal_data[:i]) if len(signal_data[:i]) > 1 else 0.1
            right_std = np.std(signal_data[i:]) if len(signal_data[i:]) > 1 else 0.1
            z_score = abs(left_mean - right_mean) / np.sqrt(left_std**2/len(signal_data[:i]) + right_std**2/len(signal_data[i:]))
            
            if z_score > self.penalty:
                # 检查是否与之前的变点太近
                if not change_points or (i - change_points[-1]) >= self.min_distance:
                    change_points.append(i)
        
        return change_points
    
    def detect_change_points_cusum(self, signal_data):
        """
        使用CUSUM算法检测变点
        """
        n = len(signal_data)
        if n < 10:
            return []
        
        # 计算累积和
        mean_val = np.mean(signal_data)
        residuals = signal_data - mean_val
        cusum = np.cumsum(residuals)
        
        # 标准化CUSUM
        cusum_norm = np.abs(cusum) / np.sqrt(np.arange(1, n+1) * np.var(signal_data) + 1e-10)
        
        # 检测变点
        change_points = []
        threshold = np.mean(cusum_norm) + self.penalty * np.std(cusum_norm)
        
        for i in range(self.min_distance, n - self.min_distance):
            if cusum_norm[i] > threshold:
                if not change_points or (i - change_points[-1]) >= self.min_distance:
                    change_points.append(i)
        
        return change_points
    
    def detect_all_change_points(self):
        """
        检测所有指标的变点
        """
        if not hasattr(self, 'data'):
            print("请先使用set_data()方法设置数据")
            return {}
        
        change_points = {}
        
        # 检测每个指标的变点
        for metric_name, signal_data in self.data.items():
            if metric_name == 'p':  # 跳过p值本身
                continue
            
            # 尝试三种检测方法
            try:
                cp1 = self.detect_change_points_alternative(signal_data)
            except:
                cp1 = []
            
            cp2 = self.detect_change_points_simple(signal_data)
            cp3 = self.detect_change_points_cusum(signal_data)
            
            # 合并结果（取并集）
            all_cps = sorted(set(cp1 + cp2 + cp3))
            
            # 过滤太接近的变点
            filtered_cps = []
            for cp in all_cps:
                if not filtered_cps or (cp - filtered_cps[-1]) >= self.min_distance:
                    filtered_cps.append(cp)
            
            change_points[metric_name] = filtered_cps
        
        return change_points
    
    def set_data(self, data_dict):
        """
        设置要检测的数据
        
        参数:
        data_dict: 字典，键是指标名称，值是数据序列
        """
        self.data = data_dict
        self.n_points = len(next(iter(data_dict.values())))
    
    def get_p_values_at_change_points(self, change_points_dict):
        """
        获取变点对应的p值
        """
        if not hasattr(self, 'data') or 'p' not in self.data:
            print("数据中没有p值序列")
            return {}
        
        p_series = self.data['p']
        p_change_points = {}
        
        for metric, cps in change_points_dict.items():
            p_cps = []
            for cp_idx in cps:
                if 0 <= cp_idx < len(p_series):
                    p_cps.append(p_series[cp_idx])
            p_change_points[metric] = p_cps
        
        return p_change_points


def load_simulation_json(filename):
    """
    加载仿真JSON文件
    """
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        p_values = data['p_values']
        results = data['results']
        
        print(f"成功加载 {filename}")
        print(f"数据包含 {len(p_values)} 个p值点")
        
        return p_values, results
    
    except Exception as e:
        print(f"加载JSON文件失败: {e}")
        return None, None


def extract_metrics_for_pelt(p_values, results):
    """
    从仿真结果中提取PELT需要的指标
    """
    # 按p值排序
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    
    # 初始化指标字典
    metrics = {
        'p': sorted_p,
        'speed': [],
        'flow': [],
        'congestion': [],
        'density': [],
        'travel_time': []
    }
    
    # 按排序后的顺序提取数据
    for idx in sorted_indices:
        result = results[idx]
        metrics['speed'].append(float(result['mean_speed']))
        metrics['flow'].append(float(result['mean_flow']))
        metrics['congestion'].append(float(result['congestion_index']))
        metrics['density'].append(float(result['mean_density']))
        
        # 计算旅行时间
        speed = float(result['mean_speed'])
        if speed > 0:
            travel_time = 10 / speed * 60  # 10km，转换为分钟
        else:
            travel_time = 999.0
        metrics['travel_time'].append(travel_time)
    
    # 转换为numpy数组
    for key in metrics:
        metrics[key] = np.array(metrics[key])
    
    return metrics


def analyze_change_points(p_values, results, visualize=True):
    """
    分析仿真结果中的变点
    """
    print("\n" + "="*60)
    print("开始PELT变点检测分析")
    print("="*60)
    
    # 提取指标
    print("\n提取性能指标...")
    metrics = extract_metrics_for_pelt(p_values, results)
    
    # 创建并配置检测器
    print("初始化变点检测器...")
    detector = SimplePELTDetector(
        min_distance=5,
        penalty=5.0,
        smooth_window=9
    )
    
    detector.set_data(metrics)
    
    # 检测变点
    print("检测性能指标变点...")
    change_points = detector.detect_all_change_points()
    
    # 获取对应的p值
    p_at_change_points = detector.get_p_values_at_change_points(change_points)
    
    # 打印结果
    print("\n" + "="*60)
    print("变点检测结果")
    print("="*60)
    
    summary = {}
    for metric, cp_indices in change_points.items():
        if metric == 'p':
            continue
            
        p_values_at_cp = p_at_change_points.get(metric, [])
        
        print(f"\n{metric} 指标:")
        if cp_indices:
            print(f"  检测到 {len(cp_indices)} 个变点")
            for i, (cp_idx, p_val) in enumerate(zip(cp_indices, p_values_at_cp)):
                print(f"  变点 {i+1}: 索引={cp_idx}, p={p_val:.3f}")
                
                # 保存到摘要
                if metric not in summary:
                    summary[metric] = []
                summary[metric].append({
                    'index': cp_idx,
                    'p_value': float(p_val),
                    'value_before': float(metrics[metric][cp_idx-1]) if cp_idx > 0 else None,
                    'value_after': float(metrics[metric][cp_idx]) if cp_idx < len(metrics[metric]) else None
                })
        else:
            print(f"  未检测到明显变点")
    
    # 计算综合临界点
    print("\n" + "="*60)
    print("综合临界点分析")
    print("="*60)
    
    all_p_values = []
    for metric, cps in summary.items():
        for cp in cps:
            all_p_values.append(cp['p_value'])
    
    if all_p_values:
        # 统计最常见的临界点
        from collections import Counter
        # 将p值四舍五入到2位小数进行统计
        rounded_p_values = [round(p, 2) for p in all_p_values]
        p_counter = Counter(rounded_p_values)
        
        print("\n临界点统计:")
        for p_val, count in p_counter.most_common():
            print(f"  p={p_val:.2f}: {count}个指标检测到变点")
        
        # 找到最重要的临界点（出现次数最多）
        if p_counter:
            most_common_p = p_counter.most_common(1)[0][0]
            print(f"\n最重要的临界点: p ≈ {most_common_p:.2f}")
            
            # 找出在这个临界点附近变化的指标
            print("在这个临界点附近发生变化的指标:")
            threshold = 0.05  # ±0.05的容忍度
            for metric, cps in summary.items():
                for cp in cps:
                    if abs(cp['p_value'] - most_common_p) <= threshold:
                        change_magnitude = abs(cp['value_after'] - cp['value_before']) if cp['value_before'] and cp['value_after'] else 0
                        change_percent = (change_magnitude / cp['value_before'] * 100) if cp['value_before'] and cp['value_before'] != 0 else 0
                        
                        direction = "增加" if cp['value_after'] > cp['value_before'] else "减少"
                        print(f"  {metric}: {direction} {change_percent:.1f}%")
    
    # 可视化
    if visualize:
        visualize_change_points(metrics, change_points, p_at_change_points)
    
    return detector, summary


def visualize_change_points(metrics, change_points, p_at_change_points, save_path='change_points_analysis.png'):
    """
    可视化变点检测结果
    """
    # 获取要绘制的指标列表（排除p值）
    metric_names = [k for k in metrics.keys() if k != 'p']
    n_metrics = len(metric_names)
    
    # 创建子图，每行最多3个
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    
    # 如果只有一个子图，确保axes是列表
    if n_metrics == 1:
        axes = np.array([axes])
    
    # 展平axes数组以便迭代
    axes_flat = axes.flatten() if n_metrics > 1 else axes
    
    p_series = metrics['p']
    
    for idx, metric in enumerate(metric_names):
        if idx >= len(axes_flat):
            break
            
        ax = axes_flat[idx]
        data = metrics[metric]
        
        # 绘制原始数据
        ax.plot(p_series, data, 'b-', linewidth=2, alpha=0.7, label=f'{metric}')
        
        # 标记变点
        if metric in change_points and change_points[metric]:
            cp_indices = change_points[metric]
            cp_p_values = p_at_change_points.get(metric, [])
            
            for cp_idx, p_val in zip(cp_indices, cp_p_values):
                if 0 <= cp_idx < len(data):
                    ax.axvline(x=p_val, color='red', linestyle='--', alpha=0.7, linewidth=1)
                    ax.plot(p_val, data[cp_idx], 'ro', markersize=6, markerfacecolor='none')
        
        ax.set_xlabel('自动驾驶比例 (p)', fontsize=10)
        
        # 根据指标类型设置合适的ylabel
        if metric == 'speed':
            ax.set_ylabel('速度 (mph)', fontsize=10)
        elif metric == 'flow':
            ax.set_ylabel('流量 (辆/小时)', fontsize=10)
        elif metric == 'congestion':
            ax.set_ylabel('拥堵指数', fontsize=10)
        elif metric == 'density':
            ax.set_ylabel('密度 (辆/km)', fontsize=10)
        elif metric == 'travel_time':
            ax.set_ylabel('旅行时间 (分钟)', fontsize=10)
        else:
            ax.set_ylabel(metric, fontsize=10)
        
        ax.set_title(f'{metric} 变点检测', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    
    # 隐藏多余的子图
    for idx in range(len(metric_names), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    plt.suptitle('CA模型PELT变点检测分析', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n变点检测图表已保存到: {save_path}")
    
    return fig

def analyze_change_points(p_values, results, visualize=True):
    """
    分析仿真结果中的变点
    """
    print("\n" + "="*60)
    print("开始PELT变点检测分析")
    print("="*60)
    
    # 提取指标
    print("\n提取性能指标...")
    metrics = extract_metrics_for_pelt(p_values, results)
    
    # 创建并配置检测器
    print("初始化变点检测器...")
    detector = SimplePELTDetector(
        min_distance=3,
        penalty=2.5,
        smooth_window=7
    )
    
    detector.set_data(metrics)
    
    # 检测变点
    print("检测性能指标变点...")
    change_points = detector.detect_all_change_points()
    
    # 获取对应的p值
    p_at_change_points = detector.get_p_values_at_change_points(change_points)
    
    # 打印结果
    print("\n" + "="*60)
    print("变点检测结果")
    print("="*60)
    
    summary = {}
    for metric, cp_indices in change_points.items():
        if metric == 'p':
            continue
            
        p_values_at_cp = p_at_change_points.get(metric, [])
        
        print(f"\n{metric} 指标:")
        if cp_indices:
            print(f"  检测到 {len(cp_indices)} 个变点")
            for i, (cp_idx, p_val) in enumerate(zip(cp_indices, p_values_at_cp)):
                print(f"  变点 {i+1}: 索引={cp_idx}, p={p_val:.3f}")
                
                # 保存到摘要
                if metric not in summary:
                    summary[metric] = []
                
                # 安全地获取前后值
                value_before = None
                value_after = None
                
                if 0 <= cp_idx < len(metrics[metric]):
                    value_after = float(metrics[metric][cp_idx])
                    if cp_idx > 0:
                        value_before = float(metrics[metric][cp_idx-1])
                    elif cp_idx < len(metrics[metric]) - 1:
                        value_before = float(metrics[metric][cp_idx+1])
                
                summary[metric].append({
                    'index': cp_idx,
                    'p_value': float(p_val),
                    'value_before': value_before,
                    'value_after': value_after
                })
        else:
            print(f"  未检测到明显变点")
    
    # 计算综合临界点
    print("\n" + "="*60)
    print("综合临界点分析")
    print("="*60)
    
    all_p_values = []
    for metric, cps in summary.items():
        for cp in cps:
            if cp['value_before'] is not None and cp['value_after'] is not None:
                all_p_values.append(cp['p_value'])
    
    if all_p_values:
        # 统计最常见的临界点
        from collections import Counter
        # 将p值四舍五入到2位小数进行统计
        rounded_p_values = [round(p, 2) for p in all_p_values]
        p_counter = Counter(rounded_p_values)
        
        print("\n临界点统计:")
        for p_val, count in p_counter.most_common():
            print(f"  p={p_val:.2f}: {count}个指标检测到变点")
        
        # 找到最重要的临界点（出现次数最多）
        if p_counter:
            most_common_p = p_counter.most_common(1)[0][0]
            print(f"\n最重要的临界点: p ≈ {most_common_p:.2f}")
            
            # 找出在这个临界点附近变化的指标
            print("在这个临界点附近发生变化的指标:")
            threshold = 0.05  # ±0.05的容忍度
            for metric, cps in summary.items():
                for cp in cps:
                    if abs(cp['p_value'] - most_common_p) <= threshold:
                        if cp['value_before'] is not None and cp['value_after'] is not None:
                            change_magnitude = abs(cp['value_after'] - cp['value_before'])
                            if cp['value_before'] != 0:
                                change_percent = (change_magnitude / cp['value_before'] * 100)
                            else:
                                change_percent = 0
                            
                            direction = "增加" if cp['value_after'] > cp['value_before'] else "减少"
                            print(f"  {metric}: {direction} {change_percent:.1f}%")
                        else:
                            print(f"  {metric}: 数据缺失")
    
    # 可视化
    if visualize:
        try:
            visualize_change_points(metrics, change_points, p_at_change_points)
        except Exception as e:
            print(f"\n可视化时出错: {e}")
            # 尝试简化版可视化
            try:
                visualize_simple(metrics, change_points, p_at_change_points)
            except:
                print("简化可视化也失败，跳过图表")
    
    return detector, summary


def main():
    """
    主函数：运行PELT变点检测
    """
    print("=" * 60)
    print("步骤3：使用CA仿真数据运行PELT变点检测")
    print("=" * 60)
    
    # 检查当前目录下的数据文件
    data_files = []
    for file in ['pelt_ready_data.json', 'simulation_results_optimized.json', 'simulation_results_optimized.csv']:
        if os.path.exists(file):
            data_files.append(file)
    
    if data_files:
        print(f"找到以下数据文件: {', '.join(data_files)}")
    else:
        print("未找到数据文件，请先运行数据提取模块")
        return
    
    print("\n数据来源选项:")
    print("1. 使用PELT专用格式数据 (pelt_ready_data.json)")
    print("2. 使用仿真结果JSON文件 (simulation_results_optimized.json)")
    print("3. 使用仿真结果CSV文件 (simulation_results_optimized.csv)")
    print("4. 手动输入数据文件路径")
    
    choice = input("\n请选择数据来源 (1-4): ").strip()
    
    if choice == '1':
        filename = 'pelt_ready_data.json'
    elif choice == '2':
        filename = 'simulation_results_optimized.json'
    elif choice == '3':
        filename = 'simulation_results_optimized.csv'
    elif choice == '4':
        filename = input("请输入数据文件路径: ").strip()
    else:
        print("无效选择，退出程序")
        return
    
    if not os.path.exists(filename):
        print(f"文件 {filename} 不存在")
        return
    
    # 根据文件类型加载数据
    if filename.endswith('.json'):
        if 'pelt_ready' in filename:
            # 加载PELT专用格式
            try:
                with open(filename, 'r') as f:
                    pelt_data = json.load(f)
                
                print(f"成功加载 {filename}")
                
                # 转换为统一格式
                p_values = pelt_data['p']
                # 创建虚拟results结构
                results = []
                for i, p in enumerate(p_values):
                    result = {
                        'p_value': p,
                        'mean_speed': pelt_data['speed'][i],
                        'mean_flow': pelt_data['flow'][i],
                        'congestion_index': pelt_data['congestion'][i],
                        'mean_density': 20.0,  # 默认值
                        'travel_time': pelt_data.get('travel_time', [10]*len(p_values))[i]
                    }
                    results.append(result)
                
            except Exception as e:
                print(f"加载PELT专用格式失败: {e}")
                return
        else:
            # 加载仿真JSON
            p_values, results = load_simulation_json(filename)
            if p_values is None:
                return
    
    elif filename.endswith('.csv'):
        # 加载CSV文件
        try:
            df = pd.read_csv(filename, encoding='utf-8-sig')
            print(f"成功加载 {filename}")
            print(f"数据包含 {len(df)} 行")
            
            # 转换为统一格式
            p_values = df['p_value'].tolist()
            results = []
            for _, row in df.iterrows():
                result = {
                    'p_value': row['p_value'],
                    'mean_speed': row['mean_speed'],
                    'mean_flow': row['mean_flow'],
                    'congestion_index': row['congestion_index'],
                    'mean_density': row.get('mean_density', 20.0),
                    'travel_time': 10 / row['mean_speed'] * 60 if row['mean_speed'] > 0 else 999.0
                }
                results.append(result)
        
        except Exception as e:
            print(f"加载CSV文件失败: {e}")
            return
    
    else:
        print("不支持的文件格式")
        return
    
    # 运行变点检测
    try:
        detector, summary = analyze_change_points(p_values, results, visualize=True)
        
        # 保存结果
        output_filename = 'pelt_analysis_results.json'
        with open(output_filename, 'w') as f:
            json.dump({
                'data_source': filename,
                'summary': summary,
                'analysis_time': pd.Timestamp.now().isoformat()
            }, f, indent=4, cls=NumpyEncoder)
        
        print(f"\n分析结果已保存到: {output_filename}")
        
        # 显示图表
        plt.show()
        
    except Exception as e:
        print(f"变点检测过程中出错: {e}")
        import traceback
        traceback.print_exc()


class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理numpy数据类型"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


if __name__ == "__main__":
    main()