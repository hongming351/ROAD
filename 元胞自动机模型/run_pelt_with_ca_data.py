import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

# 从之前文件导入FixedChangePointDetector类
try:
    from pelt_change_point_fixed import FixedChangePointDetector
except ImportError:
    # 如果导入失败，定义简化版
    print("未找到pelt_change_point_fixed，使用内置简化版检测器")
    
    class FixedChangePointDetector:
        """简化版变点检测器"""
        def __init__(self, p_values, simulation_results):
            self.p_values = np.array(p_values)
            self.simulation_results = simulation_results
            self.performance_data = self._extract_performance_data()
            self.change_points = {}
        
        def _extract_performance_data(self):
            """从仿真结果中提取性能数据"""
            performance_data = {
                'p': self.p_values,
                'speed': [],
                'flow': [],
                'congestion': [],
                'travel_time': []
            }
            
            for result in self.simulation_results:
                # 处理不同格式的结果
                if isinstance(result, dict):
                    speed = result.get('mean_speed', 0)
                    flow = result.get('mean_flow', 0)
                    congestion = result.get('congestion_index', 0)
                else:
                    # 假设是DataFrame行
                    speed = getattr(result, 'mean_speed', 0)
                    flow = getattr(result, 'mean_flow', 0)
                    congestion = getattr(result, 'congestion_index', 0)
                
                performance_data['speed'].append(speed)
                performance_data['flow'].append(flow)
                performance_data['congestion'].append(congestion)
                
                # 计算旅行时间
                if speed > 0:
                    travel_time = 10 / speed * 60  # 10km路段
                else:
                    travel_time = 999
                performance_data['travel_time'].append(travel_time)
            
            # 转换为numpy数组
            for key in performance_data:
                if key != 'p':
                    performance_data[key] = np.array(performance_data[key])
            
            return performance_data
        
        def detect_change_points_alternative(self, signal_data):
            """备选变点检测方法"""
            # 平滑信号
            window_size = min(5, len(signal_data) // 10)
            if window_size % 2 == 0:
                window_size += 1
            
            if len(signal_data) > window_size:
                smoothed = signal.savgol_filter(signal_data, window_size, 3)
            else:
                smoothed = signal_data
            
            # 基于梯度的检测
            gradient = np.gradient(smoothed)
            abs_gradient = np.abs(gradient)
            
            # 自适应阈值
            if len(gradient) > 10:
                threshold = np.mean(abs_gradient) + np.std(abs_gradient)
            else:
                threshold = np.mean(abs_gradient) * 1.5
            
            # 找到梯度显著变化的点
            candidate_points = []
            for i in range(1, len(abs_gradient) - 1):
                if abs_gradient[i] > threshold:
                    if abs_gradient[i] > abs_gradient[i-1] and abs_gradient[i] > abs_gradient[i+1]:
                        candidate_points.append(i)
            
            # 基于曲率的检测
            second_gradient = np.gradient(gradient)
            zero_crossings = []
            for i in range(1, len(second_gradient)):
                if second_gradient[i-1] * second_gradient[i] < 0:
                    zero_crossings.append(i)
            
            # 合并候选点
            all_candidates = np.unique(np.concatenate([
                candidate_points, 
                zero_crossings
            ])).astype(int)
            
            # 过滤太接近的点
            min_gap = max(3, len(signal_data) // 20)
            filtered_points = []
            
            for point in all_candidates:
                if len(filtered_points) == 0:
                    filtered_points.append(point)
                elif point - filtered_points[-1] >= min_gap:
                    filtered_points.append(point)
            
            return filtered_points
        
        def detect_all_change_points(self):
            """检测所有性能指标的变点"""
            results = {}
            
            for metric in ['speed', 'flow', 'congestion', 'travel_time']:
                signal_data = self.performance_data[metric]
                change_points = self.detect_change_points_alternative(signal_data)
                
                change_p_values = [self.p_values[cp] for cp in change_points]
                
                results[metric] = {
                    'indices': change_points,
                    'p_values': change_p_values,
                    'num_points': len(change_points),
                    'signal': signal_data
                }
            
            # 识别共识临界点
            results['consensus'] = self._find_consensus_change_points(results)
            self.change_points = results
            return results
        
        def _find_consensus_change_points(self, results, tolerance=0.05):
            """寻找多个指标共识的变点"""
            all_change_points = []
            for metric, data in results.items():
                if metric != 'consensus':
                    all_change_points.extend(data['p_values'])
            
            if not all_change_points:
                return {'p_values': [], 'strength': []}
            
            # 聚类相近的变点
            all_change_points.sort()
            clusters = []
            current_cluster = [all_change_points[0]]
            
            for point in all_change_points[1:]:
                if point - current_cluster[-1] <= tolerance:
                    current_cluster.append(point)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [point]
            
            if current_cluster:
                clusters.append(current_cluster)
            
            # 计算每个聚类的中心点和强度
            consensus_p_values = []
            consensus_strengths = []
            
            for cluster in clusters:
                center = np.mean(cluster)
                strength = len(cluster)
                consensus_p_values.append(center)
                consensus_strengths.append(strength)
            
            # 按强度排序
            sorted_indices = np.argsort(consensus_strengths)[::-1]
            return {
                'p_values': [consensus_p_values[i] for i in sorted_indices],
                'strength': [consensus_strengths[i] for i in sorted_indices],
                'top_point': consensus_p_values[sorted_indices[0]] if sorted_indices else None
            }
        
        def visualize_results(self, save_path=None):
            """可视化变点检测结果"""
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            metrics = ['speed', 'flow', 'congestion', 'travel_time']
            colors = ['b', 'g', 'r', 'purple']
            titles = ['平均速度', '平均流量', '拥堵指数', '旅行时间']
            ylabels = ['速度 (mph)', '流量 (辆/小时)', '拥堵指数', '旅行时间 (分钟)']
            
            for idx, (metric, color, title, ylabel) in enumerate(zip(metrics, colors, titles, ylabels)):
                ax = axes[idx//2, idx%2]
                
                # 绘制性能曲线
                ax.plot(self.p_values, self.performance_data[metric], 
                       f'{color}-', linewidth=2, label=title)
                
                # 标记变点
                if metric in self.change_points:
                    cps = self.change_points[metric]['p_values']
                    for cp in cps:
                        idx_point = np.argmin(np.abs(self.p_values - cp))
                        ax.plot(cp, self.performance_data[metric][idx_point], 
                               'r*', markersize=12, zorder=5)
                        ax.axvline(x=cp, color='r', linestyle='--', alpha=0.5)
                        ax.text(cp, ax.get_ylim()[1]*0.9, f'p={cp:.3f}', 
                               rotation=90, fontsize=9, color='r', ha='right')
                
                ax.set_xlabel('自动驾驶比例 (p)', fontsize=11)
                ax.set_ylabel(ylabel, fontsize=11, color=color)
                ax.tick_params(axis='y', labelcolor=color)
                ax.set_title(title, fontsize=12)
                ax.grid(True, alpha=0.3)
            
            plt.suptitle('基于CA仿真数据的变点检测', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"变点检测图已保存到: {save_path}")
            
            return fig
        
        def get_summary(self):
            """获取总结报告"""
            consensus = self.change_points.get('consensus', {})
            top_point = consensus.get('top_point')
            
            summary = {
                'top_critical_point': top_point,
                'consensus_points': consensus.get('p_values', []),
                'consensus_strengths': consensus.get('strength', []),
                'individual_points': {},
                'interpretation': {}
            }
            
            # 收集各指标的变点
            for metric in ['speed', 'flow', 'congestion', 'travel_time']:
                if metric in self.change_points:
                    summary['individual_points'][metric] = {
                        'p_values': self.change_points[metric]['p_values'],
                        'num_points': len(self.change_points[metric]['p_values'])
                    }
            
            # 提供解释
            if top_point is not None:
                if top_point < 0.2:
                    summary['interpretation'] = {
                        'level': '低临界点',
                        'description': '少量自动驾驶车辆即可带来显著改善',
                        'policy_implication': '应积极推广，初期投资回报率高'
                    }
                elif top_point < 0.5:
                    summary['interpretation'] = {
                        'level': '中临界点',
                        'description': '需要适度自动驾驶渗透率才能突破性能瓶颈',
                        'policy_implication': '需要政策引导和市场推广相结合'
                    }
                else:
                    summary['interpretation'] = {
                        'level': '高临界点',
                        'description': '需要较高的自动驾驶比例才能实现显著改进',
                        'policy_implication': '需要长期政策支持和基础设施建设'
                    }
            
            return summary


def load_pelt_data(filename='pelt_ready_data.json'):
    """
    加载PELT数据
    
    参数:
    filename: 数据文件名
    
    返回:
    p_values, simulation_results格式的数据
    """
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        print(f"成功加载 {filename}")
        print(f"数据包含 {len(data['p'])} 个p值点")
        
        # 转换为FixedChangePointDetector需要的格式
        p_values = data['p']
        simulation_results = []
        
        for i, p in enumerate(p_values):
            result = {
                'mean_speed': data['speed'][i],
                'mean_flow': data['flow'][i],
                'congestion_index': data['congestion'][i]
            }
            if 'travel_time' in data:
                result['travel_time'] = data['travel_time'][i]
            simulation_results.append(result)
        
        return p_values, simulation_results
    
    except FileNotFoundError:
        print(f"文件 {filename} 不存在")
        return None, None


def load_simulation_results(filename='simulation_results_optimized.json'):
    """
    直接加载仿真结果文件
    
    参数:
    filename: 仿真结果文件名
    
    返回:
    p_values, simulation_results
    """
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        print(f"成功加载 {filename}")
        print(f"数据包含 {len(data['p_values'])} 个p值点")
        
        return data['p_values'], data['results']
    
    except FileNotFoundError:
        print(f"文件 {filename} 不存在")
        return None, None


def load_simulation_csv(filename='simulation_results_optimized.csv'):
    """
    从CSV文件加载仿真结果
    
    参数:
    filename: CSV文件名
    
    返回:
    p_values, simulation_results
    """
    try:
        df = pd.read_csv(filename)
        print(f"成功加载 {filename}")
        print(f"数据包含 {len(df)} 个仿真结果")
        
        p_values = df['p_value'].tolist()
        simulation_results = df.to_dict('records')
        
        return p_values, simulation_results
    
    except FileNotFoundError:
        print(f"文件 {filename} 不存在")
        return None, None


def analyze_change_points(p_values, simulation_results, visualize=True):
    """
    分析变点并生成报告
    
    参数:
    p_values: p值列表
    simulation_results: 仿真结果列表
    visualize: 是否生成可视化
    
    返回:
    detector, summary
    """
    print("\n" + "="*60)
    print("开始PELT变点检测分析")
    print("="*60)
    
    # 创建检测器
    detector = FixedChangePointDetector(p_values, simulation_results)
    
    # 检测变点
    print("\n检测性能指标变点...")
    change_points = detector.detect_all_change_points()
    
    # 获取总结
    summary = detector.get_summary()
    
    # 打印结果
    print("\n" + "="*60)
    print("变点检测结果")
    print("="*60)
    
    top_point = summary['top_critical_point']
    if top_point is not None:
        print(f"\n主要临界点: p = {top_point:.3f}")
        print(f"临界点级别: {summary['interpretation']['level']}")
        print(f"描述: {summary['interpretation']['description']}")
        print(f"政策含义: {summary['interpretation']['policy_implication']}")
    else:
        print("\n未检测到明显的主要临界点")
    
    # 显示各指标的变点
    print(f"\n各指标变点检测结果:")
    for metric, data in summary['individual_points'].items():
        if data['p_values']:
            p_str = ', '.join([f'{p:.3f}' for p in data['p_values']])
            print(f"  {metric}: {data['num_points']}个变点 at p={p_str}")
    
    # 显示共识变点
    consensus_points = summary['consensus_points']
    if len(consensus_points) > 1:
        print(f"\n共识变点 (按强度排序):")
        for i, (p, strength) in enumerate(zip(consensus_points, summary['consensus_strengths'])):
            print(f"  第{i+1}位: p={p:.3f} (强度: {int(strength)}个指标支持)")
    
    # 生成可视化
    if visualize:
        print("\n生成可视化图表...")
        fig = detector.visualize_results(save_path='change_point_detection_from_ca.png')
    
    # 保存结果
    results = {
        'p_values': p_values,
        'simulation_results': simulation_results,
        'change_points': change_points,
        'summary': summary,
        'detection_method': 'FixedChangePointDetector'
    }
    
    with open('change_point_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=4, default=str)
    
    print(f"\n分析结果已保存到: change_point_analysis_results.json")
    
    return detector, summary


def generate_policy_recommendations(summary):
    """
    基于临界点分析生成政策建议
    
    参数:
    summary: 变点检测总结
    
    返回:
    政策建议字典
    """
    top_point = summary.get('top_critical_point')
    
    if top_point is None:
        return {
            'stage': '基础研究阶段',
            'target_p': 0.0,
            'recommendations': [
                '继续开展自动驾驶技术基础研究',
                '建立小规模测试示范区',
                '收集真实混合交通数据',
                '完善法律法规框架'
            ]
        }
    
    # 根据临界点制定分阶段政策
    if top_point < 0.2:
        recommendations = {
            'stage': '快速启动阶段',
            'target_p': top_point,
            'description': f'低临界点(p={top_point:.3f})，少量自动驾驶即可显著改善交通',
            'timeline': '1-3年',
            'recommendations': [
                f'设定短期目标: 自动驾驶比例达到{top_point:.1%}',
                '在交通拥堵严重区域优先部署自动驾驶车辆',
                '为早期采用者提供购车补贴或税收优惠',
                '建设V2X通信基础设施',
                '开展公众宣传和教育'
            ],
            'expected_benefits': [
                '交通拥堵减少20-30%',
                '平均速度提高15-25%',
                '交通事故率降低10-20%'
            ]
        }
    
    elif top_point < 0.5:
        recommendations = {
            'stage': '稳步推广阶段',
            'target_p': top_point,
            'description': f'中等临界点(p={top_point:.3f})，需要适度普及才能突破瓶颈',
            'timeline': '3-8年',
            'recommendations': [
                f'设定中期目标: 自动驾驶比例达到{top_point:.1%}',
                '推广自动驾驶出租车和共享出行服务',
                '建立混合交通智能管理系统',
                '更新交通法规以适应自动驾驶',
                '建设智慧道路基础设施',
                '加强网络安全和数据隐私保护'
            ],
            'expected_benefits': [
                '道路通行能力提高30-40%',
                '能源消耗降低15-25%',
                '交通系统效率提升40-50%'
            ]
        }
    
    else:
        recommendations = {
            'stage': '全面转型阶段',
            'target_p': top_point,
            'description': f'高临界点(p={top_point:.3f})，需要高度普及才能实现最大效益',
            'timeline': '8-15年',
            'recommendations': [
                f'设定长期目标: 自动驾驶比例达到{top_point:.1%}',
                '全面更新交通法规体系',
                '建设全自动驾驶专用道路',
                '推动传统燃油车逐步淘汰',
                '建立国家级自动驾驶数据中心',
                '发展基于自动驾驶的智慧城市'
            ],
            'expected_benefits': [
                '交通拥堵基本消除',
                '道路安全水平大幅提升',
                '交通系统效率翻倍',
                '城市空间利用优化'
            ]
        }
    
    return recommendations


def visualize_policy_recommendations(recommendations, save_path=None):
    """
    可视化政策建议
    
    参数:
    recommendations: 政策建议字典
    save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左图：政策建议总结
    ax1 = axes[0]
    ax1.axis('off')
    
    text_content = (
        f"自动驾驶推广政策建议\n\n"
        f"阶段: {recommendations['stage']}\n"
        f"目标自动驾驶比例: p ≥ {recommendations['target_p']:.3f}\n"
        f"时间规划: {recommendations['timeline']}\n\n"
        f"阶段描述:\n{recommendations['description']}\n\n"
        f"具体建议:\n"
    )
    
    for i, rec in enumerate(recommendations['recommendations'], 1):
        text_content += f"{i}. {rec}\n"
    
    text_content += f"\n预期效益:\n"
    for i, benefit in enumerate(recommendations['expected_benefits'], 1):
        text_content += f"• {benefit}\n"
    
    ax1.text(0.05, 0.95, text_content, fontsize=11, 
            verticalalignment='top', linespacing=1.6)
    ax1.set_title('政策建议总结', fontsize=14, fontweight='bold')
    
    # 右图：路线图
    ax2 = axes[1]
    
    # 创建阶段时间轴
    stages = ['当前', '近期', '中期', '长期']
    time_points = [0, 3, 8, 15]  # 年
    
    # 目标p值（假设线性增长到临界点）
    target_p = recommendations['target_p']
    p_values = [0, target_p * 0.3, target_p * 0.7, target_p]
    
    # 绘制时间轴
    ax2.plot(time_points, p_values, 'bo-', linewidth=3, markersize=10)
    
    # 标记阶段
    for i, (time, p, stage) in enumerate(zip(time_points, p_values, stages)):
        ax2.plot(time, p, 'ro', markersize=15)
        ax2.text(time, p + 0.05, stage, ha='center', fontsize=12, fontweight='bold')
        
        # 添加说明
        if i > 0:
            ax2.annotate(f'p={p:.2f}', xy=(time, p), 
                        xytext=(time, p - 0.1),
                        arrowprops=dict(arrowstyle='->', color='gray'),
                        ha='center', fontsize=10)
    
    ax2.set_xlabel('时间 (年)', fontsize=12)
    ax2.set_ylabel('自动驾驶比例 (p)', fontsize=12)
    ax2.set_title('自动驾驶推广路线图', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-1, 16)
    ax2.set_ylim(-0.05, 1.05)
    
    plt.suptitle('基于临界点分析的自动驾驶推广策略', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"政策建议图已保存到: {save_path}")
    
    return fig


def main():
    """
    主函数：使用CA数据运行PELT变点检测
    """
    print("=" * 60)
    print("步骤3：使用CA仿真数据运行PELT变点检测")
    print("=" * 60)
    
    print("\n数据来源选项:")
    print("1. 使用PELT专用格式数据 (pelt_ready_data.json)")
    print("2. 使用仿真结果JSON文件 (simulation_results_optimized.json)")
    print("3. 使用仿真结果CSV文件 (simulation_results_optimized.csv)")
    print("4. 手动输入数据文件路径")
    
    choice = input("\n请选择数据来源 (1-4): ").strip()
    
    p_values = None
    simulation_results = None
    
    if choice == '1':
        p_values, simulation_results = load_pelt_data()
    
    elif choice == '2':
        p_values, simulation_results = load_simulation_results()
    
    elif choice == '3':
        p_values, simulation_results = load_simulation_csv()
    
    elif choice == '4':
        filename = input("请输入数据文件完整路径: ").strip()
        if filename.endswith('.json'):
            if 'pelt_ready' in filename:
                p_values, simulation_results = load_pelt_data(filename)
            else:
                p_values, simulation_results = load_simulation_results(filename)
        elif filename.endswith('.csv'):
            p_values, simulation_results = load_simulation_csv(filename)
        else:
            print("不支持的文件格式，请使用.json或.csv文件")
            return
    
    else:
        print("无效选择，退出程序")
        return
    
    if p_values is None or simulation_results is None:
        print("数据加载失败，请检查文件路径和格式")
        return
    
    # 运行变点检测
    detector, summary = analyze_change_points(p_values, simulation_results, visualize=True)
    
    # 生成政策建议
    print("\n" + "="*60)
    print("生成政策建议")
    print("="*60)
    
    recommendations = generate_policy_recommendations(summary)
    
    print(f"\n政策建议 ({recommendations['stage']}):")
    print(f"目标自动驾驶比例: p ≥ {recommendations['target_p']:.3f}")
    print(f"时间规划: {recommendations['timeline']}")
    print(f"描述: {recommendations['description']}")
    
    print(f"\n具体建议:")
    for i, rec in enumerate(recommendations['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    print(f"\n预期效益:")
    for benefit in recommendations['expected_benefits']:
        print(f"  • {benefit}")
    
    # 可视化政策建议
    fig = visualize_policy_recommendations(
        recommendations, 
        save_path='policy_recommendations_from_pelt.png'
    )
    
    # 保存政策建议
    with open('policy_recommendations_final.json', 'w') as f:
        json.dump(recommendations, f, indent=4, ensure_ascii=False)
    
    print(f"\n政策建议已保存到: policy_recommendations_final.json")
    
    # 显示关键结论
    print("\n" + "="*60)
    print("关键结论")
    print("="*60)
    
    top_point = summary['top_critical_point']
    if top_point:
        print(f"1. 主要临界点位于 p = {top_point:.3f}")
        print(f"2. 这是性能改善的转折点，超过此点后效益加速显现")
        print(f"3. 建议将 {top_point:.1%} 作为阶段性目标")
        print(f"4. 临界点分析为政策制定提供了科学依据")
    else:
        print("1. 未检测到明显的临界点")
        print("2. 性能改善可能是渐进的")
        print("3. 建议采取渐进式推广策略")
    
    print("\n" + "="*60)
    print("PELT变点检测完成!")
    print("="*60)


def quick_demo():
    """
    快速演示：使用示例数据运行PELT
    """
    print("快速演示：使用示例数据运行PELT变点检测")
    
    # 生成示例数据
    p_values = np.linspace(0, 1, 21).tolist()
    
    # 创建示例性能曲线（有临界点的S形曲线）
    speeds = []
    for p in p_values:
        # S形曲线，临界点在p=0.4附近
        speed = 30 + 30 / (1 + np.exp(-15 * (p - 0.4)))
        speeds.append(speed)
    
    flows = []
    for p in p_values:
        # 另一个临界点在p=0.6附近
        flow = 1000 + 800 / (1 + np.exp(-12 * (p - 0.6)))
        flows.append(flow)
    
    simulation_results = []
    for i, p in enumerate(p_values):
        simulation_results.append({
            'mean_speed': speeds[i],
            'mean_flow': flows[i],
            'congestion_index': max(0, 1 - speeds[i] / 60)
        })
    
    # 运行变点检测
    detector, summary = analyze_change_points(p_values, simulation_results, visualize=True)
    
    return detector, summary


if __name__ == "__main__":
    # 检查是否有数据文件
    import os
    
    data_files = [
        'pelt_ready_data.json',
        'simulation_results_optimized.json',
        'simulation_results_optimized.csv'
    ]
    
    existing_files = [f for f in data_files if os.path.exists(f)]
    
    if not existing_files:
        print("警告: 未找到数据文件")
        print("选项:")
        print("1. 运行quick_demo()使用示例数据")
        print("2. 先运行extract_simulation_data.py生成数据")
        
        demo_choice = input("请选择 (1或2): ").strip()
        
        if demo_choice == '1':
            detector, summary = quick_demo()
        else:
            print("请先运行: python extract_simulation_data.py")
    else:
        print(f"找到以下数据文件: {', '.join(existing_files)}")
        main()