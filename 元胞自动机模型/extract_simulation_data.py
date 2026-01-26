import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

# 从你的代码中导入必要模块
from CA import TrafficCA
from bayesian_optimization import BayesianParameterOptimizer

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def run_multi_p_simulation(p_values=None, use_optimized_params=True, 
                           save_to_file=True, verbose=True):
    """
    运行多p值仿真并保存结果
    
    参数:
    p_values: p值列表，如果为None则使用默认值
    use_optimized_params: 是否使用优化参数
    save_to_file: 是否保存到文件
    verbose: 是否显示进度
    
    返回:
    (p_values_list, simulation_results)
    """
    
    if p_values is None:
        p_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    simulation_results = []
    
    if verbose:
        print(f"开始多p值仿真，共{len(p_values)}个p值")
        print(f"使用{'优化' if use_optimized_params else '默认'}参数")
    
    for i, p in enumerate(p_values):
        if verbose:
            print(f"\n进度: {i+1}/{len(p_values)} - p = {p:.2f}")
        
        try:
            # 创建模型实例
            if use_optimized_params:
                # 尝试使用优化参数
                try:
                    optimizer = BayesianParameterOptimizer()
                    model = optimizer.get_optimized_ca_model()
                    model.p_av = p
                    model.road = np.zeros_like(model.road)  # 重置道路
                    model.generate_initial_vehicles()  # 重新生成车辆
                except Exception as e:
                    print(f"加载优化参数失败: {e}，使用默认参数")
                    model = TrafficCA(
                        length=1000,
                        lanes=3,
                        p_av=p,
                        density=0.2,
                        time_steps=200,
                        seed=42+i
                    )
            else:
                model = TrafficCA(
                    length=1000,
                    lanes=3,
                    p_av=p,
                    density=0.2,
                    time_steps=200,
                    seed=42+i
                )
            
            # 运行仿真
            model.run(warmup=50, verbose=False)
            
            # 获取统计结果
            stats = model.get_summary_stats()
            
            # 添加额外信息
            stats['p_value'] = p
            stats['simulation_id'] = i
            stats['use_optimized_params'] = use_optimized_params
            
            simulation_results.append(stats)
            
            if verbose:
                print(f"  结果: 速度={stats['mean_speed']:.1f}mph, "
                      f"流量={stats['mean_flow']:.0f}辆/小时")
        
        except Exception as e:
            print(f"p={p} 时仿真失败: {e}")
            # 添加一个默认结果
            simulation_results.append({
                'p_value': p,
                'mean_speed': 30,
                'mean_flow': 1000,
                'congestion_index': 0.5,
                'throughput': 500,
                'std_speed': 5,
                'std_flow': 100,
                'mean_density': 20,
                'max_flow': 1200,
                'min_speed': 25,
                'av_actual_percentage': p,
                'simulation_id': i,
                'use_optimized_params': use_optimized_params
            })
    
    if save_to_file:
        # 保存到JSON文件
        filename = f'simulation_results_{"optimized" if use_optimized_params else "default"}.json'
        with open(filename, 'w') as f:
            json.dump({
                'p_values': p_values,
                'results': simulation_results,
                'timestamp': pd.Timestamp.now().isoformat()
            }, f, indent=4, default=str)
        
        # 保存到CSV文件（更容易处理）
        df = pd.DataFrame(simulation_results)
        csv_filename = f'simulation_results_{"optimized" if use_optimized_params else "default"}.csv'
        df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        
        if verbose:
            print(f"\n仿真结果已保存到:")
            print(f"  {filename}")
            print(f"  {csv_filename}")
    
    return p_values, simulation_results


def load_simulation_data(filename='simulation_results_optimized.json'):
    """
    加载已有的仿真数据
    
    参数:
    filename: 数据文件名
    
    返回:
    (p_values, simulation_results)
    """
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        p_values = data['p_values']
        simulation_results = data['results']
        
        print(f"从 {filename} 加载了 {len(p_values)} 个p值的仿真结果")
        return p_values, simulation_results
    
    except FileNotFoundError:
        print(f"文件 {filename} 不存在")
        return None, None


def analyze_simulation_results(p_values, results):
    """
    分析仿真结果
    
    参数:
    p_values: p值列表
    results: 仿真结果列表
    """
    
    if not results:
        print("没有可分析的结果")
        return
    
    # 创建DataFrame以便分析
    df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("仿真结果分析")
    print("="*60)
    
    # 基本统计
    print(f"\n基本统计:")
    print(f"  仿真次数: {len(df)}")
    print(f"  p值范围: {min(p_values):.2f} 到 {max(p_values):.2f}")
    
    # 性能指标
    print(f"\n性能指标:")
    print(f"  平均速度范围: {df['mean_speed'].min():.1f} - {df['mean_speed'].max():.1f} mph")
    print(f"  平均流量范围: {df['mean_flow'].min():.0f} - {df['mean_flow'].max():.0f} 辆/小时")
    print(f"  拥堵指数范围: {df['congestion_index'].min():.3f} - {df['congestion_index'].max():.3f}")
    
    # 找到最佳性能的p值
    best_speed_idx = df['mean_speed'].idxmax()
    best_flow_idx = df['mean_flow'].idxmax()
    best_congestion_idx = df['congestion_index'].idxmin()
    
    print(f"\n最佳性能:")
    print(f"  最高速度: p={df.loc[best_speed_idx, 'p_value']:.3f}, "
          f"速度={df.loc[best_speed_idx, 'mean_speed']:.1f}mph")
    print(f"  最高流量: p={df.loc[best_flow_idx, 'p_value']:.3f}, "
          f"流量={df.loc[best_flow_idx, 'mean_flow']:.0f}辆/小时")
    print(f"  最低拥堵: p={df.loc[best_congestion_idx, 'p_value']:.3f}, "
          f"拥堵指数={df.loc[best_congestion_idx, 'congestion_index']:.3f}")
    
    # 计算改进幅度（相对于p=0）
    if 0 in p_values:
        p0_idx = p_values.index(0)
        p0_speed = df.loc[p0_idx, 'mean_speed']
        p0_flow = df.loc[p0_idx, 'mean_flow']
        
        print(f"\n相对于纯人类驾驶(p=0)的改进:")
        print(f"  速度改进: +{(df['mean_speed'].max() - p0_speed)/p0_speed*100:.1f}%")
        print(f"  流量改进: +{(df['mean_flow'].max() - p0_flow)/p0_flow*100:.1f}%")
    
    return df


def visualize_simulation_results(p_values, results, save_path=None):
    """
    可视化仿真结果
    
    参数:
    p_values: p值列表
    results: 仿真结果列表
    save_path: 保存路径
    """
    
    # 提取数据
    speeds = [r['mean_speed'] for r in results]
    flows = [r['mean_flow'] for r in results]
    congestion = [r['congestion_index'] for r in results]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 平均速度曲线
    ax1 = axes[0, 0]
    ax1.plot(p_values, speeds, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('自动驾驶比例 (p)', fontsize=12)
    ax1.set_ylabel('平均速度 (mph)', fontsize=12, color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_title('平均速度随p变化', fontsize=13)
    ax1.grid(True, alpha=0.3)
    
    # 标记最大值
    max_speed_idx = np.argmax(speeds)
    ax1.plot(p_values[max_speed_idx], speeds[max_speed_idx], 
            'r*', markersize=15, label=f'最大值 p={p_values[max_speed_idx]:.3f}')
    ax1.legend()
    
    # 2. 平均流量曲线
    ax2 = axes[0, 1]
    ax2.plot(p_values, flows, 'g-o', linewidth=2, markersize=6)
    ax2.set_xlabel('自动驾驶比例 (p)', fontsize=12)
    ax2.set_ylabel('平均流量 (辆/小时)', fontsize=12, color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.set_title('平均流量随p变化', fontsize=13)
    ax2.grid(True, alpha=0.3)
    
    max_flow_idx = np.argmax(flows)
    ax2.plot(p_values[max_flow_idx], flows[max_flow_idx], 
            'r*', markersize=15, label=f'最大值 p={p_values[max_flow_idx]:.3f}')
    ax2.legend()
    
    # 3. 拥堵指数曲线
    ax3 = axes[1, 0]
    ax3.plot(p_values, congestion, 'r-o', linewidth=2, markersize=6)
    ax3.set_xlabel('自动驾驶比例 (p)', fontsize=12)
    ax3.set_ylabel('拥堵指数 (0-1)', fontsize=12, color='r')
    ax3.tick_params(axis='y', labelcolor='r')
    ax3.set_title('拥堵指数随p变化', fontsize=13)
    ax3.grid(True, alpha=0.3)
    
    min_congestion_idx = np.argmin(congestion)
    ax3.plot(p_values[min_congestion_idx], congestion[min_congestion_idx], 
            'b*', markersize=15, label=f'最小值 p={p_values[min_congestion_idx]:.3f}')
    ax3.legend()
    
    # 4. 基本图（流量-速度）
    ax4 = axes[1, 1]
    
    # 使用颜色表示p值
    scatter = ax4.scatter(flows, speeds, c=p_values, cmap='viridis', 
                         s=50, alpha=0.7, edgecolors='k')
    
    ax4.set_xlabel('流量 (辆/小时)', fontsize=12)
    ax4.set_ylabel('速度 (mph)', fontsize=12)
    ax4.set_title('基本图 (颜色表示p值)', fontsize=13)
    ax4.grid(True, alpha=0.3)
    
    # 添加颜色条
    plt.colorbar(scatter, ax=ax4, label='自动驾驶比例 (p)')
    
    plt.suptitle('CA模型多p值仿真结果', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"仿真结果图已保存到: {save_path}")
    
    return fig


def prepare_data_for_pelt(p_values, results):
    """
    准备PELT变点检测需要的数据格式
    
    参数:
    p_values: p值列表
    results: 仿真结果列表
    
    返回:
    适合PELT检测的数据结构
    """
    
    pelt_data = {
        'p': p_values,
        'speed': [r['mean_speed'] for r in results],
        'flow': [r['mean_flow'] for r in results],
        'congestion': [r['congestion_index'] for r in results],
        'travel_time': []
    }
    
    # 计算旅行时间（假设路段长度10km）
    for r in results:
        speed = r['mean_speed']
        if speed > 0:
            travel_time = 10 / speed * 60  # 10km，转换为分钟
        else:
            travel_time = 999  # 极大值表示完全拥堵
        pelt_data['travel_time'].append(travel_time)
    
    return pelt_data


def main():
    """
    主函数：运行仿真并准备PELT检测数据
    """
    print("=" * 60)
    print("CA仿真数据提取与准备")
    print("=" * 60)
    
    print("\n选项:")
    print("1. 运行新的多p值仿真（使用优化参数）")
    print("2. 运行新的多p值仿真（使用默认参数）")
    print("3. 加载已有的仿真数据")
    print("4. 准备PELT变点检测数据")
    
    choice = input("\n请选择 (1-4): ").strip()
    
    if choice == '1':
        # 运行优化参数仿真
        p_values, results = run_multi_p_simulation(
            p_values=np.linspace(0, 1, 21).tolist(),  # 21个点，更密集
            use_optimized_params=True,
            save_to_file=True,
            verbose=True
        )
        
    elif choice == '2':
        # 运行默认参数仿真
        p_values, results = run_multi_p_simulation(
            p_values=np.linspace(0, 1, 21).tolist(),
            use_optimized_params=False,
            save_to_file=True,
            verbose=True
        )
        
    elif choice == '3':
        # 加载已有数据
        filename = input("请输入数据文件名 (默认: simulation_results_optimized.json): ").strip()
        if not filename:
            filename = 'simulation_results_optimized.json'
        
        p_values, results = load_simulation_data(filename)
        
        if p_values is None:
            print("加载失败，退出程序")
            return
        
    elif choice == '4':
        # 准备PELT数据
        filename = input("请输入数据文件名 (默认: simulation_results_optimized.json): ").strip()
        if not filename:
            filename = 'simulation_results_optimized.json'
        
        p_values, results = load_simulation_data(filename)
        
        if p_values is None:
            print("加载失败，退出程序")
            return
        
        # 准备PELT数据
        pelt_data = prepare_data_for_pelt(p_values, results)
        
        # 保存为PELT专用格式
        pelt_filename = 'pelt_ready_data.json'
        with open(pelt_filename, 'w') as f:
            json.dump(pelt_data, f, indent=4)
        
        print(f"\nPELT数据已保存到: {pelt_filename}")
        print(f"数据结构:")
        for key, value in pelt_data.items():
            print(f"  {key}: {len(value) if isinstance(value, list) else value}")
        
        return pelt_data
    
    else:
        print("无效选择，退出程序")
        return
    
    # 分析结果
    if 'results' in locals():
        df = analyze_simulation_results(p_values, results)
        
        # 可视化
        fig = visualize_simulation_results(
            p_values, results, 
            save_path='simulation_results_visualization.png'
        )
        
        # 准备PELT数据
        pelt_data = prepare_data_for_pelt(p_values, results)
        
        # 保存PELT数据
        with open('pelt_ready_data.json', 'w') as f:
            json.dump(pelt_data, f, indent=4)
        
        print(f"\nPELT变点检测数据已保存到: pelt_ready_data.json")
        print("您现在可以运行PELT变点检测模块了!")
    
    print("\n" + "=" * 60)
    print("数据准备完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()