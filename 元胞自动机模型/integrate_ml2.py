import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bayesian_optimization import BayesianParameterOptimizer
from CA import TrafficCA, TrafficVisualizer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class EnhancedTrafficCA(TrafficCA):
    """
    增强的交通CA模型，使用贝叶斯优化后的参数
    继承原有CA模型，但使用优化后的参数
    """
    
    def __init__(self, length=1000, lanes=3, p_av=0.5, density=0.15, 
                 time_steps=1000, seed=42, use_optimized_params=True):
        """
        初始化增强的CA模型
        
        参数:
        use_optimized_params: 是否使用优化后的参数
        """
        # 调用父类初始化
        super().__init__(length, lanes, p_av, density, time_steps, seed)
        
        # 如果使用优化参数，则更新参数
        if use_optimized_params:
            self._load_and_apply_optimized_params()
    
    def _load_and_apply_optimized_params(self):
        """
        加载并应用优化后的参数
        """
        try:
            # 尝试从文件加载优化参数
            import json
            with open('optimized_parameters.json', 'r') as f:
                optimized_params = json.load(f)['best_params']
            
            print("应用优化参数...")
            
            # 更新人类车辆参数
            self.params['human']['reaction_time'] = optimized_params['human_reaction_time']
            self.params['human']['safe_gap'] = optimized_params['human_safe_gap']
            self.params['human']['random_slow_prob'] = optimized_params['human_random_slow_prob']
            self.params['human']['acceleration'] = optimized_params['human_acceleration']
            self.params['human']['deceleration'] = optimized_params['human_deceleration']
            
            # 更新自动驾驶车辆参数
            self.params['av']['reaction_time'] = optimized_params['av_reaction_time']
            self.params['av']['safe_gap'] = optimized_params['av_safe_gap']
            self.params['av']['acceleration'] = optimized_params['av_acceleration']
            self.params['av']['deceleration'] = optimized_params['av_deceleration']
            
            # 存储交互参数
            self.optimized_params = optimized_params
            
            print("优化参数应用成功!")
            
        except FileNotFoundError:
            print("未找到优化参数文件，使用默认参数")
            self.optimized_params = None
    
    def update_speed_human_optimized(self, current_speed, gap, front_speed, front_type):
        """
        使用优化参数更新人类车辆速度
        """
        params = self.params['human']
        v_max = params['v_max_cells']
        
        # 1. 期望速度（趋向最大速度）
        v_desired = min(current_speed + params['acceleration'], v_max)
        
        # 2. 安全距离检查
        safe_distance = max(current_speed * params['safe_gap'], 2)  # 至少2个单元格
        
        if gap < safe_distance:
            # 需要减速
            if front_type == 2:  # 前车是AV
                # 对AV前车更谨慎（使用优化参数）
                if self.optimized_params and 'human_follow_av_caution' in self.optimized_params:
                    caution_factor = self.optimized_params['human_follow_av_caution']
                else:
                    caution_factor = 1.2
                
                deceleration = params['deceleration'] * caution_factor
            else:
                deceleration = params['deceleration']
            
            v_desired = max(gap - 1, 0)
        
        # 3. 随机减速（模拟人类不确定性）
        if np.random.random() < params['random_slow_prob'] and v_desired > 0:
            v_desired = max(v_desired - 1, 0)
        
        return v_desired
    
    def update_speed_av_optimized(self, current_speed, gap, front_speed, front_type):
        """
        使用优化参数更新自动驾驶车辆速度
        """
        params = self.params['av']
        v_max = params['v_max_cells']
        
        # 1. 更激进的加速
        v_desired = min(current_speed + params['acceleration'], v_max)
        
        # 2. 智能跟驰（考虑前车速度和类型）
        if front_type > 0:  # 有前车
            if front_type == 2:  # 前车也是AV
                # AV跟随AV：可以更紧密，考虑协同
                if self.optimized_params and 'av_follow_av_gap_factor' in self.optimized_params:
                    gap_factor = self.optimized_params['av_follow_av_gap_factor']
                else:
                    gap_factor = 0.7
                
                safe_distance = max(current_speed * params['safe_gap'] * gap_factor, 1)
                
                # 尝试匹配前车速度（协同）
                if front_speed < v_desired and gap < safe_distance * 2:
                    v_desired = min(v_desired, front_speed + 1)
            else:  # 前车是人类
                # AV跟随人类：更谨慎
                safe_distance = max(current_speed * params['safe_gap'] * 1.2, 2)
            
            if gap < safe_distance:
                v_desired = max(gap - 1, 0)
        
        # 3. 平滑加速（避免急加速）
        acceleration_limit = params['acceleration'] * self.dt
        v_desired = min(v_desired, current_speed + acceleration_limit)
        
        return v_desired
    
    def run_with_optimized_rules(self, warmup=100, verbose=True):
        """
        使用优化规则运行仿真
        """
        # 保存原方法
        original_human_update = self.update_speed_human
        original_av_update = self.update_speed_av
        
        # 替换为优化方法
        self.update_speed_human = self.update_speed_human_optimized
        self.update_speed_av = self.update_speed_av_optimized
        
        # 运行仿真
        self.run(warmup=warmup, verbose=verbose)
        
        # 恢复原方法
        self.update_speed_human = original_human_update
        self.update_speed_av = original_av_update


def integrated_main():
    """
    集成贝叶斯优化的主函数
    """
    print("=" * 60)
    print("2017 MCM C题：完整模型（集成贝叶斯优化）")
    print("=" * 60)
    
    # 选项：是否重新运行优化
    print("\n选项:")
    print("1. 运行贝叶斯优化参数校准（首次使用或重新优化）")
    print("2. 使用已有优化参数进行仿真")
    print("3. 对比优化前后效果")
    
    choice = input("\n请选择 (1-3): ").strip()
    
    if choice == '1':
        # 运行贝叶斯优化
        print("\n" + "=" * 60)
        print("运行贝叶斯优化参数校准")
        print("=" * 60)
        
        from bayesian_optimization import main as run_optimization
        optimizer, optimized_model = run_optimization()
        
        # 使用优化后的模型进行分析
        p_values = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        all_results = []
        
        print("\n使用优化参数进行多场景分析...")
        for p in tqdm(p_values, desc="不同p值仿真"):
            model = EnhancedTrafficCA(
                length=1000,
                lanes=3,
                p_av=p,
                density=0.2,
                time_steps=300,
                use_optimized_params=True
            )
            model.run_with_optimized_rules(warmup=50, verbose=False)
            stats = model.get_summary_stats()
            all_results.append(stats)
        
        # 可视化
        visualizer = TrafficVisualizer(optimized_model)
        fig, tipping_point = visualizer.plot_comparison(
            all_results, p_values, save_path='optimized_comparison.png'
        )
        
    elif choice == '2':
        # 使用已有优化参数
        print("\n" + "=" * 60)
        print("使用已有优化参数进行仿真")
        print("=" * 60)
        
        # 创建使用优化参数的模型
        model = EnhancedTrafficCA(
            length=1000,
            lanes=3,
            p_av=0.5,
            density=0.2,
            time_steps=500,
            use_optimized_params=True
        )
        
        print("运行优化后的CA模型...")
        model.run_with_optimized_rules(warmup=50, verbose=True)
        
        # 可视化
        visualizer = TrafficVisualizer(model)
        fig = visualizer.plot_road_snapshot(save_path='optimized_road_snapshot.png')
        
        # 运行多p值分析
        print("\n运行多p值分析...")
        p_values = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        all_results = []
        
        for p in tqdm(p_values, desc="不同p值仿真"):
            model_p = EnhancedTrafficCA(
                length=1000,
                lanes=3,
                p_av=p,
                density=0.2,
                time_steps=300,
                use_optimized_params=True
            )
            model_p.run_with_optimized_rules(warmup=50, verbose=False)
            stats = model_p.get_summary_stats()
            all_results.append(stats)
        
        # 比较分析
        fig2, tipping_point = visualizer.plot_comparison(
            all_results, p_values, save_path='optimized_multi_p.png'
        )
        
        print(f"\n优化模型临界点: p ≈ {tipping_point:.2f}")
        
    elif choice == '3':
        # 对比优化前后效果
        print("\n" + "=" * 60)
        print("优化前后对比分析")
        print("=" * 60)
        
        p_values = [0, 0.3, 0.5, 0.7, 1.0]
        
        # 存储结果
        default_results = []
        optimized_results = []
        
        print("\n运行默认参数模型...")
        for p in tqdm(p_values, desc="默认参数"):
            model = TrafficCA(
                length=1000,
                lanes=3,
                p_av=p,
                density=0.2,
                time_steps=300
            )
            model.run(warmup=50, verbose=False)
            stats = model.get_summary_stats()
            default_results.append(stats)
        
        print("\n运行优化参数模型...")
        for p in tqdm(p_values, desc="优化参数"):
            model = EnhancedTrafficCA(
                length=1000,
                lanes=3,
                p_av=p,
                density=0.2,
                time_steps=300,
                use_optimized_params=True
            )
            model.run_with_optimized_rules(warmup=50, verbose=False)
            stats = model.get_summary_stats()
            optimized_results.append(stats)
        
        # 创建对比图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 平均速度对比
        ax1 = axes[0, 0]
        default_speeds = [r['mean_speed'] for r in default_results]
        optimized_speeds = [r['mean_speed'] for r in optimized_results]
        
        ax1.plot(p_values, default_speeds, 'ro-', linewidth=2, markersize=8, label='默认参数')
        ax1.plot(p_values, optimized_speeds, 'bo-', linewidth=2, markersize=8, label='优化参数')
        ax1.fill_between(p_values, default_speeds, optimized_speeds, alpha=0.2, color='green')
        
        ax1.set_xlabel('自动驾驶比例 (p)', fontsize=12)
        ax1.set_ylabel('平均速度 (mph)', fontsize=12)
        ax1.set_title('平均速度对比', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 计算改进幅度
        for i, p in enumerate(p_values):
            improvement = (optimized_speeds[i] - default_speeds[i]) / default_speeds[i] * 100
            ax1.annotate(f'+{improvement:.1f}%', 
                        xy=(p, (default_speeds[i] + optimized_speeds[i])/2),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', fontsize=9, color='green')
        
        # 2. 平均流量对比
        ax2 = axes[0, 1]
        default_flows = [r['mean_flow'] for r in default_results]
        optimized_flows = [r['mean_flow'] for r in optimized_results]
        
        ax2.plot(p_values, default_flows, 'ro-', linewidth=2, markersize=8, label='默认参数')
        ax2.plot(p_values, optimized_flows, 'bo-', linewidth=2, markersize=8, label='优化参数')
        
        ax2.set_xlabel('自动驾驶比例 (p)', fontsize=12)
        ax2.set_ylabel('平均流量 (辆/小时)', fontsize=12)
        ax2.set_title('流量对比', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 拥堵指数对比
        ax3 = axes[1, 0]
        default_congestion = [r['congestion_index'] for r in default_results]
        optimized_congestion = [r['congestion_index'] for r in optimized_results]
        
        width = 0.35
        x = np.arange(len(p_values))
        
        ax3.bar(x - width/2, default_congestion, width, label='默认参数', alpha=0.8, color='red')
        ax3.bar(x + width/2, optimized_congestion, width, label='优化参数', alpha=0.8, color='blue')
        
        ax3.set_xticks(x)
        ax3.set_xticklabels(p_values, fontsize=10)
        ax3.set_xlabel('自动驾驶比例 (p)', fontsize=12)
        ax3.set_ylabel('拥堵指数', fontsize=12)
        ax3.set_title('拥堵指数对比', fontsize=14)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. 改进百分比总结
        ax4 = axes[1, 1]
        
        improvements = []
        for i in range(len(p_values)):
            speed_improvement = (optimized_speeds[i] - default_speeds[i]) / default_speeds[i] * 100
            flow_improvement = (optimized_flows[i] - default_flows[i]) / default_flows[i] * 100
            congestion_improvement = (default_congestion[i] - optimized_congestion[i]) / default_congestion[i] * 100
            
            improvements.append({
                'p': p_values[i],
                'speed': speed_improvement,
                'flow': flow_improvement,
                'congestion': congestion_improvement
            })
        
        # 创建堆积条形图
        speed_improvements = [imp['speed'] for imp in improvements]
        flow_improvements = [imp['flow'] for imp in improvements]
        
        ax4.bar(x, speed_improvements, width, label='速度改进', alpha=0.7, color='green')
        ax4.bar(x, flow_improvements, width, bottom=speed_improvements, 
               label='流量改进', alpha=0.7, color='blue')
        
        ax4.set_xticks(x)
        ax4.set_xticklabels([f'p={p}' for p in p_values], fontsize=10)
        ax4.set_xlabel('自动驾驶比例', fontsize=12)
        ax4.set_ylabel('改进百分比 (%)', fontsize=12)
        ax4.set_title('优化改进总结', fontsize=14)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for i, imp in enumerate(improvements):
            total_height = imp['speed'] + imp['flow']
            ax4.text(i, total_height + 2, f'+{total_height:.1f}%', 
                    ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('贝叶斯优化参数校准效果对比分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('optimization_comparison.png', dpi=300, bbox_inches='tight')
        
        print("\n对比分析完成!")
        print(f"优化参数在p=0.5时的改进:")
        print(f"  速度改进: {improvements[2]['speed']:.1f}%")
        print(f"  流量改进: {improvements[2]['flow']:.1f}%")
        print(f"  拥堵改进: {improvements[2]['congestion']:.1f}%")
        
        plt.show()
    
    else:
        print("无效选择，退出程序")
        return
    
    print("\n" + "=" * 60)
    print("ML2模块集成完成!")
    print("=" * 60)


if __name__ == "__main__":
    integrated_main()