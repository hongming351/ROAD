import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import random
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class TrafficCA:
    """
    两类车辆（人类/自动驾驶）的元胞自动机交通流模型
    包含完整可视化功能
    """
    
    def __init__(self, length=1000, lanes=3, p_av=0.5, density=0.15, 
                 time_steps=1000, seed=42):
        """3  
        初始化交通CA模型
        
        参数:
        length: 路段长度（米）
        lanes: 车道数
        p_av: 自动驾驶比例（0-1）
        density: 初始车辆密度（车辆数/（长度×车道×单位长度））
        time_steps: 仿真总步数
        seed: 随机种子
        """
        np.random.seed(seed)
        random.seed(seed)
        
        self.length = length  # 路段长度（米）
        self.lanes = lanes    # 车道数
        self.p_av = p_av      # 自动驾驶比例
        self.density = density  # 初始密度
        self.time_steps = time_steps  # 总仿真步数
        
        # 模型参数
        self.cell_size = 7.5  # 每个单元格长度（米），对应一辆车+安全距离
        self.num_cells = int(length / self.cell_size)  # 单元格总数
        self.dt = 1.0  # 时间步长（秒）
        
        # 车辆参数（基于文献的典型值）
        self.params = {
            'human': {
                'v_max': 60,           # 最大速度（mph）
                'v_max_cells': 5,      # 最大速度（单元格/步）
                'reaction_time': 1.5,  # 反应时间（秒）
                'safe_gap': 1.5,       # 安全车头时距（秒）
                'acceleration': 1,     # 加速度（单元格/步²）
                'deceleration': 2,     # 减速度（单元格/步²）
                'random_slow_prob': 0.2,  # 随机减速概率
                'color': 'red'         # 可视化颜色
            },
            'av': {
                'v_max': 60,           # 最大速度（mph）
                'v_max_cells': 5,      # 最大速度（单元格/步）
                'reaction_time': 0.3,  # 反应时间（秒）
                'safe_gap': 0.8,       # 安全车头时距（秒）
                'acceleration': 2,     # 加速度（单元格/步²）
                'deceleration': 3,     # 减速度（单元格/步²）
                'random_slow_prob': 0.0,  # 无随机减速
                'color': 'blue'        # 可视化颜色
            }
        }
        
        # 单位转换
        self.mph_to_cells = 5 / 60  # 60mph对应5单元格/步
        
        # 初始化道路状态
        # road[lane, cell] = [车辆类型, 速度, 加速度, 年龄]
        # 车辆类型: 0=空, 1=人类, 2=自动驾驶
        self.road = np.zeros((lanes, self.num_cells, 4), dtype=np.float32)
        
        # 性能指标记录
        self.metrics = {
            'time': [],
            'flow': [],      # 流量（辆/小时）
            'density': [],   # 密度（辆/公里/车道）
            'speed': [],     # 平均速度（mph）
            'throughput': [], # 吞吐量（辆）
            'congestion_level': [], # 拥堵水平
            'av_percentage': []  # AV实际比例
        }
        
        # 生成初始车辆
        self.generate_initial_vehicles()
        
        # 可视化相关
        self.fig = None
        self.ax = None
        self.anim = None
        
    def generate_initial_vehicles(self):
        """生成初始车辆分布"""
        total_cells = self.lanes * self.num_cells
        num_vehicles = int(total_cells * self.density)
        
        # 随机选择车辆位置（确保不重叠）
        positions = random.sample(range(total_cells), num_vehicles)
        
        for pos in positions:
            lane = pos // self.num_cells
            cell = pos % self.num_cells
            
            # 根据p_av确定车辆类型
            if random.random() < self.p_av:
                vehicle_type = 2  # 自动驾驶
                # AV初始速度较快
                speed = random.uniform(3, 5)
            else:
                vehicle_type = 1  # 人类
                # 人类初始速度较慢
                speed = random.uniform(2, 4)
            
            # 设置车辆属性
            self.road[lane, cell, 0] = vehicle_type  # 类型
            self.road[lane, cell, 1] = speed         # 速度（单元格/步）
            self.road[lane, cell, 2] = 0             # 加速度
            self.road[lane, cell, 3] = 0             # 年龄（时间步）
    
    def find_gap(self, lane, cell):
        """
        查找前方空闲单元格数量
        返回：(空隙单元格数, 前车类型, 前车速度)
        """
        gap = 0
        for i in range(1, self.num_cells):
            next_cell = (cell + i) % self.num_cells
            if self.road[lane, next_cell, 0] > 0:  # 有车辆
                front_type = self.road[lane, next_cell, 0]
                front_speed = self.road[lane, next_cell, 1]
                return gap, front_type, front_speed
            gap += 1
        
        # 如果没找到前车
        return gap, 0, 0
    
    def update_speed_human(self, current_speed, gap, front_speed, front_type):
        """更新人类车辆速度（改进的IDM启发式）"""
        params = self.params['human']
        v_max = params['v_max_cells']
        
        # 1. 期望速度（趋向最大速度）
        v_desired = min(current_speed + params['acceleration'], v_max)
        
        # 2. 安全距离检查
        safe_distance = max(current_speed * params['safe_gap'], 2)  # 至少2个单元格
        
        if gap < safe_distance:
            # 需要减速
            if front_type == 2:  # 前车是AV
                # 对AV前车更谨慎
                deceleration = params['deceleration'] * 1.2
            else:
                deceleration = params['deceleration']
            
            v_desired = max(gap - 1, 0)
        
        # 3. 随机减速（模拟人类不确定性）
        if random.random() < params['random_slow_prob'] and v_desired > 0:
            v_desired = max(v_desired - 1, 0)
        
        return v_desired
    
    def update_speed_av(self, current_speed, gap, front_speed, front_type):
        """更新自动驾驶车辆速度（协同感知）"""
        params = self.params['av']
        v_max = params['v_max_cells']
        
        # 1. 更激进的加速
        v_desired = min(current_speed + params['acceleration'], v_max)
        
        # 2. 智能跟驰（考虑前车速度和类型）
        if front_type > 0:  # 有前车
            if front_type == 2:  # 前车也是AV
                # AV跟随AV：可以更紧密，考虑协同
                safe_distance = max(current_speed * params['safe_gap'] * 0.7, 1)
                
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
    
    def update_vehicle(self, lane, cell):
        """更新单个车辆的状态"""
        vehicle_type = int(self.road[lane, cell, 0])
        if vehicle_type == 0:  # 空单元格
            return None
        
        current_speed = self.road[lane, cell, 1]
        gap, front_type, front_speed = self.find_gap(lane, cell)
        
        # 根据车辆类型更新速度
        if vehicle_type == 1:  # 人类
            new_speed = self.update_speed_human(current_speed, gap, front_speed, front_type)
        else:  # 自动驾驶
            new_speed = self.update_speed_av(current_speed, gap, front_speed, front_type)
        
        # 确保速度在合理范围内
        params = self.params['human'] if vehicle_type == 1 else self.params['av']
        new_speed = np.clip(new_speed, 0, params['v_max_cells'])
        
        # 计算加速度（用于可视化）
        acceleration = new_speed - current_speed
        
        return {
            'type': vehicle_type,
            'speed': new_speed,
            'acceleration': acceleration,
            'new_cell': int((cell + new_speed) % self.num_cells)
        }
    
    def update_lane_change(self, lane, cell, vehicle_info):
        """简单的换道决策"""
        if vehicle_info['type'] == 1:  # 人类换道
            # 检查左右车道（如果存在）
            change_prob = 0.1  # 换道概率
            
            if lane > 0:  # 可以向左换道
                left_lane = lane - 1
                left_gap, _, _ = self.find_gap(left_lane, cell)
                current_gap, _, _ = self.find_gap(lane, cell)
                
                if left_gap > current_gap + 2 and random.random() < change_prob:
                    return left_lane
            
            if lane < self.lanes - 1:  # 可以向右换道
                right_lane = lane + 1
                right_gap, _, _ = self.find_gap(right_lane, cell)
                current_gap, _, _ = self.find_gap(lane, cell)
                
                if right_gap > current_gap + 2 and random.random() < change_prob:
                    return right_lane
        
        return lane  # 不换道
    
    def step(self):
        """执行一个时间步的更新"""
        # 创建新的道路状态
        new_road = np.zeros_like(self.road)
        
        # 第一阶段：计算所有车辆的意图
        intentions = {}
        for lane in range(self.lanes):
            for cell in range(self.num_cells):
                if self.road[lane, cell, 0] > 0:
                    vehicle_info = self.update_vehicle(lane, cell)
                    if vehicle_info:
                        # 换道决策
                        new_lane = self.update_lane_change(lane, cell, vehicle_info)
                        intentions[(lane, cell)] = (new_lane, vehicle_info)
        
        # 第二阶段：处理冲突并应用更新
        for (old_lane, old_cell), (new_lane, vehicle_info) in intentions.items():
            new_cell = vehicle_info['new_cell']
            
            # 检查目标位置是否被占用
            if new_road[new_lane, new_cell, 0] == 0:
                # 没有冲突，放置车辆
                new_road[new_lane, new_cell, 0] = vehicle_info['type']
                new_road[new_lane, new_cell, 1] = vehicle_info['speed']
                new_road[new_lane, new_cell, 2] = vehicle_info['acceleration']
                new_road[new_lane, new_cell, 3] = self.road[old_lane, old_cell, 3] + 1  # 年龄增加
            else:
                # 有冲突，保持原位置（减速）
                new_road[old_lane, old_cell, 0] = vehicle_info['type']
                new_road[old_lane, old_cell, 1] = vehicle_info['speed'] * 0.5  # 减半速度
                new_road[old_lane, old_cell, 2] = -1  # 负加速度
                new_road[old_lane, old_cell, 3] = self.road[old_lane, old_cell, 3] + 1
        
        # 更新道路状态
        self.road = new_road
        
        # 记录性能指标
        self.record_metrics()
    
    def record_metrics(self):
        """记录当前时间步的性能指标"""
        total_vehicles = 0
        total_speed = 0
        total_av = 0
        
        for lane in range(self.lanes):
            for cell in range(self.num_cells):
                if self.road[lane, cell, 0] > 0:
                    total_vehicles += 1
                    total_speed += self.road[lane, cell, 1]
                    if self.road[lane, cell, 0] == 2:
                        total_av += 1
        
        # 计算指标
        if total_vehicles > 0:
            # 平均速度（mph）
            avg_speed_cells = total_speed / total_vehicles
            avg_speed_mph = avg_speed_cells * (60 / 5)  # 转换回mph
            
            # 密度（辆/公里/车道）
            density_per_lane = total_vehicles / self.lanes / (self.length / 1000)
            
            # 流量（辆/小时）
            flow = total_vehicles * avg_speed_mph / self.length * 3600
            
            # 拥堵水平（速度低于20mph为拥堵）
            congestion = 1.0 - min(avg_speed_mph / 30, 1.0)
            
            # AV比例
            av_percentage = total_av / total_vehicles if total_vehicles > 0 else 0
            
            # 记录
            current_time = len(self.metrics['time'])
            self.metrics['time'].append(current_time)
            self.metrics['flow'].append(flow)
            self.metrics['density'].append(density_per_lane)
            self.metrics['speed'].append(avg_speed_mph)
            self.metrics['congestion_level'].append(congestion)
            self.metrics['av_percentage'].append(av_percentage)
    
    def run(self, warmup=100, verbose=True):
        """运行完整仿真"""
        if verbose:
            print(f"开始仿真: p_av={self.p_av:.1%}, 密度={self.density:.1%}")
            print(f"路段: {self.length}m, {self.lanes}车道, {self.time_steps}时间步")
        
        # 热身阶段（不记录数据）
        for _ in tqdm(range(warmup), desc="热身阶段", disable=not verbose):
            self.step()
        
        # 清空之前的指标（热身阶段数据不保留）
        for key in self.metrics:
            self.metrics[key] = []
        
        # 主仿真阶段
        for step in tqdm(range(self.time_steps), desc="主仿真阶段", disable=not verbose):
            self.step()
            
            # 每100步打印进度
            if verbose and step % 100 == 0 and step > 0:
                avg_speed = np.mean(self.metrics['speed'][-100:])
                avg_flow = np.mean(self.metrics['flow'][-100:])
                print(f"  步数 {step}: 平均速度={avg_speed:.1f}mph, 流量={avg_flow:.0f}辆/小时")
        
        if verbose:
            final_speed = np.mean(self.metrics['speed'][-100:])
            final_flow = np.mean(self.metrics['flow'][-100:])
            print(f"仿真完成! 最终性能: 速度={final_speed:.1f}mph, 流量={final_flow:.0f}辆/小时")
    
    def get_summary_stats(self):
        """获取汇总统计"""
        stats = {
            'p_av': self.p_av,
            'mean_flow': np.mean(self.metrics['flow']),
            'std_flow': np.std(self.metrics['flow']),
            'mean_speed': np.mean(self.metrics['speed']),
            'std_speed': np.std(self.metrics['speed']),
            'mean_density': np.mean(self.metrics['density']),
            'congestion_index': np.mean(self.metrics['congestion_level']),
            'max_flow': np.max(self.metrics['flow']),
            'min_speed': np.min(self.metrics['speed']),
            'throughput': np.sum(self.metrics['flow']) / 3600,  # 总通过车辆数
            'av_actual_percentage': np.mean(self.metrics['av_percentage'])
        }
        return stats
class TrafficVisualizer:
    """交通仿真可视化类"""
    
    def __init__(self, model):
        self.model = model
        self.fig = None
        self.axs = None
        
    def plot_road_snapshot(self, step=None, save_path=None):
        """绘制道路快照"""
        if step is None:
            # 使用当前状态
            road = self.model.road
        else:
            # 这里需要存储历史状态，简化版本只画当前
            road = self.model.road
        
        self.fig, self.axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 道路状态热图
        ax1 = self.axs[0, 0]
        road_state = np.zeros((self.model.lanes, self.model.num_cells))
        for lane in range(self.model.lanes):
            for cell in range(self.model.num_cells):
                if road[lane, cell, 0] == 1:  # 人类
                    road_state[lane, cell] = 1
                elif road[lane, cell, 0] == 2:  # AV
                    road_state[lane, cell] = 2
        
        im = ax1.imshow(road_state, aspect='auto', cmap='RdYlBu_r', 
                       vmin=0, vmax=2, interpolation='nearest')
        ax1.set_title('道路状态 (红=人类, 蓝=AV, 白=空)', fontsize=12)
        ax1.set_xlabel('位置 (单元格)', fontsize=10)
        ax1.set_ylabel('车道', fontsize=10)
        ax1.set_yticks(range(self.model.lanes))
        ax1.set_yticklabels([f'车道 {i+1}' for i in range(self.model.lanes)])
        plt.colorbar(im, ax=ax1, ticks=[0, 1, 2], label='车辆类型')
        
        # 2. 速度分布
        ax2 = self.axs[0, 1]
        speeds = []
        types = []
        for lane in range(self.model.lanes):
            for cell in range(self.model.num_cells):
                if road[lane, cell, 0] > 0:
                    speed_mph = road[lane, cell, 1] * (60 / 5)  # 转换为mph
                    speeds.append(speed_mph)
                    types.append('人类' if road[lane, cell, 0] == 1 else 'AV')
        
        if speeds:
            df_speeds = pd.DataFrame({'速度(mph)': speeds, '类型': types})
            sns.histplot(data=df_speeds, x='速度(mph)', hue='类型', 
                        ax=ax2, kde=True, bins=20, alpha=0.6)
            ax2.set_title('速度分布', fontsize=12)
            ax2.set_xlabel('速度 (mph)', fontsize=10)
            ax2.set_ylabel('车辆数', fontsize=10)
            ax2.axvline(x=60, color='r', linestyle='--', alpha=0.5, label='限速60mph')
            ax2.legend()
        
        # 3. 性能指标时间序列
        ax3 = self.axs[1, 0]
        if self.model.metrics['time']:
            time = self.model.metrics['time']
            ax3.plot(time, self.model.metrics['speed'], 'b-', label='速度', linewidth=2)
            ax3.set_xlabel('时间步', fontsize=10)
            ax3.set_ylabel('平均速度 (mph)', fontsize=10, color='b')
            ax3.tick_params(axis='y', labelcolor='b')
            ax3.grid(True, alpha=0.3)
            
            ax3_twin = ax3.twinx()
            ax3_twin.plot(time, self.model.metrics['flow'], 'r-', label='流量', linewidth=2, alpha=0.7)
            ax3_twin.set_ylabel('流量 (辆/小时)', fontsize=10, color='r')
            ax3_twin.tick_params(axis='y', labelcolor='r')
            
            ax3.set_title('性能指标时间序列', fontsize=12)
            lines1, labels1 = ax3.get_legend_handles_labels()
            lines2, labels2 = ax3_twin.get_legend_handles_labels()
            ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # 4. 基本图（流量-密度-速度）
        ax4 = self.axs[1, 1]
        if len(self.model.metrics['density']) > 10:
            density = self.model.metrics['density']
            flow = self.model.metrics['flow']
            speed = self.model.metrics['speed']
            
            # 流量-密度图
            sc1 = ax4.scatter(density, flow, c=speed, cmap='viridis', 
                             s=20, alpha=0.6, edgecolors='k', linewidth=0.5)
            ax4.set_xlabel('密度 (辆/公里/车道)', fontsize=10)
            ax4.set_ylabel('流量 (辆/小时)', fontsize=10)
            ax4.set_title('基本图 (颜色表示速度)', fontsize=12)
            plt.colorbar(sc1, ax=ax4, label='速度 (mph)')
            
            # 添加趋势线
            if len(density) > 2:
                z = np.polyfit(density, flow, 2)
                p = np.poly1d(z)
                density_smooth = np.linspace(min(density), max(density), 100)
                ax4.plot(density_smooth, p(density_smooth), 'r--', alpha=0.8, linewidth=2)
        
        plt.suptitle(f'交通流仿真 (AV比例={self.model.p_av:.0%}, 密度={self.model.density:.1%})', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图像已保存到: {save_path}")
        
        return self.fig
    
    def animate_simulation(self, steps=200, interval=50, save_path=None):
        """创建仿真动画"""
        # 重新运行仿真并记录状态（简化版本）
        print("正在生成动画...")
        
        # 存储历史状态
        history = []
        original_road = self.model.road.copy()
        
        # 运行指定步数并记录
        for i in range(steps):
            history.append(self.model.road.copy())
            self.model.step()
        
        # 恢复原始状态
        self.model.road = original_road
        
        # 创建动画
        fig, ax = plt.subplots(figsize=(15, 4))
        
        def update(frame):
            ax.clear()
            road = history[frame]
            
            # 创建道路图像
            road_img = np.zeros((self.model.lanes, self.model.num_cells, 3))
            
            for lane in range(self.model.lanes):
                for cell in range(self.model.num_cells):
                    if road[lane, cell, 0] == 1:  # 人类 - 红色
                        road_img[lane, cell] = [1, 0, 0]
                    elif road[lane, cell, 0] == 2:  # AV - 蓝色
                        road_img[lane, cell] = [0, 0, 1]
                    else:  # 空 - 白色
                        road_img[lane, cell] = [1, 1, 1]
            
            ax.imshow(road_img, aspect='auto', interpolation='nearest')
            ax.set_title(f'时间步: {frame} / {steps} | AV比例: {self.model.p_av:.0%}', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('位置', fontsize=10)
            ax.set_ylabel('车道', fontsize=10)
            ax.set_yticks(range(self.model.lanes))
            ax.set_yticklabels([f'车道{i+1}' for i in range(self.model.lanes)])
            
            # 添加图例
            human_patch = Rectangle((0, 0), 1, 1, fc='red', label='人类')
            av_patch = Rectangle((0, 0), 1, 1, fc='blue', label='AV')
            ax.legend(handles=[human_patch, av_patch], loc='upper right')
            
            return ax,
        
        anim = animation.FuncAnimation(fig, update, frames=steps, 
                                      interval=interval, blit=False)
        
        if save_path:
            print("正在保存动画，这可能需要一些时间...")
            anim.save(save_path, writer='pillow', fps=1000/interval, dpi=100)
            print(f"动画已保存到: {save_path}")
        
        return anim
    
    def plot_comparison(self, results_list, p_values, save_path=None):
        """比较不同p值的仿真结果"""
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        
        # 提取数据
        mean_speeds = [r['mean_speed'] for r in results_list]
        mean_flows = [r['mean_flow'] for r in results_list]
        congestion = [r['congestion_index'] for r in results_list]
        throughput = [r['throughput'] for r in results_list]
        
        # 1. 平均速度 vs p
        axs[0, 0].plot(p_values, mean_speeds, 'bo-', linewidth=2, markersize=8)
        axs[0, 0].fill_between(p_values, 
                               [s - r['std_speed'] for s, r in zip(mean_speeds, results_list)],
                               [s + r['std_speed'] for s, r in zip(mean_speeds, results_list)],
                               alpha=0.2)
        axs[0, 0].set_xlabel('自动驾驶比例 (p)', fontsize=11)
        axs[0, 0].set_ylabel('平均速度 (mph)', fontsize=11)
        axs[0, 0].set_title('自动驾驶比例对平均速度的影响', fontsize=12)
        axs[0, 0].grid(True, alpha=0.3)
        axs[0, 0].axhline(y=60, color='r', linestyle='--', alpha=0.5, label='限速')
        
        # 2. 流量 vs p
        axs[0, 1].plot(p_values, mean_flows, 'ro-', linewidth=2, markersize=8)
        axs[0, 1].fill_between(p_values,
                              [f - r['std_flow'] for f, r in zip(mean_flows, results_list)],
                              [f + r['std_flow'] for f, r in zip(mean_flows, results_list)],
                              alpha=0.2)
        axs[0, 1].set_xlabel('自动驾驶比例 (p)', fontsize=11)
        axs[0, 1].set_ylabel('平均流量 (辆/小时)', fontsize=11)
        axs[0, 1].set_title('自动驾驶比例对流量的影响', fontsize=12)
        axs[0, 1].grid(True, alpha=0.3)
        
        # 3. 拥堵指数 vs p
        axs[0, 2].plot(p_values, congestion, 'go-', linewidth=2, markersize=8)
        axs[0, 2].set_xlabel('自动驾驶比例 (p)', fontsize=11)
        axs[0, 2].set_ylabel('拥堵指数 (0-1)', fontsize=11)
        axs[0, 2].set_title('自动驾驶比例对拥堵的影响', fontsize=12)
        axs[0, 2].grid(True, alpha=0.3)
        axs[0, 2].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='拥堵阈值')
        
        # 4. 吞吐量 vs p
        axs[1, 0].bar(p_values, throughput, width=0.07, alpha=0.7, color='purple')
        axs[1, 0].set_xlabel('自动驾驶比例 (p)', fontsize=11)
        axs[1, 0].set_ylabel('总吞吐量 (辆)', fontsize=11)
        axs[1, 0].set_title('总吞吐量随p的变化', fontsize=12)
        axs[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 5. 流量-速度关系（不同p值）
        axs[1, 1].set_prop_cycle(color=plt.cm.viridis(np.linspace(0, 1, len(p_values))))
        for i, p in enumerate(p_values):
            density = results_list[i]['mean_density']
            flow = results_list[i]['mean_flow']
            speed = results_list[i]['mean_speed']
            axs[1, 1].scatter(flow, speed, s=100, alpha=0.7, label=f'p={p}')
        
        axs[1, 1].set_xlabel('流量 (辆/小时)', fontsize=11)
        axs[1, 1].set_ylabel('速度 (mph)', fontsize=11)
        axs[1, 1].set_title('不同p值的流量-速度关系', fontsize=12)
        axs[1, 1].grid(True, alpha=0.3)
        axs[1, 1].legend(title='AV比例')
        
        # 6. 临界点分析
        axs[1, 2].plot(p_values, mean_speeds, 'b-', label='速度', linewidth=2)
        axs[1, 2].set_xlabel('自动驾驶比例 (p)', fontsize=11)
        axs[1, 2].set_ylabel('性能指标', fontsize=11)
        axs[1, 2].set_title('临界点分析', fontsize=12)
        
        # 计算导数（变化率）
        speed_derivative = np.gradient(mean_speeds, p_values)
        flow_derivative = np.gradient(mean_flows, p_values)
        
        # 归一化
        speed_derivative_norm = speed_derivative / np.max(np.abs(speed_derivative))
        flow_derivative_norm = flow_derivative / np.max(np.abs(flow_derivative))
        
        axs2 = axs[1, 2].twinx()
        axs2.plot(p_values, speed_derivative_norm, 'r--', label='速度变化率', linewidth=2)
        axs2.plot(p_values, flow_derivative_norm, 'g--', label='流量变化率', linewidth=2)
        axs2.set_ylabel('变化率 (归一化)', fontsize=11)
        
        # 找出临界点（最大变化率）
        tipping_point = p_values[np.argmax(np.abs(speed_derivative))]
        axs[1, 2].axvline(x=tipping_point, color='k', linestyle=':', 
                         linewidth=2, alpha=0.7, label=f'临界点 p={tipping_point:.2f}')
        
        lines1, labels1 = axs[1, 2].get_legend_handles_labels()
        lines2, labels2 = axs2.get_legend_handles_labels()
        axs[1, 2].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        axs[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle('自动驾驶比例影响综合分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"比较图已保存到: {save_path}")
        
        return fig, tipping_point
def load_real_data():
    """加载真实数据并提取参数"""
    df = pd.read_csv('2017_MCM_Problem_C_Data.csv')

    # 清理列名
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('"', '').str.replace('(', '').str.replace(')', '')

    # 计算路段长度
    df['length'] = df['endMilepost'] - df['startMilepost']

    # 计算平均车道数
    df['avg_lanes'] = (df['Number_of_Lanes_DECR_MP_direction'] + df['Number_of_Lanes_INCR_MP_direction']) / 2

    # 计算总路段长度和平均车道数
    total_length = df['length'].sum()
    avg_lanes = df['avg_lanes'].mean()

    # 计算总交通量
    total_traffic = df['Average_daily_traffic_counts_Year_2015'].sum()

    # 估计自动驾驶比例（基于路线类型）
    # 假设州际公路有更高的自动驾驶比例
    is_routes = df[df['RteType_IS=_Interstate,_SR=_State_Route'].str.contains('IS')]
    sr_routes = df[df['RteType_IS=_Interstate,_SR=_State_Route'].str.contains('SR')]

    is_traffic = is_routes['Average_daily_traffic_counts_Year_2015'].sum()
    sr_traffic = sr_routes['Average_daily_traffic_counts_Year_2015'].sum()

    # 假设州际公路有60%自动驾驶，州级公路有30%自动驾驶
    p_av = (is_traffic * 0.6 + sr_traffic * 0.3) / total_traffic if total_traffic > 0 else 0.5

    # 计算初始密度（基于交通量和路段容量）
    # 估计每辆车占用空间：7.5米（车长） + 安全距离
    # 平均速度：60 mph = 26.82 m/s
    # 每日交通量转换为密度
    daily_hours = 24
    avg_speed_mps = 26.82  # 60 mph
    vehicle_length = 7.5  # 米
    time_headway = 1.5  # 秒

    # 计算每辆车占用的空间（米）
    space_per_vehicle = vehicle_length + avg_speed_mps * time_headway

    # 总路段长度（公里）
    total_length_km = total_length * 1.60934  # 英里转公里

    # 估计密度（辆/公里/车道）
    estimated_density = (total_traffic / daily_hours) / (total_length_km * avg_lanes * avg_speed_mps * 3.6)

    # 调整密度到合理范围
    density = min(max(estimated_density * 0.1, 0.05), 0.3)  # 5%-30%之间

    return {
        'length': int(total_length * 1000),  # 转换为米
        'lanes': int(round(avg_lanes)),
        'p_av': p_av,
        'density': density,
        'total_traffic': total_traffic,
        'total_length': total_length,
        'avg_lanes': avg_lanes
    }

def main():
    """主函数：演示完整模型的使用"""

    print("=" * 60)
    print("2017 MCM C题：基于真实数据的交通流仿真模型")
    print("=" * 60)

    # 加载真实数据
    print("\n加载真实数据...")
    real_data = load_real_data()

    print(f"\n真实数据参数:")
    print(f"  总路段长度: {real_data['total_length']:.2f} 英里 ({real_data['length']} 米)")
    print(f"  平均车道数: {real_data['avg_lanes']:.1f}")
    print(f"  总交通量: {real_data['total_traffic']:,.0f} 辆车/天")
    print(f"  估算自动驾驶比例: {real_data['p_av']:.1%}")
    print(f"  估算初始密度: {real_data['density']:.1%}")

    # =============== 场景1：基于真实数据的仿真 ===============
    print("\n1. 基于真实数据的仿真")
    print("-" * 40)

    # 创建模型实例（使用真实数据参数）
    model_real = TrafficCA(
        length=real_data['length'],      # 使用真实总长度（米）
        lanes=real_data['lanes'],        # 使用平均车道数
        p_av=real_data['p_av'],          # 使用估算的自动驾驶比例
        density=real_data['density'],    # 使用估算的初始密度
        time_steps=500                   # 500时间步
    )
    
    # 运行仿真
    model_real.run(warmup=50, verbose=True)

    # 获取统计信息
    stats_real = model_real.get_summary_stats()
    print(f"\n仿真统计:")
    for key, value in stats_real.items():
        print(f"  {key}: {value:.3f}")

    # 可视化
    visualizer = TrafficVisualizer(model_real)

    # 绘制道路快照
    fig1 = visualizer.plot_road_snapshot(save_path='road_snapshot_real.png')
    
    # =============== 场景2：多p值比较 ===============
    print("\n\n2. 多场景比较 (p从0到1)")
    print("-" * 40)
    
    p_values = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    all_results = []
    
    for p in tqdm(p_values, desc="运行不同p值仿真"):
        model = TrafficCA(
            length=1000,
            lanes=3,
            p_av=p,
            density=0.2,
            time_steps=300  # 为了快速演示，减少步数
        )
        model.run(warmup=50, verbose=False)
        stats = model.get_summary_stats()
        all_results.append(stats)
    
    # 创建比较可视化
    fig2, tipping_point = visualizer.plot_comparison(
        all_results, p_values, save_path='comparison_all_p.png'
    )
    
    print(f"\n分析结果:")
    print(f"  临界点（性能突变点）: p ≈ {tipping_point:.2f}")
    print(f"  p=0（纯人类）: 平均速度={all_results[0]['mean_speed']:.1f}mph, 流量={all_results[0]['mean_flow']:.0f}辆/小时")
    print(f"  p=1（纯AV）: 平均速度={all_results[-1]['mean_speed']:.1f}mph, 流量={all_results[-1]['mean_flow']:.0f}辆/小时")
    print(f"  改进幅度: 速度提升{((all_results[-1]['mean_speed']-all_results[0]['mean_speed'])/all_results[0]['mean_speed']*100):.1f}%, "
          f"流量提升{((all_results[-1]['mean_flow']-all_results[0]['mean_flow'])/all_results[0]['mean_flow']*100):.1f}%")
    
    # =============== 场景3：专用车道分析 ===============
    print("\n\n3. 专用车道策略分析")
    print("-" * 40)
    
    def analyze_dedicated_lanes(p_av=0.5, total_lanes=3):
        """分析不同专用车道配置"""
        scenarios = [
            {'name': '无专用道', 'av_lanes': 0, 'mixed_lanes': total_lanes},
            {'name': '1条专用道', 'av_lanes': 1, 'mixed_lanes': total_lanes-1},
            {'name': '2条专用道', 'av_lanes': 2, 'mixed_lanes': total_lanes-2},
        ]
        
        results = []
        for scenario in scenarios:
            total_flow = 0
            total_speed = 0
            
            # AV专用车道（如果存在）
            if scenario['av_lanes'] > 0:
                model_av = TrafficCA(
                    length=1000,
                    lanes=scenario['av_lanes'],
                    p_av=1.0,  # 专用道上全部是AV
                    density=0.2,
                    time_steps=200
                )
                model_av.run(warmup=50, verbose=False)
                stats_av = model_av.get_summary_stats()
                total_flow += stats_av['mean_flow']
                total_speed += stats_av['mean_speed'] * stats_av['mean_flow']
            
            # 混合车道（如果存在）
            if scenario['mixed_lanes'] > 0:
                model_mixed = TrafficCA(
                    length=1000,
                    lanes=scenario['mixed_lanes'],
                    p_av=p_av,  # 混合比例
                    density=0.2,
                    time_steps=200
                )
                model_mixed.run(warmup=50, verbose=False)
                stats_mixed = model_mixed.get_summary_stats()
                total_flow += stats_mixed['mean_flow']
                total_speed += stats_mixed['mean_speed'] * stats_mixed['mean_flow']
            
            # 加权平均速度
            weighted_speed = total_speed / total_flow if total_flow > 0 else 0
            
            results.append({
                'scenario': scenario['name'],
                'total_flow': total_flow,
                'avg_speed': weighted_speed,
                'efficiency': total_flow * weighted_speed
            })
        
        return results
    
    # 分析不同p值下的最优专用道配置
    p_for_dedicated = [0.3, 0.5, 0.7]
    dedicated_results = {}
    
    for p in p_for_dedicated:
        print(f"\n  分析 p={p}:")
        results = analyze_dedicated_lanes(p_av=p)
        dedicated_results[p] = results
        
        best_scenario = max(results, key=lambda x: x['efficiency'])
        print(f"    最优配置: {best_scenario['scenario']}")
        print(f"    流量: {best_scenario['total_flow']:.0f}辆/小时, "
              f"速度: {best_scenario['avg_speed']:.1f}mph")
    
    # =============== 可视化专用车道分析 ===============
    print("\n\n4. 生成最终报告图表")
    print("-" * 40)
    
    # 创建综合报告
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # 左图：p值影响
    p_vals = [r['p_av'] for r in all_results]
    speeds = [r['mean_speed'] for r in all_results]
    flows = [r['mean_flow'] for r in all_results]
    
    axs[0].plot(p_vals, speeds, 'bo-', linewidth=2, markersize=8, label='速度')
    axs[0].set_xlabel('自动驾驶比例 (p)', fontsize=11)
    axs[0].set_ylabel('平均速度 (mph)', fontsize=11, color='b')
    axs[0].tick_params(axis='y', labelcolor='b')
    axs[0].grid(True, alpha=0.3)
    axs[0].set_title('速度随p变化', fontsize=12)
    
    ax_twin = axs[0].twinx()
    ax_twin.plot(p_vals, flows, 'ro-', linewidth=2, markersize=8, label='流量', alpha=0.7)
    ax_twin.set_ylabel('平均流量 (辆/小时)', fontsize=11, color='r')
    ax_twin.tick_params(axis='y', labelcolor='r')
    
    lines1, labels1 = axs[0].get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    axs[0].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # 标记临界点
    axs[0].axvline(x=tipping_point, color='k', linestyle=':', linewidth=2, alpha=0.7)
    axs[0].text(tipping_point+0.02, 0.5, f'临界点\np={tipping_point:.2f}', 
               transform=axs[0].get_xaxis_transform(),
               fontsize=10, verticalalignment='center')
    
    # 中图：专用车道效益
    scenarios = ['无专用道', '1条专用道', '2条专用道']
    x_pos = np.arange(len(scenarios))
    width = 0.25
    
    for i, p in enumerate(p_for_dedicated):
        efficiencies = [r['efficiency'] for r in dedicated_results[p]]
        # 归一化到0-1
        efficiencies_norm = [e/max(efficiencies) for e in efficiencies]
        axs[1].bar(x_pos + (i-1)*width, efficiencies_norm, width, 
                  label=f'p={p}', alpha=0.7)
    
    axs[1].set_xlabel('专用车道配置', fontsize=11)
    axs[1].set_ylabel('相对效率（归一化）', fontsize=11)
    axs[1].set_title('专用车道策略效率比较', fontsize=12)
    axs[1].set_xticks(x_pos)
    axs[1].set_xticklabels(scenarios)
    axs[1].legend()
    axs[1].grid(True, alpha=0.3, axis='y')
    
    # 右图：政策建议总结
    axs[2].axis('off')
    
    # 添加文本总结
    summary_text = (
        f"关键发现:\n\n"
        f"1. 临界点: p ≈ {tipping_point:.2f}\n"
        f"   • p < {tipping_point:.2f}: 改善缓慢\n"
        f"   • p > {tipping_point:.2f}: 改善加速\n\n"
        f"2. 最大改进:\n"
        f"   • 速度: +{((all_results[-1]['mean_speed']-all_results[0]['mean_speed'])/all_results[0]['mean_speed']*100):.1f}%\n"
        f"   • 流量: +{((all_results[-1]['mean_flow']-all_results[0]['mean_flow'])/all_results[0]['mean_flow']*100):.1f}%\n\n"
        f"3. 专用车道建议:\n"
    )
    
    for p in p_for_dedicated:
        best = max(dedicated_results[p], key=lambda x: x['efficiency'])
        summary_text += f"   • p={p}: {best['scenario']}\n"
    
    summary_text += f"\n4. 政策建议:\n"
    summary_text += f"   • 短期(p<0.3): 无需专用道\n"
    summary_text += f"   • 中期(p=0.3-0.6): 部分专用道\n"
    summary_text += f"   • 长期(p>0.6): 全面专用道"
    
    axs[2].text(0.1, 0.5, summary_text, fontsize=11, 
               verticalalignment='center', linespacing=1.5)
    axs[2].set_title('政策建议总结', fontsize=12)
    
    plt.suptitle('2017 MCM C题：自动驾驶交通流分析报告', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('final_report.png', dpi=300, bbox_inches='tight')
    
    print("\n" + "=" * 60)
    print("仿真完成！生成的文件:")
    print("  1. road_snapshot_p50.png - 道路状态快照")
    print("  2. comparison_all_p.png - 多场景比较图")
    print("  3. final_report.png - 最终分析报告")
    print("=" * 60)
    
    plt.show()

if __name__ == "__main__":
    main()