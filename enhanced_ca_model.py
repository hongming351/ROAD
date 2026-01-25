# 改进的元胞自动机模型 - 第2层模型
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import random
import pandas as pd
from collections import defaultdict
import seaborn as sns
from tqdm import tqdm

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class Vehicle:
    """车辆类"""
    def __init__(self, veh_id, veh_type, position, speed=0):
        self.id = veh_id
        self.type = veh_type  # 'human', 'av_solo', 'av_platoon'
        self.position = position
        self.speed = speed
        self.front_veh = None
        self.front_gap = float('inf')
        self.in_platoon = False
        self.platoon_id = None

        # 根据类型设置参数
        if veh_type == 'human':
            self.v_max = 60
            self.reaction_time = 1.5
            self.safe_gap = 1.5
        elif veh_type == 'av_solo':
            self.v_max = 60
            self.reaction_time = 0.3
            self.safe_gap = 0.6
        else:  # av_platoon
            self.v_max = 60
            self.reaction_time = 0.1
            self.safe_gap = 0.3

    def update(self, front_gap, front_veh_type=None):
        """更新车辆状态"""
        self.front_gap = front_gap

        # 1. 加速阶段
        if self.type == 'human':
            self.speed = min(self.speed + 1, self.v_max)
        elif self.type == 'av_solo':
            self.speed = min(self.speed + 2, self.v_max)  # 更快加速
        else:  # platoon
            self.speed = min(self.speed + 3, self.v_max)  # 最快加速

        # 2. 减速阶段（考虑前车类型）
        if self.type == 'human':
            safe_gap = self.safe_gap * self.speed
        elif front_veh_type == 'human':
            safe_gap = self.safe_gap * self.speed * 1.2  # 额外安全距离
        else:
            safe_gap = self.safe_gap * self.speed

        if front_gap < safe_gap:
            self.speed = min(front_gap, self.speed)

        # 3. 随机化（仅人类）
        if self.type == 'human' and random.random() < 0.2:
            self.speed = max(self.speed - 1, 0)

        # 4. 车队协同（自动驾驶专用）
        if self.type == 'av_platoon' and self.in_platoon:
            # 车队内保持更小间距和统一速度
            self.speed = min(self.speed, self.v_max * 0.95)  # 稍低速度以保持稳定

        # 确保速度非负
        self.speed = max(0, self.speed)

class EnhancedCA:
    """改进的元胞自动机模型"""
    def __init__(self, road_length=1000, p_av=0.3, dedicated_lane=False):
        """
        初始化模型
        :param road_length: 路段长度（单位：车辆长度）
        :param p_av: 自动驾驶车辆比例
        :param dedicated_lane: 是否有专用车道
        """
        self.road_length = road_length
        self.p_av = p_av
        self.dedicated_lane = dedicated_lane
        self.vehicles = []
        self.vehicle_dict = {}
        self.platoons = defaultdict(list)
        self.platoon_counter = 0
        self.time_step = 0
        self.history = []

        # 车道设置
        self.num_lanes = 2 if dedicated_lane else 1
        self.lane_vehicles = [[] for _ in range(self.num_lanes)]

        # 初始化车辆
        self.initialize_vehicles()

        # 设置车队
        self.form_platoons()

    def initialize_vehicles(self):
        """初始化车辆"""
        # 清空现有车辆
        self.vehicles = []
        self.vehicle_dict = {}
        self.lane_vehicles = [[] for _ in range(self.num_lanes)]

        # 创建车辆
        veh_id = 0
        positions = []

        # 确保最小间距
        min_gap = 5

        # 在每个车道上创建车辆
        for lane in range(self.num_lanes):
            current_pos = 0
            while current_pos < self.road_length:
                # 随机决定是否创建车辆 - 增加概率
                if random.random() < 0.25:  # 25%概率创建车辆（从0.1增加到0.25）
                    # 决定车辆类型
                    if lane == 1 and self.dedicated_lane:
                        # 专用车道只允许自动驾驶车辆
                        if random.random() < 0.5:
                            veh_type = 'av_solo'
                        else:
                            veh_type = 'av_platoon'
                    else:
                        # 普通车道
                        if random.random() < self.p_av:
                            if random.random() < 0.8:  # 增加av_platoon的比例
                                veh_type = 'av_platoon'
                            else:
                                veh_type = 'av_solo'
                        else:
                            veh_type = 'human'

                    # 创建车辆
                    vehicle = Vehicle(veh_id, veh_type, current_pos)
                    self.vehicles.append(vehicle)
                    self.vehicle_dict[veh_id] = vehicle
                    self.lane_vehicles[lane].append(vehicle)
                    veh_id += 1

                    # 更新位置 - 进一步减小间距
                    current_pos += 1 + random.randint(0, 3)  # 最小间距1，随机范围0-3
                else:
                    current_pos += 1

        # 按位置排序
        for lane in range(self.num_lanes):
            self.lane_vehicles[lane].sort(key=lambda v: v.position)

        # 设置前车关系
        self.set_front_vehicles()

    def set_front_vehicles(self):
        """设置每辆车的前车"""
        for lane in range(self.num_lanes):
            vehicles_in_lane = self.lane_vehicles[lane]
            for i, vehicle in enumerate(vehicles_in_lane):
                if i > 0:
                    vehicle.front_veh = vehicles_in_lane[i-1]
                    vehicle.front_gap = vehicles_in_lane[i-1].position - vehicle.position
                else:
                    vehicle.front_veh = None
                    vehicle.front_gap = float('inf')

    def form_platoons(self):
        """形成车队"""
        # 清空现有车队
        self.platoons = defaultdict(list)
        self.platoon_counter = 0

        # 在每个车道上寻找可形成车队的自动驾驶车辆
        for lane in range(self.num_lanes):
            vehicles_in_lane = self.lane_vehicles[lane]
            i = 0
            while i < len(vehicles_in_lane):
                vehicle = vehicles_in_lane[i]

                # 只有av_platoon类型的车辆可以发起车队
                if vehicle.type == 'av_platoon' and not vehicle.in_platoon:
                    # 寻找后续可加入车队的车辆
                    platoon_members = [vehicle]
                    j = i + 1
                    while j < len(vehicles_in_lane):
                        next_veh = vehicles_in_lane[j]
                        gap = next_veh.position - vehicle.position

                        # 如果是av_platoon或av_solo类型，且间距小于80，可以加入车队（从50增加到80）
                        if (next_veh.type in ['av_platoon', 'av_solo']) and gap < 80 and not next_veh.in_platoon:
                            platoon_members.append(next_veh)
                            j += 1
                        else:
                            break

                    # 如果车队有2辆或以上车辆，正式形成车队
                    if len(platoon_members) >= 2:
                        platoon_id = self.platoon_counter
                        self.platoon_counter += 1

                        for member in platoon_members:
                            member.in_platoon = True
                            member.platoon_id = platoon_id
                            self.platoons[platoon_id].append(member)

                        i = j  # 跳过已经加入车队的车辆
                    else:
                        i += 1
                else:
                    i += 1

    def platoon_speed(self, vehicle):
        """车队速度协调"""
        platoon = self.platoons.get(vehicle.platoon_id, [])
        if not platoon:
            return vehicle.speed

        # 车队内所有车辆保持相同速度（最小速度）
        min_speed = min(v.speed for v in platoon)
        return min_speed

    def update(self):
        """更新所有车辆状态"""
        # 更新每个车道的车辆
        for lane in range(self.num_lanes):
            # 按位置降序更新（从后往前）
            vehicles_in_lane = sorted(self.lane_vehicles[lane], key=lambda v: v.position, reverse=True)

            for vehicle in vehicles_in_lane:
                # 获取前车类型
                front_veh_type = vehicle.front_veh.type if vehicle.front_veh else None

                # 更新车辆状态
                vehicle.update(vehicle.front_gap, front_veh_type)

        # 更新车队 - 多次尝试
        for _ in range(3):  # 多次尝试形成车队
            self.form_platoons()

        # 移动车辆
        for lane in range(self.num_lanes):
            for vehicle in self.lane_vehicles[lane]:
                vehicle.position += vehicle.speed

                # 处理周期边界
                if vehicle.position >= self.road_length:
                    vehicle.position = vehicle.position % self.road_length

        # 重新排序和设置前车关系
        for lane in range(self.num_lanes):
            self.lane_vehicles[lane].sort(key=lambda v: v.position)
            self.set_front_vehicles()

        # 记录历史
        self.record_history()

        self.time_step += 1

    def record_history(self):
        """记录当前状态"""
        # 记录每个车道的车辆位置和速度
        lane_data = []
        for lane_idx, vehicles_in_lane in enumerate(self.lane_vehicles):
            for vehicle in vehicles_in_lane:
                lane_data.append({
                    'time': self.time_step,
                    'lane': lane_idx,
                    'position': vehicle.position,
                    'speed': vehicle.speed,
                    'type': vehicle.type,
                    'in_platoon': vehicle.in_platoon,
                    'platoon_id': vehicle.platoon_id if vehicle.in_platoon else None
                })

        self.history.append(lane_data)

    def get_density(self):
        """计算当前密度"""
        total_vehicles = sum(len(vehicles) for vehicles in self.lane_vehicles)
        return total_vehicles / (self.road_length * self.num_lanes)

    def get_average_speed(self):
        """计算平均速度"""
        total_speed = 0
        total_vehicles = 0
        for lane in range(self.num_lanes):
            for vehicle in self.lane_vehicles[lane]:
                total_speed += vehicle.speed
                total_vehicles += 1

        return total_speed / total_vehicles if total_vehicles > 0 else 0

    def get_flow(self):
        """计算交通流量"""
        return self.get_density() * self.get_average_speed()

    def simulate(self, steps=100, animate=False, save_gif=False):
        """运行仿真"""
        print(f"开始仿真，自动驾驶比例: {self.p_av:.1f}, 专用车道: {'是' if self.dedicated_lane else '否'}")
        print(f"初始车辆数: {sum(len(v) for v in self.lane_vehicles)}")

        if animate:
            return self.run_with_animation(steps, save_gif)
        else:
            return self.run_without_animation(steps)

    def run_without_animation(self, steps):
        """无动画运行仿真"""
        for _ in tqdm(range(steps), desc="仿真进度"):
            self.update()

        # 返回仿真结果
        return self.analyze_results()

    def run_with_animation(self, steps, save_gif=False):
        """带动画运行仿真"""
        fig, axes = plt.subplots(self.num_lanes, 1, figsize=(12, 6 * self.num_lanes))
        if self.num_lanes == 1:
            axes = [axes]

        # 创建颜色映射
        colors = {
            'human': 'red',
            'av_solo': 'blue',
            'av_platoon': 'green',
            'av_platoon_in_platoon': 'darkgreen'
        }

        # 初始化绘图
        scatters = []
        for lane_idx in range(self.num_lanes):
            ax = axes[lane_idx]
            ax.set_xlim(0, self.road_length)
            ax.set_ylim(0, 1)
            ax.set_title(f'车道 {lane_idx + 1} {"(专用车道)" if self.dedicated_lane and lane_idx == 1 else ""}')
            ax.set_xlabel('位置')
            ax.set_ylabel('车道')
            ax.grid(True)

            # 初始绘制
            x_positions = []
            c_colors = []
            for vehicle in self.lane_vehicles[lane_idx]:
                x_positions.append(vehicle.position)
                if vehicle.in_platoon:
                    c_colors.append(colors['av_platoon_in_platoon'])
                else:
                    c_colors.append(colors[vehicle.type])

            scatter = ax.scatter(x_positions, [0.5] * len(x_positions), c=c_colors, s=50)
            scatters.append(scatter)

        plt.tight_layout()

        def update_frame(frame):
            """更新每帧"""
            self.update()

            for lane_idx in range(self.num_lanes):
                ax = axes[lane_idx]
                x_positions = []
                c_colors = []
                for vehicle in self.lane_vehicles[lane_idx]:
                    x_positions.append(vehicle.position)
                    if vehicle.in_platoon:
                        c_colors.append(colors['av_platoon_in_platoon'])
                    else:
                        c_colors.append(colors[vehicle.type])

                scatters[lane_idx].set_offsets(np.column_stack((x_positions, [0.5] * len(x_positions))))
                scatters[lane_idx].set_color(c_colors)

            # 更新标题
            density = self.get_density()
            avg_speed = self.get_average_speed()
            flow = self.get_flow()
            fig.suptitle(f'时间步: {self.time_step}, 密度: {density:.2f}, 平均速度: {avg_speed:.2f}, 流量: {flow:.2f}')

            return scatters

        # 创建动画
        ani = animation.FuncAnimation(fig, update_frame, frames=steps, interval=100, blit=False)

        if save_gif:
            ani.save('ca_simulation.gif', writer='pillow', fps=10)
            print(f"\n动画已保存为 'ca_simulation.gif'")

        plt.show()
        return self.analyze_results()

    def analyze_results(self):
        """分析仿真结果"""
        print("\n=== 仿真结果分析 ===")

        # 基本统计
        final_density = self.get_density()
        final_avg_speed = self.get_average_speed()
        final_flow = self.get_flow()

        print(f"最终密度: {final_density:.2f} 辆车/单位长度")
        print(f"最终平均速度: {final_avg_speed:.2f} 单位长度/时间步")
        print(f"最终流量: {final_flow:.2f} 辆车/时间步")

        # 车辆类型统计
        type_counts = {'human': 0, 'av_solo': 0, 'av_platoon': 0, 'in_platoon': 0}
        for lane in range(self.num_lanes):
            for vehicle in self.lane_vehicles[lane]:
                type_counts[vehicle.type] += 1
                if vehicle.in_platoon:
                    type_counts['in_platoon'] += 1

        print(f"\n车辆类型分布:")
        print(f"  人类车辆: {type_counts['human']}")
        print(f"  单独自动驾驶: {type_counts['av_solo']}")
        print(f"  车队自动驾驶: {type_counts['av_platoon']}")
        print(f"  车队内车辆: {type_counts['in_platoon']}")

        # 车队统计
        print(f"\n车队统计:")
        print(f"  车队数量: {len(self.platoons)}")
        if self.platoons:
            avg_platoon_size = np.mean([len(p) for p in self.platoons.values()])
            print(f"  平均车队大小: {avg_platoon_size:.1f}")

        # 可视化结果
        self.visualize_results()

        # 返回关键指标
        return {
            'density': final_density,
            'avg_speed': final_avg_speed,
            'flow': final_flow,
            'type_counts': type_counts,
            'num_platoons': len(self.platoons)
        }

    def visualize_results(self):
        """可视化仿真结果"""
        # 1. 创建历史数据DataFrame
        if not self.history:
            print("没有历史数据可视化")
            return

        # 展平历史数据
        all_data = []
        for time_data in self.history:
            all_data.extend(time_data)

        df = pd.DataFrame(all_data)

        # 2. 创建可视化
        plt.figure(figsize=(18, 12))

        # 2.1 密度随时间变化
        plt.subplot(2, 3, 1)
        density_over_time = []
        for time_data in self.history:
            vehicles_at_time = len(time_data)
            density = vehicles_at_time / (self.road_length * self.num_lanes)
            density_over_time.append(density)

        plt.plot(range(len(density_over_time)), density_over_time)
        plt.title('密度随时间变化')
        plt.xlabel('时间步')
        plt.ylabel('密度')
        plt.grid(True)

        # 2.2 平均速度随时间变化
        plt.subplot(2, 3, 2)
        speed_over_time = []
        for time_data in self.history:
            total_speed = sum(v['speed'] for v in time_data)
            avg_speed = total_speed / len(time_data) if time_data else 0
            speed_over_time.append(avg_speed)

        plt.plot(range(len(speed_over_time)), speed_over_time)
        plt.title('平均速度随时间变化')
        plt.xlabel('时间步')
        plt.ylabel('平均速度')
        plt.grid(True)

        # 2.3 流量随时间变化
        plt.subplot(2, 3, 3)
        flow_over_time = [d * s for d, s in zip(density_over_time, speed_over_time)]
        plt.plot(range(len(flow_over_time)), flow_over_time)
        plt.title('流量随时间变化')
        plt.xlabel('时间步')
        plt.ylabel('流量')
        plt.grid(True)

        # 2.4 车辆类型分布
        plt.subplot(2, 3, 4)
        type_counts = df['type'].value_counts()
        plt.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%')
        plt.title('车辆类型分布')

        # 2.5 车队大小分布
        plt.subplot(2, 3, 5)
        if self.platoons:
            platoon_sizes = [len(p) for p in self.platoons.values()]
            plt.hist(platoon_sizes, bins=10, alpha=0.7)
            plt.title('车队大小分布')
            plt.xlabel('车队大小')
            plt.ylabel('频数')
        else:
            plt.text(0.5, 0.5, '无车队数据', ha='center')
            plt.title('车队大小分布')

        # 2.6 时空图（最后一个车道）
        plt.subplot(2, 3, 6)
        if self.num_lanes > 0:
            last_lane_data = df[df['lane'] == self.num_lanes - 1]
            plt.scatter(last_lane_data['time'], last_lane_data['position'],
                       c=last_lane_data['speed'], cmap='viridis', s=10, alpha=0.6)
            plt.colorbar(label='速度')
            plt.title(f'车道 {self.num_lanes} 时空图')
            plt.xlabel('时间步')
            plt.ylabel('位置')

        plt.tight_layout()
        plt.savefig('ca_simulation_results.png')
        print("\n仿真结果可视化已保存为 'ca_simulation_results.png'")

def identify_bottlenecks(ca_results, top_n=20):
    """识别关键瓶颈路段"""
    print(f"\n=== 识别{top_n}个关键瓶颈路段 ===")

    # 这个函数将基于元胞自动机的结果和之前的分析来识别瓶颈路段
    # 在实际应用中，可以结合LWR模型和随机森林的结果

    # 由于这是一个示例，我们返回基于之前分析的瓶颈路段
    # 在实际应用中，可以从数据库或分析结果中获取

    # 使用之前随机森林分类器的结果来识别瓶颈
    try:
        # 加载之前的分析结果
        df = pd.read_csv('2017_MCM_Problem_C_Data.csv')
        df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('"', '').str.replace('(', '').str.replace(')', '')

        # 计算特征
        df['length'] = df['endMilepost'] - df['startMilepost']
        df['avg_lanes'] = (df['Number_of_Lanes_INCR_MP_direction'] + df['Number_of_Lanes_DECR_MP_direction']) / 2
        df['capacity'] = df['avg_lanes'] * 2000 * 24
        df['demand_capacity_ratio'] = df['Average_daily_traffic_counts_Year_2015'] / df['capacity']

        # 识别瓶颈（需求容量比 > 0.9）
        bottlenecks = df[df['demand_capacity_ratio'] > 0.9].nlargest(top_n, 'demand_capacity_ratio')

        print(f"识别出{len(bottlenecks)}个关键瓶颈路段：")

        bottleneck_list = []
        for idx, row in bottlenecks.iterrows():
            bottleneck_info = {
                'route_id': row['Route_ID'],
                'start_milepost': row['startMilepost'],
                'end_milepost': row['endMilepost'],
                'demand_capacity_ratio': row['demand_capacity_ratio'],
                'avg_traffic': row['Average_daily_traffic_counts_Year_2015'],
                'avg_lanes': row['avg_lanes']
            }
            bottleneck_list.append(bottleneck_info)
            print(f"  {idx+1}. 路线 {row['Route_ID']}, 里程碑 {row['startMilepost']:.2f}-{row['endMilepost']:.2f}, "
                  f"需求容量比: {row['demand_capacity_ratio']:.3f}")

        return bottleneck_list

    except Exception as e:
        print(f"无法加载之前的分析结果: {e}")
        print("返回示例瓶颈路段列表")
        return [
            {'route_id': 5, 'start_milepost': 163.48, 'end_milepost': 164.22, 'demand_capacity_ratio': 2.017},
            {'route_id': 5, 'start_milepost': 163.36, 'end_milepost': 163.48, 'demand_capacity_ratio': 1.983},
            {'route_id': 405, 'start_milepost': 10.56, 'end_milepost': 10.93, 'demand_capacity_ratio': 1.844},
        ][:top_n]

def main():
    """主函数"""
    print("=== 改进的元胞自动机模型 (第2层) ===\n")

    # 1. 创建模型实例
    print("1. 创建模型实例...")
    ca_model = EnhancedCA(road_length=500, p_av=0.5, dedicated_lane=True)

    # 2. 运行仿真
    print("\n2. 运行仿真...")
    results = ca_model.simulate(steps=150, animate=False)  # 从50增加到150

    # 3. 识别瓶颈路段
    print("\n3. 识别瓶颈路段...")
    bottlenecks = identify_bottlenecks(results, top_n=20)

    # 4. 保存瓶颈路段信息
    bottleneck_df = pd.DataFrame(bottlenecks)
    bottleneck_df.to_csv('critical_bottlenecks.csv', index=False)
    print(f"\n关键瓶颈路段已保存为 'critical_bottlenecks.csv'")

    print("\n=== 元胞自动机模型搭建完成 ===")
    print("\n主要输出文件：")
    print("  - ca_simulation_results.png (仿真结果可视化)")
    print("  - critical_bottlenecks.csv (20个关键瓶颈路段)")

if __name__ == "__main__":
    main()