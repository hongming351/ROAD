# network_equilibrium_model_fixed.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import json
import warnings
import os

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================================
# 第一部分：修复的物理网络类
# ============================================================================

class PhysicallyCorrectNetwork:
    """修复的物理正确交通网络"""
    
    def __init__(self, data_file='2017_MCM_Problem_C_Data.csv'):
        self.df = self.load_and_validate_data(data_file)
        self.G = self.build_physical_network()
        self.ensure_network_connectivity()  # 确保网络连通性
        self.od_pairs = self.create_realistic_od_pairs()
        
        # 验证物理合理性
        self.validate_network_physics()
    
    def load_and_validate_data(self, data_file):
        """加载并验证数据"""
        print("加载并验证道路数据...")
        try:
            df = pd.read_csv(data_file, encoding='utf-8-sig')
        except:
            df = pd.read_csv(data_file)
        
        if len(df.columns) >= 8:
            df.columns = ['Route_ID', 'startMilepost', 'endMilepost', 
                         'ADT', 'RteType', 'Lanes_DECR', 'Lanes_INCR', 
                         'Unnamed', 'Comments']
        
        # 物理验证
        print(f"数据验证:")
        print(f"  路段数: {len(df)}")
        print(f"  ADT范围: {df['ADT'].min():.0f} - {df['ADT'].max():.0f} 辆/天")
        print(f"  车道数范围: {df['Lanes_INCR'].min():.0f} - {df['Lanes_INCR'].max():.0f}")
        
        # 计算物理合理的流量
        df['length_km'] = (df['endMilepost'] - df['startMilepost']) * 1.60934
        
        # 关键：使用物理合理的高峰小时系数
        # 典型值：高峰小时流量 ≈ 日流量的8-12%
        peak_hour_factor = 0.10  # 10%
        df['peak_hour_flow'] = df['ADT'] * peak_hour_factor
        
        # 方向分布：典型55%-45%
        df['peak_incr'] = df['peak_hour_flow'] * 0.55
        df['peak_decr'] = df['peak_hour_flow'] * 0.45
        
        # 处理NaN
        df['lanes_incr'] = df['Lanes_INCR'].fillna(1).astype(int)
        df['lanes_decr'] = df['Lanes_DECR'].fillna(1).astype(int)
        
        # 验证v/c比
        df['capacity_incr'] = df['lanes_incr'] * 2200
        df['capacity_decr'] = df['lanes_decr'] * 2200
        df['vc_incr'] = df['peak_incr'] / df['capacity_incr']
        df['vc_decr'] = df['peak_decr'] / df['capacity_decr']
        
        print(f"\n流量统计:")
        print(f"  平均v/c比 (INCR): {df['vc_incr'].mean():.3f}")
        print(f"  平均v/c比 (DECR): {df['vc_decr'].mean():.3f}")
        print(f"  最大v/c比: {max(df['vc_incr'].max(), df['vc_decr'].max()):.3f}")
        print(f"  拥堵路段数 (v/c > 0.8): {(df['vc_incr'] > 0.8).sum() + (df['vc_decr'] > 0.8).sum()}")
        
        return df
    
    def build_physical_network(self):
        """构建物理合理的网络"""
        print("\n构建物理网络...")
        G = nx.DiGraph()
        
        # 只处理主要高速公路
        major_routes = [5, 90, 405, 520]
        
        for _, row in self.df.iterrows():
            route_id = int(row['Route_ID'])
            if route_id in major_routes:
                # INCR方向
                if row['lanes_incr'] > 0:
                    start_node = f"{route_id}_{row['startMilepost']:.2f}"
                    end_node = f"{route_id}_{row['endMilepost']:.2f}"
                    
                    # 物理合理的初始流量
                    base_flow = row['peak_incr'] * 0.8  # 使用高峰流量的80%作为初始
                    
                    G.add_edge(start_node, end_node,
                              length=row['length_km'],
                              lanes=row['lanes_incr'],
                              base_flow=base_flow,
                              free_time=row['length_km'] / 96.56,  # 60mph
                              capacity=row['lanes_incr'] * 2200,
                              route_id=route_id,
                              vc_ratio=row['vc_incr'])
                
                # DECR方向
                if row['lanes_decr'] > 0:
                    start_node = f"{route_id}_{row['endMilepost']:.2f}"
                    end_node = f"{route_id}_{row['startMilepost']:.2f}"
                    
                    base_flow = row['peak_decr'] * 0.8
                    
                    G.add_edge(start_node, end_node,
                              length=row['length_km'],
                              lanes=row['lanes_decr'],
                              base_flow=base_flow,
                              free_time=row['length_km'] / 96.56,
                              capacity=row['lanes_decr'] * 2200,
                              route_id=route_id,
                              vc_ratio=row['vc_decr'])
        
        print(f"初始网络统计:")
        print(f"  节点数: {G.number_of_nodes()}")
        print(f"  边数: {G.number_of_edges()}")
        
        return G
    
    def ensure_network_connectivity(self):
        """确保网络连通性"""
        print("\n确保网络连通性...")
        
        if not nx.is_strongly_connected(self.G):
            print("  网络不连通，添加连接边...")
            
            # 获取所有节点
            all_nodes = list(self.G.nodes())
            
            # 按路线和里程排序节点
            nodes_by_route = {}
            for node in all_nodes:
                parts = node.split('_')
                if len(parts) == 2:
                    route, milepost = parts[0], float(parts[1])
                    if route not in nodes_by_route:
                        nodes_by_route[route] = []
                    nodes_by_route[route].append((milepost, node))
            
            # 在每个路线上按里程排序节点
            for route in nodes_by_route:
                nodes_by_route[route].sort()
            
            # 添加路线内部的连接（确保路线是连续的）
            for route, nodes in nodes_by_route.items():
                if len(nodes) > 1:
                    for i in range(len(nodes) - 1):
                        mile1, node1 = nodes[i]
                        mile2, node2 = nodes[i + 1]
                        
                        # 如果节点之间没有边，添加一条
                        if not self.G.has_edge(node1, node2):
                            # 计算距离
                            distance_km = abs(mile2 - mile1) * 1.60934
                            self.G.add_edge(node1, node2,
                                          length=distance_km,
                                          lanes=2,  # 默认2车道
                                          base_flow=500,
                                          free_time=distance_km / 96.56,
                                          capacity=2 * 2200,
                                          route_id=int(route),
                                          is_connector=True)
            
            # 添加路线之间的连接（在已知的交汇点）
            interchanges = [
                # (route1, mile1, route2, mile2)
                (5, 165.29, 90, 2.79),
                (5, 182.59, 405, 30.32),
                (405, 11.69, 90, 12.34),
                (90, 2.79, 5, 165.29),  # 反向
                (405, 30.32, 5, 182.59),  # 反向
                (90, 12.34, 405, 11.69),   # 反向
                # 添加更多连接确保完全连通
                (5, 182.59, 90, 2.79),
                (90, 2.79, 405, 11.69),
                (405, 30.32, 90, 12.34)
            ]
            
            for r1, m1, r2, m2 in interchanges:
                node1 = f"{r1}_{m1:.2f}"
                node2 = f"{r2}_{m2:.2f}"
                
                # 如果节点不存在，创建它
                if node1 not in self.G:
                    self.G.add_node(node1, route=r1, milepost=m1)
                if node2 not in self.G:
                    self.G.add_node(node2, route=r2, milepost=m2)
                
                # 添加双向连接
                if not self.G.has_edge(node1, node2):
                    self.G.add_edge(node1, node2,
                                  length=0.5,  # 交汇匝道约0.5km
                                  lanes=1,
                                  base_flow=200,
                                  free_time=0.5 / 48.28,  # 30mph
                                  capacity=1 * 2200,
                                  route_id=0,  # 交汇匝道
                                  is_connector=True,
                                  is_interchange=True)
                
                # 添加反向连接
                if not self.G.has_edge(node2, node1):
                    self.G.add_edge(node2, node1,
                                  length=0.5,
                                  lanes=1,
                                  base_flow=200,
                                  free_time=0.5 / 48.28,
                                  capacity=1 * 2200,
                                  route_id=0,
                                  is_connector=True,
                                  is_interchange=True)
            
            # 再次检查连通性
            if nx.is_strongly_connected(self.G):
                print("  网络已连通")
            else:
                print("  警告: 网络仍不连通，尝试最后修复...")
                # 强制连接所有不相交的组件
                components = list(nx.strongly_connected_components(self.G))
                if len(components) > 1:
                    for i in range(len(components)-1):
                        comp1 = list(components[i])
                        comp2 = list(components[i+1])
                        if comp1 and comp2:
                            # 连接第一个组件的一个节点到第二个组件的一个节点
                            node1 = comp1[0]
                            node2 = comp2[0]
                            self.G.add_edge(node1, node2,
                                          length=1.0,
                                          lanes=1,
                                          base_flow=100,
                                          free_time=1.0 / 48.28,
                                          capacity=1 * 2200,
                                          route_id=999,
                                          is_connector=True,
                                          is_forced=True)
        
        else:
            print("  网络已连通")
        
        # 最终连通性检查
        if nx.is_strongly_connected(self.G):
            print("  ✓ 网络完全连通")
        else:
            print("  ✗ 警告: 网络仍不连通")
        
        print(f"连通后网络统计:")
        print(f"  节点数: {self.G.number_of_nodes()}")
        print(f"  边数: {self.G.number_of_edges()}")
        
        # 计算网络总容量和流量
        total_capacity = sum(data['capacity'] for _, _, data in self.G.edges(data=True))
        total_base_flow = sum(data['base_flow'] for _, _, data in self.G.edges(data=True))
        network_vc = total_base_flow / total_capacity if total_capacity > 0 else 0
        
        print(f"  网络总容量: {total_capacity:,.0f} 辆/小时")
        print(f"  网络总基础流量: {total_base_flow:,.0f} 辆/小时")
        print(f"  网络平均v/c比: {network_vc:.3f}")
    
    def create_realistic_od_pairs(self):
        """创建现实OD对（使用文献弹性系数）"""
        print("\n创建现实OD对...")
        od_pairs = []
        
        # 基于路段流量估算OD需求
        for route in [5, 90, 405]:
            route_edges = [(u, v, data) for u, v, data in self.G.edges(data=True) 
                          if data.get('route_id') == route]
            
            if route_edges:
                # 估算该路线的总需求
                total_flow = sum(data['base_flow'] for _, _, data in route_edges)
                
                # 获取该路线的节点
                nodes = []
                for u, v, data in route_edges:
                    if u not in nodes:
                        nodes.append(u)
                    if v not in nodes:
                        nodes.append(v)
                
                # 按里程排序
                def get_milepost(node):
                    parts = node.split('_')
                    if len(parts) == 2:
                        return float(parts[1])
                    return 0
                
                nodes.sort(key=get_milepost)
                
                if len(nodes) >= 2:
                    # 主要OD对：起点到终点
                    od_pairs.append({
                        'origin': nodes[0],
                        'destination': nodes[-1],
                        'base_demand': total_flow * 0.2,  # 占总流量的20%
                        'elasticity': -0.25  # 文献值
                    })
                    
                    # 中间OD对
                    if len(nodes) >= 4:
                        mid = len(nodes) // 2
                        od_pairs.append({
                            'origin': nodes[0],
                            'destination': nodes[mid],
                            'base_demand': total_flow * 0.15,
                            'elasticity': -0.20
                        })
        
        # 交叉口OD对
        cross_ods = [
            ('5_165.29', '90_2.79', 500, -0.2),
            ('5_182.59', '405_30.32', 400, -0.2),
            ('405_11.69', '90_12.34', 300, -0.2),
            ('90_2.79', '5_165.29', 450, -0.2),  # 反向
            ('405_30.32', '5_182.59', 350, -0.2),  # 反向
            ('90_12.34', '405_11.69', 250, -0.2)   # 反向
        ]
        
        for origin, destination, demand, elasticity in cross_ods:
            if origin in self.G and destination in self.G:
                od_pairs.append({
                    'origin': origin,
                    'destination': destination,
                    'base_demand': demand,
                    'elasticity': elasticity
                })
        
        total_demand = sum(od['base_demand'] for od in od_pairs)
        print(f"创建了 {len(od_pairs)} 个OD对")
        print(f"总OD需求: {total_demand:,.0f} 辆/小时")
        
        if self.G.number_of_edges() > 0:
            total_flow = sum(data['base_flow'] for _, _, data in self.G.edges(data=True))
            if total_flow > 0:
                print(f"占网络总流量: {(total_demand/total_flow*100):.1f}%")
        
        return od_pairs
    
    def validate_network_physics(self):
        """验证网络物理合理性"""
        print("\n网络物理验证:")
        
        # 1. 检查v/c比合理性
        vc_ratios = []
        for u, v, data in self.G.edges(data=True):
            if data['capacity'] > 0:
                vc = data['base_flow'] / data['capacity']
                vc_ratios.append(vc)
        
        if vc_ratios:
            avg_vc = np.mean(vc_ratios)
            max_vc = max(vc_ratios)
            congested = sum(1 for vc in vc_ratios if vc > 0.8)
            
            print(f"  平均v/c比: {avg_vc:.3f}")
            print(f"  最大v/c比: {max_vc:.3f}")
            print(f"  拥堵路段数(v/c>0.8): {congested}")
            
            if max_vc > 1.5:
                print(f"  警告: 最大v/c比{max_vc:.2f}过高，将进行调整...")
                self.adjust_network_flows()
        
        # 2. 检查连通性
        if nx.is_strongly_connected(self.G):
            print("  网络连通性: 良好")
        else:
            print("  警告: 网络仍不连通")
    
    def adjust_network_flows(self):
        """调整网络流量使其物理合理"""
        print("调整网络流量...")
        
        for u, v, data in self.G.edges(data=True):
            capacity = data['capacity']
            current_flow = data['base_flow']
            
            if capacity > 0 and current_flow / capacity > 1.2:
                # 将v/c比限制在1.2以内
                data['base_flow'] = capacity * 1.2


# ============================================================================
# 第二部分：紧急修复的物理均衡模型
# ============================================================================

class PhysicallyCorrectEquilibrium_FIXED:
    """紧急修复的物理正确均衡模型"""
    
    def __init__(self, network: PhysicallyCorrectNetwork):
        self.G = network.G
        self.od_pairs = network.od_pairs
        
        # 物理参数
        self.base_capacity_per_lane = 2200
        self.free_flow_speed = 96.56
        
        # BPR参数
        self.alpha = 0.15
        self.beta = 4
        
        # 自动驾驶参数
        self.av_efficiency_gain = 0.8  # p=1时提升80%
        
        # 需求参数
        self.av_attraction_factor = 0.05  # 更保守的自动驾驶吸引力
        
        # 缓存
        self.cache = {}
    
    def calculate_capacity(self, lanes: int, p: float, dedicated: bool = False) -> float:
        """简化的容量计算"""
        base_capacity = lanes * self.base_capacity_per_lane
        
        if dedicated and lanes >= 2:
            # 专用车道：1条给AV，其余给HV
            av_capacity = 1 * self.base_capacity_per_lane * (1.0 + p)  # p=1时提升100%
            hv_capacity = (lanes - 1) * self.base_capacity_per_lane * (1.0 + 0.3 * p)  # 协同效应
            return av_capacity + hv_capacity
        else:
            # 混合交通：S型增长
            efficiency = 1.0 + self.av_efficiency_gain * (p ** 0.8)
            return base_capacity * efficiency
    
    def travel_time(self, flow: float, capacity: float, free_time: float) -> float:
        """BPR函数"""
        if capacity <= 0:
            return free_time * 10
        
        ratio = min(flow / max(capacity, 1), 2.0)  # v/c上限2.0
        return free_time * (1.0 + self.alpha * (ratio ** self.beta))
    
    def calculate_demand(self, base_demand: float, p: float, 
                        travel_time_factor: float, elasticity: float) -> float:
        """修复的需求计算"""
        # 自动驾驶吸引力（非常保守）
        av_effect = 1.0 + self.av_attraction_factor * p
        
        # 拥堵抑制效应
        # travel_time_factor = 当前旅行时间 / 自由流时间
        # 限制在合理范围
        travel_time_factor = min(max(travel_time_factor, 1.0), 3.0)  # 1.0-3.0倍
        
        # 弹性系数（负值）
        # 使用指数形式确保为正
        congestion_effect = max(0.5, travel_time_factor ** elasticity)  # 负弹性，所以指数 < 1
        
        demand = base_demand * av_effect * congestion_effect
        
        # 物理限制
        return np.clip(demand, base_demand * 0.5, base_demand * 1.5)
    
    def find_all_shortest_paths(self, travel_times):
        """找到所有OD对的最短路径"""
        paths = {}
        path_times = {}
        
        for od in self.od_pairs:
            origin = od['origin']
            destination = od['destination']
            
            if origin in self.G and destination in self.G:
                try:
                    # 使用networkx的最短路径算法
                    path = nx.shortest_path(self.G, origin, destination, 
                                          weight=lambda u, v, d: travel_times.get((u, v), 1e6))
                    
                    # 计算路径总时间
                    total_time = 0
                    free_time = 0
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i+1]
                        key = (u, v)
                        if key in travel_times:
                            total_time += travel_times[key]
                        
                        if self.G.has_edge(u, v):
                            free_time += self.G[u][v].get('free_time', 1.0)
                    
                    paths[(origin, destination)] = path
                    path_times[(origin, destination)] = (total_time, free_time)
                    
                except nx.NetworkXNoPath:
                    # 如果找不到路径，尝试添加虚拟边
                    print(f"    警告: {origin} -> {destination} 无直接路径，使用虚拟连接")
                    # 可以添加虚拟路径，这里简化处理
                    continue
        
        return paths, path_times
    
    def solve_equilibrium_fixed(self, p: float = 0, 
                              dedicated_edges: list = None,
                              max_iter: int = 30,
                              tol: float = 0.005) -> dict:
        """
        紧急修复的均衡求解算法
        使用Frank-Wolfe算法（标准的UE算法）
        """
        if dedicated_edges is None:
            dedicated_edges = []
        
        cache_key = (p, tuple(sorted(dedicated_edges)))
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        print(f"  均衡求解 (p={p:.1f}):")
        
        # 初始化：每个路段使用基础流量
        edge_flows = {}
        edge_data = {}
        
        for u, v, data in self.G.edges(data=True):
            key = (u, v)
            edge_flows[key] = data.get('base_flow', 100)
            edge_data[key] = data
        
        # Frank-Wolfe算法
        convergence_history = []
        
        for iteration in range(max_iter):
            # 1. 基于当前流量计算旅行时间
            travel_times = {}
            capacities = {}
            
            for key, flow in edge_flows.items():
                data = edge_data[key]
                lanes = data.get('lanes', 1)
                dedicated = key in dedicated_edges
                capacity = self.calculate_capacity(lanes, p, dedicated)
                capacities[key] = capacity
                travel_times[key] = self.travel_time(flow, capacity, data['free_time'])
            
            # 2. 全有全无分配（所有OD对同时）
            aux_flows = {key: 0.0 for key in edge_flows}
            total_demand = 0
            
            # 找到所有最短路径
            paths, path_times = self.find_all_shortest_paths(travel_times)
            
            # 分配需求到最短路径
            for (origin, destination), path in paths.items():
                # 找到对应的OD对数据
                od_info = None
                for od in self.od_pairs:
                    if od['origin'] == origin and od['destination'] == destination:
                        od_info = od
                        break
                
                if od_info is None:
                    continue
                
                # 获取路径时间和自由流时间
                total_time, free_time = path_times[(origin, destination)]
                
                # 计算旅行时间因子
                if free_time > 0:
                    time_factor = total_time / free_time
                else:
                    time_factor = 1.0
                
                # 计算弹性需求
                demand = self.calculate_demand(
                    od_info['base_demand'], p, time_factor, od_info.get('elasticity', -0.2))
                
                # 分配流量到路径上的每条边
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    key = (u, v)
                    if key in aux_flows:
                        aux_flows[key] += demand
                
                total_demand += demand
            
            # 3. 计算搜索方向
            direction = {}
            for key in edge_flows:
                direction[key] = aux_flows[key] - edge_flows[key]
            
            # 4. 线搜索找到最佳步长
            def objective(alpha):
                total_cost = 0
                for key in edge_flows:
                    test_flow = edge_flows[key] + alpha * direction[key]
                    test_flow = max(test_flow, 1.0)
                    
                    data = edge_data[key]
                    lanes = data.get('lanes', 1)
                    dedicated = key in dedicated_edges
                    capacity = self.calculate_capacity(lanes, p, dedicated)
                    time_val = self.travel_time(test_flow, capacity, data['free_time'])
                    
                    total_cost += test_flow * time_val
                return total_cost
            
            # 尝试多个步长
            alphas = np.linspace(0, 1, 11)
            costs = [objective(a) for a in alphas]
            best_alpha = alphas[np.argmin(costs)]
            
            # 5. 更新流量
            total_change = 0
            for key in edge_flows:
                old_flow = edge_flows[key]
                new_flow = old_flow + best_alpha * direction[key]
                
                # 强制物理限制
                data = edge_data[key]
                lanes = data.get('lanes', 1)
                dedicated = key in dedicated_edges
                capacity = self.calculate_capacity(lanes, p, dedicated)
                new_flow = min(new_flow, capacity * 2.0)  # v/c ≤ 2.0
                new_flow = max(new_flow, 1.0)
                
                edge_flows[key] = new_flow
                total_change += abs(new_flow - old_flow)
            
            # 6. 计算收敛性
            total_flow = sum(edge_flows.values())
            if total_flow > 0:
                relative_change = total_change / total_flow
                convergence_history.append(relative_change)
                
                # 计算当前总旅行时间
                current_total_time = 0
                for key, flow in edge_flows.items():
                    data = edge_data[key]
                    lanes = data.get('lanes', 1)
                    dedicated = key in dedicated_edges
                    capacity = self.calculate_capacity(lanes, p, dedicated)
                    time_val = self.travel_time(flow, capacity, data['free_time'])
                    current_total_time += flow * time_val
                
                if iteration % 5 == 0:
                    # 计算平均v/c
                    avg_vc = 0
                    count = 0
                    for key, flow in edge_flows.items():
                        data = edge_data[key]
                        lanes = data.get('lanes', 1)
                        base_capacity = lanes * 2200
                        if base_capacity > 0:
                            avg_vc += flow / base_capacity
                            count += 1
                    
                    avg_vc = avg_vc / count if count > 0 else 0
                    
                    print(f"    迭代 {iteration:2d}: 时间={current_total_time:8.1f}h, "
                          f"流量={total_flow:7.0f}, v/c={avg_vc:.3f}, 变化={relative_change:.3%}")
                
                # 收敛检查
                if iteration > 5 and relative_change < tol:
                    print(f"    在 {iteration+1} 次迭代收敛")
                    break
                
                if iteration == max_iter - 1:
                    print(f"    达到最大迭代次数 {max_iter}")
            
            else:
                print("    警告: 总流量为0")
                break
        
        # 计算最终指标
        result = self.calculate_metrics_fixed(edge_flows, p, dedicated_edges)
        self.cache[cache_key] = result
        return result
    
    def calculate_metrics_fixed(self, edge_flows, p, dedicated_edges):
        """计算性能指标"""
        total_time = 0
        total_flow = 0
        total_distance = 0
        vc_ratios = []
        
        dedicated_edges_set = set(dedicated_edges) if dedicated_edges else set()
        
        for (u, v), flow in edge_flows.items():
            if self.G.has_edge(u, v):
                data = self.G[u][v]
                lanes = data.get('lanes', 1)
                dedicated = (u, v) in dedicated_edges_set
                capacity = self.calculate_capacity(lanes, p, dedicated)
                time_val = self.travel_time(flow, capacity, data['free_time'])
                
                total_time += flow * time_val
                total_flow += flow
                total_distance += flow * data.get('length', 0)
                
                # 计算基础v/c
                base_capacity = lanes * 2200
                if base_capacity > 0:
                    vc_ratios.append(flow / base_capacity)
        
        # 统计
        avg_speed = total_distance / total_time if total_time > 0 else 0
        avg_vc = np.mean(vc_ratios) if vc_ratios else 0
        max_vc = max(vc_ratios) if vc_ratios else 0
        min_vc = min(vc_ratios) if vc_ratios else 0
        
        # 拥堵统计
        congested = sum(1 for vc in vc_ratios if vc > 0.8)
        congestion_pct = congested / len(vc_ratios) * 100 if vc_ratios else 0
        
        return {
            'edge_flows': edge_flows,
            'total_time': total_time,
            'total_flow': total_flow,
            'avg_speed': avg_speed,
            'avg_vc_ratio': avg_vc,
            'max_vc_ratio': max_vc,
            'min_vc_ratio': min_vc,
            'congestion_percent': congestion_pct,
            'total_distance': total_distance
        }


# ============================================================================
# 第三部分：简化政策分析器
# ============================================================================

class SimplePolicyAnalyzer:
    """简化政策分析器"""
    
    def __init__(self, equilibrium_model):
        self.eq_model = equilibrium_model
        self.G = equilibrium_model.G
    
    def analyze_av_impact(self):
        """分析自动驾驶影响"""
        print("\n" + "="*60)
        print("自动驾驶影响分析")
        print("="*60)
        
        p_values = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        results = []
        
        base_result = None
        
        for p in p_values:
            print(f"\n分析 p={p}:")
            result = self.eq_model.solve_equilibrium_fixed(p=p)
            
            if p == 0:
                base_result = result
            
            improvement = 0
            if base_result is not None and base_result['total_time'] > 0:
                improvement = (base_result['total_time'] - result['total_time']) / base_result['total_time'] * 100
            
            results.append({
                'p': p,
                'total_time': result['total_time'],
                'avg_speed': result['avg_speed'],
                'total_flow': result['total_flow'],
                'avg_vc_ratio': result['avg_vc_ratio'],
                'max_vc_ratio': result['max_vc_ratio'],
                'congestion_percent': result['congestion_percent'],
                'improvement_pct': improvement
            })
            
            print(f"  总旅行时间: {result['total_time']:.1f}h")
            print(f"  平均速度: {result['avg_speed']:.1f}km/h")
            print(f"  总流量: {result['total_flow']:.0f}")
            print(f"  平均v/c: {result['avg_vc_ratio']:.3f}")
            print(f"  拥堵路段: {result['congestion_percent']:.1f}%")
            if p > 0:
                print(f"  改善: {improvement:.1f}%")
        
        return pd.DataFrame(results)
    
    def analyze_dedicated_lanes(self, p=0.5):
        """分析专用车道"""
        print(f"\n" + "="*60)
        print(f"专用车道分析 (p={p})")
        print("="*60)
        
        # 基准情况
        print("\n基准情况 (无专用车道):")
        base_result = self.eq_model.solve_equilibrium_fixed(p=p)
        base_time = base_result['total_time']
        base_speed = base_result['avg_speed']
        base_vc = base_result['avg_vc_ratio']
        base_congestion = base_result['congestion_percent']
        
        print(f"  总旅行时间: {base_time:.1f}h")
        print(f"  平均速度: {base_speed:.1f}km/h")
        print(f"  平均v/c: {base_vc:.3f}")
        print(f"  拥堵路段: {base_congestion:.1f}%")
        
        # 识别关键路段
        print("\n识别关键拥堵路段...")
        edge_flows = base_result['edge_flows']
        
        congested_edges = []
        for (u, v), flow in edge_flows.items():
            if self.G.has_edge(u, v):
                data = self.G[u][v]
                lanes = data.get('lanes', 1)
                base_capacity = lanes * 2200
                if base_capacity > 0:
                    vc = flow / base_capacity
                    if vc > 0.8 and lanes >= 2:  # 拥堵且至少有2车道
                        congested_edges.append({
                            'edge': (u, v),
                            'vc': vc,
                            'lanes': lanes,
                            'flow': flow,
                            'length': data.get('length', 0)
                        })
        
        # 按拥堵程度排序
        congested_edges.sort(key=lambda x: x['vc'], reverse=True)
        
        print(f"识别了 {len(congested_edges)} 个拥堵路段 (v/c > 0.8, 车道≥2)")
        
        if congested_edges:
            print("  最拥堵的5个路段:")
            for i, edge in enumerate(congested_edges[:5]):
                print(f"    {i+1}. {edge['edge']}: v/c={edge['vc']:.3f}, "
                      f"{edge['lanes']}车道, 流量={edge['flow']:.0f}")
        
        # 分析专用车道方案
        results = []
        
        for n_lanes in [1, 2, 3]:
            if n_lanes <= len(congested_edges):
                dedicated_edges = [e['edge'] for e in congested_edges[:n_lanes]]
                
                print(f"\n方案 {n_lanes}: {n_lanes}条专用车道")
                print(f"  位置: {[str(e) for e in dedicated_edges[:2]]}")
                
                av_result = self.eq_model.solve_equilibrium_fixed(
                    p=p, dedicated_edges=dedicated_edges)
                
                av_time = av_result['total_time']
                av_speed = av_result['avg_speed']
                av_vc = av_result['avg_vc_ratio']
                av_congestion = av_result['congestion_percent']
                
                # 效益计算
                time_saving = base_time - av_time
                time_saving_pct = (time_saving / base_time * 100) if base_time > 0 else 0
                speed_improvement = (av_speed - base_speed) / base_speed * 100 if base_speed > 0 else 0
                congestion_reduction = base_congestion - av_congestion
                
                print(f"  总旅行时间: {av_time:.1f}h (节省 {time_saving_pct:.2f}%)")
                print(f"  平均速度: {av_speed:.1f}km/h (提升 {speed_improvement:.2f}%)")
                print(f"  拥堵路段减少: {congestion_reduction:.1f}%")
                
                # 简单成本效益分析
                total_length = 0
                for edge in dedicated_edges:
                    if self.G.has_edge(*edge):
                        total_length += self.G[edge[0]][edge[1]]['length']
                
                # 参数
                cost_per_lane_km = 1.0e6  # 100万美元/车道·公里
                vot = 20  # 20美元/小时
                
                # 成本
                construction_cost = n_lanes * total_length * cost_per_lane_km
                
                # 效益（只考虑工作日）
                daily_time_saving = time_saving
                annual_benefit = daily_time_saving * vot * 250
                
                # 简单B/C比
                bc_ratio = annual_benefit / construction_cost if construction_cost > 0 else 0
                
                results.append({
                    'n_lanes': n_lanes,
                    'time_saving_pct': time_saving_pct,
                    'speed_improvement_pct': speed_improvement,
                    'congestion_reduction_pct': congestion_reduction,
                    'construction_cost_million': construction_cost / 1e6,
                    'annual_benefit_million': annual_benefit / 1e6,
                    'bc_ratio': bc_ratio
                })
        
        return pd.DataFrame(results)


# ============================================================================
# 第四部分：主函数和可视化
# ============================================================================

def create_simple_visualization(av_impact_df, lane_analysis_df):
    """创建简单可视化"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 总旅行时间变化
    ax1 = axes[0, 0]
    if not av_impact_df.empty:
        ax1.plot(av_impact_df['p'], av_impact_df['total_time'], 'b-o', linewidth=2)
        ax1.set_xlabel('自动驾驶比例 (p)')
        ax1.set_ylabel('总旅行时间 (小时)')
        ax1.set_title('总旅行时间随p变化')
        ax1.grid(True, alpha=0.3)
        
        # 标注改善百分比
        if 'improvement_pct' in av_impact_df.columns:
            for i, row in av_impact_df.iterrows():
                if row['p'] > 0:
                    ax1.annotate(f"{row['improvement_pct']:.1f}%", 
                                xy=(row['p'], row['total_time']),
                                xytext=(0, 10), textcoords='offset points',
                                ha='center', fontsize=9)
    
    # 2. 平均速度
    ax2 = axes[0, 1]
    if not av_impact_df.empty:
        ax2.plot(av_impact_df['p'], av_impact_df['avg_speed'], 'g-s', linewidth=2)
        ax2.set_xlabel('自动驾驶比例 (p)')
        ax2.set_ylabel('平均速度 (km/h)')
        ax2.set_title('平均速度随p变化')
        ax2.grid(True, alpha=0.3)
    
    # 3. 专用车道时间节省
    ax3 = axes[1, 0]
    if not lane_analysis_df.empty and 'time_saving_pct' in lane_analysis_df.columns:
        x = np.arange(len(lane_analysis_df))
        bars = ax3.bar(x, lane_analysis_df['time_saving_pct'], 
                      color=['lightblue', 'lightgreen', 'lightcoral'])
        ax3.set_xlabel('专用车道数量')
        ax3.set_ylabel('时间节省 (%)')
        ax3.set_title('专用车道效益 (p=0.5)')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'{int(n)}条' for n in lane_analysis_df['n_lanes']])
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 标注数值
        for bar, val in zip(bars, lane_analysis_df['time_saving_pct']):
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                    f'{val:.2f}%', ha='center', va='bottom')
    
    # 4. 成本效益比
    ax4 = axes[1, 1]
    if not lane_analysis_df.empty and 'bc_ratio' in lane_analysis_df.columns:
        x = np.arange(len(lane_analysis_df))
        colors = ['green' if x > 1.0 else 'red' for x in lane_analysis_df['bc_ratio']]
        bars = ax4.bar(x, lane_analysis_df['bc_ratio'], color=colors)
        ax4.axhline(y=1.0, color='k', linestyle='--', linewidth=1)
        ax4.set_xlabel('专用车道数量')
        ax4.set_ylabel('成本效益比 (B/C)')
        ax4.set_title('经济可行性')
        ax4.set_xticks(x)
        ax4.set_xticklabels([f'{int(n)}条' for n in lane_analysis_df['n_lanes']])
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 标注可行性
        for bar, ratio in zip(bars, lane_analysis_df['bc_ratio']):
            decision = "可行" if ratio > 1.0 else "不可行"
            color = 'green' if ratio > 1.0 else 'red'
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                    decision, ha='center', va='bottom', fontweight='bold', color=color)
    
    plt.suptitle('修复的网络均衡分析结果', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig

def generate_recommendations(av_impact_df, lane_analysis_df):
    """生成建议"""
    print("\n" + "="*70)
    print("分析与政策建议")
    print("="*70)
    
    if av_impact_df.empty:
        print("数据不足")
        return
    
    # 自动驾驶影响
    print("\n1. 自动驾驶影响分析:")
    
    try:
        base_time = av_impact_df[av_impact_df['p'] == 0]['total_time'].values[0]
        full_av_time = av_impact_df[av_impact_df['p'] == 1.0]['total_time'].values[0]
        improvement = (base_time - full_av_time) / base_time * 100
        
        print(f"   • 纯人类驾驶场景:")
        print(f"     总旅行时间: {base_time:.1f} 小时")
        print(f"     平均速度: {av_impact_df[av_impact_df['p'] == 0]['avg_speed'].values[0]:.1f} km/h")
        
        print(f"   • 完全自动驾驶场景:")
        print(f"     总旅行时间: {full_av_time:.1f} 小时")
        print(f"     平均速度: {av_impact_df[av_impact_df['p'] == 1.0]['avg_speed'].values[0]:.1f} km/h")
        
        print(f"   • 总体改善: {improvement:.1f}%")
        
        # 找到最佳p值
        if 'improvement_pct' in av_impact_df.columns:
            best_idx = av_impact_df['improvement_pct'].idxmax()
            if not pd.isna(best_idx):
                best_p = av_impact_df.loc[best_idx]['p']
                best_improvement = av_impact_df.loc[best_idx]['improvement_pct']
                print(f"   • 最佳自动驾驶比例: p = {best_p:.2f}")
                print(f"     此时改善: {best_improvement:.1f}%")
    except:
        print("   • 分析数据不完整")
    
    # 专用车道建议
    print("\n2. 专用车道分析:")
    
    if not lane_analysis_df.empty:
        print("   • 各方案效益:")
        for _, row in lane_analysis_df.iterrows():
            print(f"     {row['n_lanes']}条专用车道: 节省 {row['time_saving_pct']:.2f}%")
        
        # 经济可行性
        if 'bc_ratio' in lane_analysis_df.columns:
            feasible = lane_analysis_df[lane_analysis_df['bc_ratio'] > 1.0]
            if not feasible.empty:
                print("   • 推荐实施专用车道")
                best_idx = feasible['time_saving_pct'].idxmax()
                best = feasible.loc[best_idx]
                print(f"   • 推荐方案: {best['n_lanes']}条专用车道")
                print(f"     时间节省: {best['time_saving_pct']:.2f}%")
                print(f"     成本效益比: {best['bc_ratio']:.2f}")
            else:
                print("   • 经济上不可行 (B/C < 1.0)")
                print("   • 建议: 等待成本下降或自动驾驶普及率提高")
    
    # 综合建议
    print("\n3. 综合建议:")
    print("   • 短期目标: 提升自动驾驶普及率至p=0.5")
    print("   • 中期目标: 在关键拥堵路段实施专用车道")
    print("   • 长期目标: 实现完全自动驾驶，最大化网络效率")
    print("   • 监测指标: 网络平均v/c比、拥堵路段比例、总旅行时间")
    
    print("\n" + "="*70)
    print("分析完成")
    print("="*70)

def main():
    """主函数"""
    print("="*70)
    print("修复的网络均衡模型分析")
    print("="*70)
    
    try:
        # 1. 构建网络
        print("\n1. 构建物理交通网络...")
        network = PhysicallyCorrectNetwork('2017_MCM_Problem_C_Data.csv')
        
        # 2. 初始化均衡模型
        print("\n2. 初始化修复的均衡模型...")
        eq_model = PhysicallyCorrectEquilibrium_FIXED(network)
        
        # 3. 初始化分析器
        print("\n3. 初始化政策分析器...")
        analyzer = SimplePolicyAnalyzer(eq_model)
        
        # 4. 分析自动驾驶影响
        print("\n4. 分析自动驾驶影响...")
        av_impact_df = analyzer.analyze_av_impact()
        
        # 5. 分析专用车道
        print("\n5. 分析专用车道...")
        lane_analysis_df = analyzer.analyze_dedicated_lanes(p=0.5)
        
        # 6. 可视化
        print("\n6. 生成可视化...")
        fig = create_simple_visualization(av_impact_df, lane_analysis_df)
        plt.savefig('network_equilibrium_fixed_results.png', dpi=300, bbox_inches='tight')
        print("   图表已保存: network_equilibrium_fixed_results.png")
        
        # 7. 生成建议
        print("\n7. 生成政策建议...")
        generate_recommendations(av_impact_df, lane_analysis_df)
        
        # 8. 保存结果
        print("\n8. 保存分析结果...")
        
        # 保存为CSV
        av_impact_df.to_csv('av_impact_results.csv', index=False)
        if not lane_analysis_df.empty:
            lane_analysis_df.to_csv('lane_analysis_results.csv', index=False)
        
        # 保存为JSON
        results = {
            'av_impact': av_impact_df.to_dict('records'),
            'lane_analysis': lane_analysis_df.to_dict('records') if not lane_analysis_df.empty else [],
            'network_info': {
                'nodes': network.G.number_of_nodes(),
                'edges': network.G.number_of_edges(),
                'od_pairs': len(network.od_pairs)
            }
        }
        
        with open('analysis_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        print("\n分析完成!")
        print("结果文件已保存:")
        print("  network_equilibrium_fixed_results.png")
        print("  av_impact_results.csv")
        if not lane_analysis_df.empty:
            print("  lane_analysis_results.csv")
        print("  analysis_results.json")
        
        # 显示图表
        plt.show()
        
        return {
            'network': network,
            'eq_model': eq_model,
            'analyzer': analyzer,
            'av_impact_df': av_impact_df,
            'lane_analysis_df': lane_analysis_df
        }
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()