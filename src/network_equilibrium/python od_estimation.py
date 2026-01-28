import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize, Bounds, LinearConstraint
import os

# ==================== 1. 路网构建类 ====================
class RoadNetwork:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes = {}
        self.edges = {}
        self.observed_flows = {}
        self.df = None
        
    def load_data(self, filepath):
        """加载CSV数据并清理列名"""
        df = pd.read_csv(filepath)
        
        # 清理列名：去除前后空格
        df.columns = df.columns.str.strip()
        
        # 特别处理有问题的列名
        column_mapping = {}
        for col in df.columns:
            cleaned = col.strip()
            if 'DECR MP direction' in cleaned:
                cleaned = 'Number of Lanes DECR MP direction'
            elif 'INCR MP direction' in cleaned:
                cleaned = 'Number of Lanes INCR MP direction'
            elif 'RteType' in cleaned:
                cleaned = 'RteType'
            elif 'Average daily traffic counts' in cleaned:
                cleaned = 'Average daily traffic counts Year_2015'
            column_mapping[col] = cleaned
        
        df = df.rename(columns=column_mapping)
        
        # 删除完全为空的列
        df = df.dropna(axis=1, how='all')
        
        self.df = df
        print(f"✓ 加载了 {len(df)} 条路段数据")
        print(f"✓ 清理后的列名: {list(df.columns)}")
        return df
    
    def build_network(self):
        """从数据构建路网图"""
        if self.df is None:
            raise ValueError("请先加载数据")
            
        # 检查必要的列是否存在
        required_cols = ['Route_ID', 'startMilepost', 'endMilepost', 
                        'Average daily traffic counts Year_2015', 'RteType']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"缺少必要的列: {missing_cols}")
            
        # 识别所有节点
        node_counter = 0
        node_ids = {}
        
        for _, row in self.df.iterrows():
            route = row['Route_ID']
            start = row['startMilepost']
            end = row['endMilepost']
            flow = row['Average daily traffic counts Year_2015']
            
            # 创建节点ID
            start_node = f"R{route}_{start}"
            end_node = f"R{route}_{end}"
            
            if start_node not in node_ids:
                node_ids[start_node] = node_counter
                node_counter += 1
            if end_node not in node_ids:
                node_ids[end_node] = node_counter
                node_counter += 1
            
            # 获取车道数（处理可能缺失的情况）
            lanes_decr = row.get('Number of Lanes DECR MP direction', 2)
            lanes_incr = row.get('Number of Lanes INCR MP direction', 2)
            
            # 添加有向边
            edge_id = f"{start_node}->{end_node}"
            self.graph.add_edge(start_node, end_node, 
                                weight=end-start,
                                flow=flow,
                                route=route,
                                lanes_incr=lanes_incr,
                                lanes_decr=lanes_decr)
            
            self.observed_flows[edge_id] = flow
            
            # 反向边（假设对称）
            if row['RteType'] == 'IS':
                reverse_edge_id = f"{end_node}->{start_node}"
                self.graph.add_edge(end_node, start_node,
                                    weight=start-end,
                                    flow=flow,
                                    route=route,
                                    lanes_incr=lanes_decr,
                                    lanes_decr=lanes_incr)
        
        # 添加交叉口连接
        self._add_intersections()
        
        print(f"✓ 构建了有 {self.graph.number_of_nodes()} 个节点和 {self.graph.number_of_edges()} 条边的路网")
        return self.graph
    
    def _add_intersections(self):
        """根据注释添加交叉口连接"""
        if 'Comments' not in self.df.columns:
            return
            
        intersections = {}
        
        for _, row in self.df.iterrows():
            comment = str(row.get('Comments', '')).lower()
            route = row['Route_ID']
            
            if 'intersection' in comment:
                if 'i5' in comment or 'rte 5' in comment or 'rte5' in comment:
                    intersections.setdefault('I5', []).append((route, row['startMilepost']))
                elif 'i90' in comment:
                    intersections.setdefault('I90', []).append((route, row['startMilepost']))
                elif 'i405' in comment:
                    intersections.setdefault('I405', []).append((route, row['startMilepost']))
                elif 'sr 520' in comment or 'sr520' in comment or '520' in comment:
                    intersections.setdefault('SR520', []).append((route, row['startMilepost']))
                elif 'sr 167' in comment or 'sr167' in comment:
                    intersections.setdefault('SR167', []).append((route, row['startMilepost']))
                elif 'sr 510' in comment or 'sr510' in comment:
                    intersections.setdefault('SR510', []).append((route, row['startMilepost']))
                elif 'sr 512' in comment or 'sr512' in comment:
                    intersections.setdefault('SR512', []).append((route, row['startMilepost']))
                elif 'sr 16' in comment or 'sr16' in comment:
                    intersections.setdefault('SR16', []).append((route, row['startMilepost']))
                elif 'i705' in comment:
                    intersections.setdefault('I705', []).append((route, row['startMilepost']))
                elif 'rte 101' in comment or 'rte101' in comment:
                    intersections.setdefault('Rte101', []).append((route, row['startMilepost']))
        
        # 添加虚拟连接边
        intersection_count = 0
        for int_name, routes in intersections.items():
            if len(routes) >= 2:
                for i in range(len(routes)):
                    for j in range(i+1, len(routes)):
                        route1, mp1 = routes[i]
                        route2, mp2 = routes[j]
                        node1 = f"R{route1}_{mp1}"
                        node2 = f"R{route2}_{mp2}"
                        
                        if node1 in self.graph and node2 in self.graph:
                            self.graph.add_edge(node1, node2, weight=0.01, is_intersection=True)
                            self.graph.add_edge(node2, node1, weight=0.01, is_intersection=True)
                            intersection_count += 1
        
        if intersection_count > 0:
            print(f"✓ 添加了 {intersection_count} 个交叉口连接")
    
    def identify_od_nodes(self):
        """识别可能的OD节点"""
        od_nodes = []
        
        # 策略1：从注释中识别交叉口
        if 'Comments' in self.df.columns:
            for _, row in self.df.iterrows():
                comment = str(row.get('Comments', ''))
                if 'intersection' in comment.lower():
                    node = f"R{row['Route_ID']}_{row['startMilepost']}"
                    if node not in od_nodes:
                        od_nodes.append(node)
        
        # 策略2：选择流量大的路段端点
        if len(od_nodes) < 10:
            top_flows = self.df.nlargest(20, 'Average daily traffic counts Year_2015')
            for _, row in top_flows.iterrows():
                for mp in [row['startMilepost'], row['endMilepost']]:
                    node = f"R{row['Route_ID']}_{mp}"
                    if node not in od_nodes:
                        od_nodes.append(node)
        
        # 策略3：添加道路起点终点
        for route in self.df['Route_ID'].unique():
            route_df = self.df[self.df['Route_ID'] == route]
            start_mp = route_df['startMilepost'].min()
            end_mp = route_df['endMilepost'].max()
            
            for mp in [start_mp, end_mp]:
                node = f"R{route}_{mp}"
                if node not in od_nodes:
                    od_nodes.append(node)
        
        # 策略4：添加所有节点（如果还太少）
        if len(od_nodes) < 15:
            for node in self.graph.nodes():
                if node not in od_nodes:
                    od_nodes.append(node)
        
        print(f"✓ 选择了 {len(od_nodes)} 个OD节点")
        return od_nodes[:25]  # 限制数量以提高计算效率
    
    def plot_network(self, save_path='results/network.png'):
        """绘制路网图"""
        plt.figure(figsize=(15, 10))
        
        # 节点位置
        pos = nx.spring_layout(self.graph, k=2, iterations=50)
        
        # 绘制节点
        nx.draw_networkx_nodes(self.graph, pos, node_size=50, node_color='lightblue')
        
        # 绘制边
        edges = self.graph.edges()
        edge_colors = []
        edge_widths = []
        
        for u, v in edges:
            if 'flow' in self.graph[u][v]:
                flow = self.graph[u][v]['flow']
                edge_colors.append(flow)
                edge_widths.append(min(5, flow / 50000))
            else:
                edge_colors.append(10000)
                edge_widths.append(1)
        
        nx.draw_networkx_edges(self.graph, pos, 
                              edge_color=edge_colors, 
                              edge_cmap=plt.cm.Reds,
                              width=edge_widths,
                              alpha=0.6)
        
        # 标注主要节点
        labels = {}
        for node in self.graph.nodes():
            if 'intersection' in str(node).lower() or any(x in node for x in ['0.0', 'start', 'end']):
                labels[node] = node.split('_')[0] if '_' in node else node
        
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)
        
        plt.title('Road Network with Traffic Flow', fontsize=14)
        plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Reds), 
                     label='Traffic Flow (veh/day)')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.show()


# ==================== 2. OD估计器类 ====================
class ODEstimator:
    def __init__(self, network):
        self.network = network
        self.graph = network.graph
        self.observed_flows = network.observed_flows
        self.A = None
        self.edges = None
        self.edge_to_idx = None
        self.od_pairs = None
        
    def build_path_assignment_matrix(self, od_nodes):
        """构建路径-路段分配矩阵A"""
        edges = list(self.graph.edges())
        edge_to_idx = {edge: i for i, edge in enumerate(edges)}
        
        # 生成所有OD对
        od_pairs = []
        for i in range(len(od_nodes)):
            for j in range(len(od_nodes)):
                if i != j:
                    od_pairs.append((od_nodes[i], od_nodes[j]))
        
        print(f"✓ 有 {len(edges)} 条观测路段，{len(od_pairs)} 个OD对")
        
        # 初始化分配矩阵
        m = len(edges)
        n = len(od_pairs)
        A = np.zeros((m, n))
        
        # 对每个OD对计算最短路径
        print("正在计算最短路径...")
        for j, (orig, dest) in enumerate(od_pairs):
            if j % 100 == 0 and j > 0:
                print(f"  处理进度: {j}/{n} ({j/n*100:.1f}%)...")
                
            try:
                path = nx.shortest_path(self.graph, orig, dest, weight='weight')
                
                # 标记路径经过的边
                for k in range(len(path)-1):
                    edge = (path[k], path[k+1])
                    if edge in edge_to_idx:
                        i = edge_to_idx[edge]
                        A[i, j] = 1
            except:
                pass  # 没有路径
        
        self.A = A
        self.edges = edges
        self.edge_to_idx = edge_to_idx
        self.od_pairs = od_pairs
        
        print(f"✓ 分配矩阵构建完成: {A.shape}")
        return A, od_pairs
    
    def estimate_od(self, method='least_squares'):
        """估计OD矩阵"""
        # 观测流量向量
        y = np.array([self.observed_flows.get(f"{u}->{v}", 0) 
                     for u, v in self.edges])
        
        # 移除零流量的行
        non_zero_mask = y > 0
        y = y[non_zero_mask]
        A = self.A[non_zero_mask, :]
        
        print(f"✓ 使用 {A.shape[0]} 个有效观测约束，估计 {A.shape[1]} 个OD对流量")
        
        if method == 'least_squares':
            return self._least_squares(A, y)
        elif method == 'entropy':
            return self._entropy_maximization(A, y)
        elif method == 'gradient_descent':
            return self._gradient_descent(A, y)
        else:
            raise ValueError(f"未知方法: {method}")
    
    def _least_squares(self, A, y):
        """最小二乘法求解"""
        print("  使用最小二乘法求解...")
        
        def objective(x):
            return np.sum((A @ x - y) ** 2)
        
        bounds = Bounds(0, np.inf)
        x0 = np.ones(A.shape[1]) * np.mean(y) / A.shape[1]
        
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 1000, 'disp': False})
        
        print(f"  优化完成，最终损失: {result.fun:.2f}")
        return result.x
    
    def _entropy_maximization(self, A, y):
        """最大熵方法"""
        print("  使用最大熵方法求解...")
        
        n_od = A.shape[1]
        
        def objective(x):
            x_safe = np.maximum(x, 1e-10)
            return np.sum(x_safe * np.log(x_safe))
        
        constraints = LinearConstraint(A, y, y)
        bounds = Bounds(0, np.inf)
        
        x0 = np.ones(n_od) * np.mean(y) / n_od
        
        result = minimize(lambda x: -objective(x), x0, 
                         method='SLSQP',
                         constraints=constraints,
                         bounds=bounds,
                         options={'maxiter': 500, 'disp': True})
        
        return result.x
    
    def _gradient_descent(self, A, y, learning_rate=0.01, iterations=1000):
        """梯度下降法"""
        print("  使用梯度下降法求解...")
        
        n_od = A.shape[1]
        x = np.ones(n_od) * np.mean(y) / n_od
        
        for it in range(iterations):
            residual = A @ x - y
            gradient = 2 * A.T @ residual
            x = x - learning_rate * gradient
            x = np.maximum(x, 0)
            
            if it % 200 == 0:
                loss = np.sum(residual ** 2)
                print(f"    迭代 {it}, 损失: {loss:.2f}")
        
        return x
    
    def save_results(self, x, output_file='results/od_matrix.csv'):
        """保存OD矩阵结果"""
        # 创建OD对DataFrame
        od_matrix = pd.DataFrame({
            'Origin': [pair[0] for pair in self.od_pairs],
            'Destination': [pair[1] for pair in self.od_pairs],
            'Estimated_Flow': x
        })
        
        # 过滤掉估计流量很小的OD对
        od_matrix = od_matrix[od_matrix['Estimated_Flow'] > 1]
        
        # 重塑为矩阵形式
        origins = od_matrix['Origin'].unique()
        destinations = od_matrix['Destination'].unique()
        
        matrix_df = pd.DataFrame(index=origins, columns=destinations, dtype=float)
        for _, row in od_matrix.iterrows():
            matrix_df.loc[row['Origin'], row['Destination']] = row['Estimated_Flow']
        
        matrix_df = matrix_df.fillna(0)
        
        # 保存文件
        od_matrix.to_csv('results/od_pairs.csv', index=False)
        matrix_df.to_csv(output_file)
        
        print(f"✓ 保存了 {len(od_matrix)} 个OD对到 {output_file}")
        
        # 统计信息
        print("\n" + "="*50)
        print("OD矩阵统计信息:")
        print("="*50)
        print(f"总出行量: {np.sum(x):,.0f} 辆/天")
        print(f"非零OD对数量: {(x > 1).sum()}")
        print(f"最大OD流量: {np.max(x):,.0f} 辆/天")
        print(f"平均OD流量: {np.mean(x[x>0]):,.0f} 辆/天")
        print(f"OD矩阵形状: {matrix_df.shape}")
        
        return matrix_df, od_matrix
    
    def validate(self, x, threshold=0.8):
        """验证反推结果"""
        y_pred = self.A @ x
        
        # 实际观测
        y_obs = np.array([self.observed_flows.get(f"{u}->{v}", 0) 
                         for u, v in self.edges])
        
        # 只比较有观测的路段
        valid_mask = y_obs > 0
        y_obs_valid = y_obs[valid_mask]
        y_pred_valid = y_pred[valid_mask]
        
        if len(y_obs_valid) == 0:
            print("警告: 没有有效的观测数据用于验证")
            return 0, 0, 100
        
        correlation = np.corrcoef(y_obs_valid, y_pred_valid)[0, 1]
        
        # 均方根误差和平均绝对百分比误差
        rmse = np.sqrt(np.mean((y_obs_valid - y_pred_valid) ** 2))
        mape = np.mean(np.abs((y_obs_valid - y_pred_valid) / (y_obs_valid + 1))) * 100
        
        print("\n" + "="*50)
        print("模型验证结果:")
        print("="*50)
        print(f"相关系数 (R): {correlation:.4f}")
        print(f"解释方差 (R²): {correlation**2:.4f}")
        print(f"均方根误差 (RMSE): {rmse:,.0f} 辆/天")
        print(f"平均绝对百分比误差 (MAPE): {mape:.2f}%")
        
        return correlation, rmse, mape


# ==================== 3. 可视化函数 ====================
def visualize_od_matrix(od_file='results/od_matrix.csv'):
    """可视化OD矩阵"""
    print("\n正在生成可视化结果...")
    
    # 加载OD矩阵
    try:
        od_matrix = pd.read_csv(od_file, index_col=0)
    except:
        print(f"错误: 无法加载文件 {od_file}")
        return []
    
    od_matrix = od_matrix.apply(pd.to_numeric, errors='coerce')
    od_matrix = od_matrix.fillna(0)
    
    if od_matrix.empty or od_matrix.shape[0] == 0 or od_matrix.shape[1] == 0:
        print("警告: OD矩阵为空")
        return []
    
    # 1. OD矩阵热力图
    plt.figure(figsize=(14, 12))
    
    # 限制大小以便可视化
    if od_matrix.shape[0] > 20:
        row_sums = od_matrix.sum(axis=1)
        col_sums = od_matrix.sum(axis=0)
        top_rows = row_sums.nlargest(min(15, len(row_sums))).index
        top_cols = col_sums.nlargest(min(15, len(col_sums))).index
        display_matrix = od_matrix.loc[top_rows, top_cols]
    else:
        display_matrix = od_matrix
    
    mask = display_matrix == 0
    sns.heatmap(display_matrix, 
                cmap='YlOrRd',
                mask=mask,
                square=True,
                linewidths=0.5,
                annot=True,
                fmt='.0f',
                cbar_kws={'label': 'Traffic Flow (veh/day)'})
    
    plt.title('OD Matrix Heatmap (Top Origins & Destinations)', fontsize=14)
    plt.xlabel('Destination', fontsize=12)
    plt.ylabel('Origin', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('results/od_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. 流量分布直方图
    plt.figure(figsize=(12, 5))
    
    flows = od_matrix.values.flatten()
    flows = flows[flows > 0]
    
    if len(flows) > 0:
        plt.subplot(1, 2, 1)
        plt.hist(flows, bins=min(50, len(flows)), alpha=0.7, color='steelblue', edgecolor='black')
        plt.axvline(np.mean(flows), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(flows):.0f}')
        plt.xlabel('OD Flow (veh/day)')
        plt.ylabel('Frequency')
        plt.title('Distribution of OD Flows')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist(np.log10(flows + 1), bins=min(30, len(flows)), alpha=0.7, color='green', edgecolor='black')
        plt.xlabel('log10(Flow + 1)')
        plt.ylabel('Frequency')
        plt.title('Log Distribution of OD Flows')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/od_distribution.png', dpi=300)
        plt.show()
    
    # 3. 前10大OD流
    top_od_flows = []
    for i in range(od_matrix.shape[0]):
        for j in range(od_matrix.shape[1]):
            flow = od_matrix.iloc[i, j]
            if flow > 0:
                top_od_flows.append((od_matrix.index[i], od_matrix.columns[j], flow))
    
    top_od_flows.sort(key=lambda x: x[2], reverse=True)
    top_10 = top_od_flows[:10]
    
    print("\n" + "="*50)
    print("前10大OD流:")
    print("="*50)
    for i, (orig, dest, flow) in enumerate(top_10, 1):
        print(f"{i:2d}. {orig:20s} → {dest:20s}: {flow:,.0f} 辆/天")
    
    # 保存前10大OD流
    if top_10:
        top_df = pd.DataFrame(top_10, columns=['Origin', 'Destination', 'Flow'])
        top_df.to_csv('results/top_od_flows.csv', index=False)
    
    return top_10


# ==================== 4. 主函数 ====================
def main():
    """主程序"""
    print("="*60)
    print("基于路段流量反推OD矩阵 - 完整解决方案")
    print("="*60)
    
    # 创建结果目录
    os.makedirs('results', exist_ok=True)
    
    # 1. 加载数据
    print("\n[步骤1] 加载数据...")
    data_file = '2017_MCM_Problem_C_Data.csv'
    
    try:
        network = RoadNetwork()
        df = network.load_data(data_file)
        print(f"  道路数量: {df['Route_ID'].nunique()}")
        print(f"  总流量: {df['Average daily traffic counts Year_2015'].sum():,.0f} 辆/天")
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{data_file}'")
        print("请确保CSV文件在当前目录下")
        return None, None
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None, None
    
    # 2. 构建路网
    print("\n[步骤2] 构建路网...")
    try:
        graph = network.build_network()
    except Exception as e:
        print(f"构建路网时出错: {e}")
        print("检查数据格式是否正确")
        return None, None
    
    # 3. 绘制路网
    print("\n[步骤3] 绘制路网图...")
    try:
        network.plot_network('results/network.png')
    except Exception as e:
        print(f"绘制路网图时出错: {e}")
        # 继续执行，这不是关键错误
    
    # 4. 识别OD节点
    print("\n[步骤4] 识别OD节点...")
    od_nodes = network.identify_od_nodes()
    print(f"  OD节点示例: {od_nodes[:8]}")
    
    # 5. 构建OD估计器
    print("\n[步骤5] 构建OD估计模型...")
    estimator = ODEstimator(network)
    
    # 6. 构建分配矩阵
    print("\n[步骤6] 计算路径分配矩阵...")
    try:
        A, od_pairs = estimator.build_path_assignment_matrix(od_nodes)
    except Exception as e:
        print(f"构建分配矩阵时出错: {e}")
        print("可能原因: 网络太大或节点太多")
        # 尝试减少节点数量
        print("尝试减少OD节点数量...")
        od_nodes = od_nodes[:15]
        A, od_pairs = estimator.build_path_assignment_matrix(od_nodes)
    
    # 7. 估计OD矩阵
    print("\n[步骤7] 估计OD矩阵...")
    print("  可选方法: least_squares, entropy, gradient_descent")
    
    try:
        # 尝试不同方法
        methods = ['least_squares']  # 可以添加更多方法
        best_result = None
        best_correlation = -1
        
        for method in methods:
            print(f"\n  正在使用 {method} 方法...")
            od_flows = estimator.estimate_od(method=method)
            correlation, rmse, mape = estimator.validate(od_flows)
            
            if correlation > best_correlation:
                best_correlation = correlation
                best_result = od_flows
        
        # 8. 保存结果
        print("\n[步骤8] 保存结果...")
        od_matrix_df, od_pairs_df = estimator.save_results(best_result, 'results/od_matrix.csv')
        
        # 9. 可视化
        print("\n[步骤9] 生成可视化结果...")
        top_flows = visualize_od_matrix('results/od_matrix.csv')
        
        # 10. 生成汇总报告
        print("\n[步骤10] 生成汇总报告...")
        with open('results/summary_report.txt', 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("OD矩阵反推结果汇总报告\n")
            f.write("="*60 + "\n\n")
            
            f.write("1. 数据概况:\n")
            f.write(f"   路段数量: {len(df)}\n")
            f.write(f"   道路数量: {df['Route_ID'].nunique()}\n")
            f.write(f"   总观测流量: {df['Average daily traffic counts Year_2015'].sum():,.0f} 辆/天\n\n")
            
            f.write("2. 模型结果:\n")
            f.write(f"   OD节点数量: {len(od_nodes)}\n")
            f.write(f"   OD对数量: {len(od_pairs)}\n")
            f.write(f"   估计的总出行量: {np.sum(best_result):,.0f} 辆/天\n")
            f.write(f"   非零OD对数量: {(best_result > 1).sum()}\n\n")
            
            f.write("3. 模型验证:\n")
            correlation, rmse, mape = estimator.validate(best_result)
            f.write(f"   相关系数: {correlation:.4f}\n")
            f.write(f"   解释方差: {correlation**2:.4f}\n")
            f.write(f"   RMSE: {rmse:,.0f} 辆/天\n")
            f.write(f"   MAPE: {mape:.2f}%\n\n")
            
            f.write("4. 前5大OD流:\n")
            for i, (orig, dest, flow) in enumerate(top_flows[:5], 1):
                f.write(f"   {i}. {orig} → {dest}: {flow:,.0f} 辆/天\n")
        
        print("\n" + "="*60)
        print("完成！所有结果已保存到 'results/' 目录")
        print("="*60)
        print("\n生成的文件:")
        print("  results/network.png        - 路网图")
        print("  results/od_matrix.csv      - OD矩阵(CSV格式)")
        print("  results/od_pairs.csv       - OD对列表")
        print("  results/od_heatmap.png     - OD矩阵热力图")
        print("  results/od_distribution.png - OD流量分布图")
        print("  results/top_od_flows.csv   - 前10大OD流")
        print("  results/summary_report.txt - 汇总报告")
        
        return od_matrix_df, top_flows
        
    except Exception as e:
        print(f"\n运行过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# ==================== 5. 直接运行 ====================
if __name__ == "__main__":
    print("开始运行OD矩阵反推程序...\n")
    
    # 检查必要的数据文件
    if not os.path.exists('2017_MCM_Problem_C_Data.csv'):
        print("错误: 未找到数据文件 '2017_MCM_Problem_C_Data.csv'")
        print("请将CSV文件放在当前目录下，然后重新运行程序。")
        
        # 显示当前目录内容
        print("\n当前目录内容:")
        for file in os.listdir('.'):
            if file.endswith('.csv'):
                print(f"  - {file}")
        
        exit(1)
    
    # 运行主程序
    try:
        od_matrix, top_flows = main()
        
        if od_matrix is not None:
            print("\n是否查看OD矩阵前几行？(y/n)")
            if input().lower() == 'y':
                print("\nOD矩阵预览:")
                print(od_matrix.head())
            
        print("\n程序结束。")
        
    except KeyboardInterrupt:
        print("\n\n程序被用户中断。")
    except Exception as e:
        print(f"\n程序运行出错: {e}")
        import traceback
        traceback.print_exc()