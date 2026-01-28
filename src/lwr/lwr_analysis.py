# 基于LWR模型的宏观交通流分析
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def load_and_prepare_data():
    """加载并准备数据"""
    df = pd.read_csv('2017_MCM_Problem_C_Data.csv')

    # 清理列名
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('"', '').str.replace('(', '').str.replace(')', '')

    # 计算路段长度
    df['segment_length'] = df['endMilepost'] - df['startMilepost']

    # 查找路线类型列（可能经过清理后名称变化）
    route_type_col = None
    for col in df.columns:
        if 'RteType' in col:
            route_type_col = col
            break

    if route_type_col:
        # 将路线类型转换为中文
        df[route_type_col] = df[route_type_col].replace({'IS': '州际公路', 'SR': '州级公路'})
        # 重命名为标准名称
        df = df.rename(columns={route_type_col: 'RteType'})
    else:
        print("警告: 无法找到路线类型列")

    # 计算每个路段的平均车道数
    df['avg_lanes'] = (df['Number_of_Lanes_DECR_MP_direction'] + df['Number_of_Lanes_INCR_MP_direction']) / 2

    return df

def calculate_density_and_velocity(df):
    """计算车辆密度和平均速度"""
    # 假设参数
    road_length = df['segment_length'].sum()  # 总路段长度（英里）
    total_time = 24  # 小时（每日平均）

    # 计算每个路段的车辆密度（辆/英里）
    # 密度 = 交通量 / (速度 * 时间)，但我们需要估计速度
    # 使用Greenshields模型：v = vf * (1 - ρ/ρj)
    # 其中vf是自由流速度，ρj是堵塞密度

    # 假设参数
    vf = 60  # 自由流速度（英里/小时）
    ρj = 200  # 堵塞密度（辆/英里/车道）

    # 估计每个路段的密度
    # 使用简化公式：ρ ≈ q / (v * L)，其中q是流量，v是速度，L是路段长度
    # 由于我们没有直接的速度数据，我们使用经验公式

    # 计算每个路段的平均密度
    df['density'] = df['Average_daily_traffic_counts_Year_2015'] / (df['segment_length'] * vf * total_time / df['avg_lanes'])

    # 计算平均速度（使用Greenshields模型）
    df['velocity'] = vf * (1 - df['density'] / ρj)

    # 确保速度不为负数
    df['velocity'] = df['velocity'].clip(lower=5)

    return df

def greenshields_model(ρ, vf, ρj):
    """Greenshields速度-密度模型"""
    return vf * (1 - ρ / ρj)

def fit_greenshields_model(df):
    """拟合Greenshields模型参数"""
    # 准备数据
    q = df['Average_daily_traffic_counts_Year_2015'] / 24  # 转换为每小时流量
    ρ = df['density']
    v = df['velocity']

    # 使用曲线拟合
    try:
        params, _ = curve_fit(greenshields_model, ρ, v, p0=[60, 200])
        vf_fit, ρj_fit = params
        return vf_fit, ρj_fit
    except:
        # 如果拟合失败，使用默认值
        return 60, 200

def analyze_traffic_characteristics(df):
    """分析交通特性"""
    print("=== 基于LWR模型的宏观交通流分析 ===\n")

    # 基本统计
    print("1. 整体交通特性：")
    print(f"总路段数：{len(df)}")
    print(f"总路线数：{df['Route_ID'].nunique()}")
    print(f"总路段长度：{df['segment_length'].sum():.2f} 英里")
    print(f"平均每日交通量：{df['Average_daily_traffic_counts_Year_2015'].mean():,.0f} 辆车")
    print(f"最大每日交通量：{df['Average_daily_traffic_counts_Year_2015'].max():,.0f} 辆车")
    print(f"最小每日交通量：{df['Average_daily_traffic_counts_Year_2015'].min():,.0f} 辆车\n")

    # 路线特性分析
    print("2. 不同路线的交通特性：")
    for route_id in sorted(df['Route_ID'].unique()):
        route_data = df[df['Route_ID'] == route_id]
        route_type = route_data['RteType'].iloc[0]

        print(f"路线 {route_id} ({route_type})：")
        print(f"  - 路段数：{len(route_data)}")
        print(f"  - 总长度：{route_data['segment_length'].sum():.2f} 英里")
        print(f"  - 平均交通量：{route_data['Average_daily_traffic_counts_Year_2015'].mean():,.0f} 辆车/天")
        print(f"  - 最大交通量：{route_data['Average_daily_traffic_counts_Year_2015'].max():,.0f} 辆车/天")
        print(f"  - 平均车道数：{route_data['avg_lanes'].mean():.1f} 个")
        print(f"  - 平均密度：{route_data['density'].mean():.2f} 辆车/英里/车道")
        print(f"  - 平均速度：{route_data['velocity'].mean():.2f} 英里/小时\n")

    # 拟合Greenshields模型
    vf_fit, ρj_fit = fit_greenshields_model(df)
    print(f"3. Greenshields模型拟合参数：")
    print(f"  - 自由流速度 (vf)：{vf_fit:.2f} 英里/小时")
    print(f"  - 堵塞密度 (ρj)：{ρj_fit:.2f} 辆车/英里/车道\n")

    return vf_fit, ρj_fit

def plot_traffic_characteristics(df, vf_fit, ρj_fit):
    """绘制交通特性图"""
    plt.figure(figsize=(15, 10))

    # 流量-密度关系
    plt.subplot(2, 2, 1)
    q = df['Average_daily_traffic_counts_Year_2015'] / 24  # 每小时流量
    ρ = df['density']
    plt.scatter(ρ, q, alpha=0.6)
    plt.title('流量-密度关系')
    plt.xlabel('车辆密度 (辆/英里/车道)')
    plt.ylabel('交通流量 (辆/小时)')
    plt.grid(True)

    # 速度-密度关系（Greenshields模型）
    plt.subplot(2, 2, 2)
    ρ_range = np.linspace(0, ρj_fit * 1.1, 100)
    v_range = greenshields_model(ρ_range, vf_fit, ρj_fit)
    plt.scatter(df['density'], df['velocity'], alpha=0.6, label='实际数据')
    plt.plot(ρ_range, v_range, 'r-', label=f'Greenshields模型 (vf={vf_fit:.1f}, ρj={ρj_fit:.1f})')
    plt.title('速度-密度关系')
    plt.xlabel('车辆密度 (辆/英里/车道)')
    plt.ylabel('平均速度 (英里/小时)')
    plt.legend()
    plt.grid(True)

    # 不同路线的交通量分布
    plt.subplot(2, 2, 3)
    sns.boxplot(data=df, x='Route_ID', y='Average_daily_traffic_counts_Year_2015')
    plt.title('不同路线的交通量分布')
    plt.xlabel('路线ID')
    plt.ylabel('每日平均交通量')
    plt.grid(True)

    # 密度与车道数的关系
    plt.subplot(2, 2, 4)
    # 使用更清晰的颜色映射（使用整数键）
    palette = {5: 'red', 90: 'blue', 405: 'green', 520: 'purple'}
    sns.scatterplot(data=df, x='avg_lanes', y='density', hue='Route_ID', size='Average_daily_traffic_counts_Year_2015',
                    sizes=(20, 200), palette=palette, alpha=0.8)
    plt.title('车辆密度与车道数的关系')
    plt.xlabel('平均车道数')
    plt.ylabel('车辆密度 (辆/英里/车道)')
    plt.grid(True)
    # 修改图例样式
    plt.legend(title='路线ID', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig('lwr_traffic_analysis.png')
    print("交通特性图已保存为 'lwr_traffic_analysis.png'")

def analyze_capacity_and_congestion(df):
    """分析道路容量和拥堵情况"""
    print("4. 道路容量和拥堵分析：")

    # 计算理论最大容量
    vf_fit, ρj_fit = fit_greenshields_model(df)
    max_capacity = vf_fit * ρj_fit / 4  # 最大流量（每车道）

    print(f"  - 理论最大容量（每车道）：约 {max_capacity:.0f} 辆车/小时")

    # 识别高密度路段（潜在拥堵）
    high_density_threshold = ρj_fit * 0.7  # 70%堵塞密度
    congested_segments = df[df['density'] > high_density_threshold]

    print(f"  - 高密度路段数（潜在拥堵）：{len(congested_segments)}")
    print(f"  - 高密度路段比例：{len(congested_segments)/len(df)*100:.1f}%")

    if len(congested_segments) > 0:
        print("\n高密度路段详情：")
        for idx, row in congested_segments.nlargest(5, 'density').iterrows():
            print(f"    - 路线 {row['Route_ID']}，里程碑 {row['startMilepost']:.2f}-{row['endMilepost']:.2f}：")
            print(f"      密度={row['density']:.2f} 辆车/英里/车道，交通量={row['Average_daily_traffic_counts_Year_2015']:,} 辆车/天")

    # 分析不同路线的拥堵情况
    print("\n不同路线的拥堵情况：")
    for route_id in sorted(df['Route_ID'].unique()):
        route_data = df[df['Route_ID'] == route_id]
        route_congested = route_data[route_data['density'] > high_density_threshold]
        congestion_ratio = len(route_congested) / len(route_data) * 100

        print(f"  - 路线 {route_id}：{len(route_congested)}个高密度路段，占比{congestion_ratio:.1f}%")

if __name__ == "__main__":
    # 加载并准备数据
    traffic_data = load_and_prepare_data()

    # 计算密度和速度
    traffic_data = calculate_density_and_velocity(traffic_data)

    # 分析交通特性
    vf, ρj = analyze_traffic_characteristics(traffic_data)

    # 绘制交通特性图
    plot_traffic_characteristics(traffic_data, vf, ρj)

    # 分析容量和拥堵
    analyze_capacity_and_congestion(traffic_data)

    print("\n=== LWR模型分析完成 ===")