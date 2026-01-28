import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置中文字体 - 根据系统选择合适字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 设置字体列表，按优先级排序
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 加载CSV数据
def load_and_clean_data():
    # 读取CSV文件
    df = pd.read_csv('2017_MCM_Problem_C_Data.csv')
    # 打印原始列名以便调试
    print("原始列名:")
    print(df.columns.tolist())
    # 清理列名 - 删除额外的空格和特殊字符
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('"', '').str.replace('(', '').str.replace(')', '')
    # 打印清理后的列名以便调试
    print("\n清理后的列名:")
    print(df.columns.tolist())
    # 将路线类型缩写转换为完整名称以便清晰显示
    # 处理可能已经被清理的特定列名
    route_type_col = None
    for col in df.columns:
        if 'RteType' in col:
            route_type_col = col
            break
    if route_type_col:
        df[route_type_col] = df[route_type_col].replace({'IS': '州际公路', 'SR': '州级公路'})
        # 重命名为标准名称
        df = df.rename(columns={route_type_col: 'RteType'})
    else:
        print("警告: 无法找到路线类型列")
    # 计算路段长度
    df['segment_length'] = df['endMilepost'] - df['startMilepost']
    return df
def analyze_data(df):
    print("=== 公路数据分析 ===\n")
    # 基本统计信息
    print("1. 整体统计信息:")
    print(f"路段总数: {len(df)}")
    print(f"路线总数: {df['Route_ID'].nunique()}")
    print(f"路线类型: {df['RteType'].unique()}")
    print(f"每日平均交通量范围: {df['Average_daily_traffic_counts_Year_2015'].min():,} 到 {df['Average_daily_traffic_counts_Year_2015'].max():,} 辆车")
    print(f"覆盖的公路总里程: {df['segment_length'].sum():.2f} 英里\n")
    # 特定路线分析
    print("2. 特定路线分析:")
    for route_id in sorted(df['Route_ID'].unique()):
        route_data = df[df['Route_ID'] == route_id]
        route_type = route_data['RteType'].iloc[0]
        total_length = route_data['segment_length'].sum()
        avg_traffic = route_data['Average_daily_traffic_counts_Year_2015'].mean()
        max_traffic = route_data['Average_daily_traffic_counts_Year_2015'].max()
        print(f"路线 {route_id} ({route_type}):")
        print(f"  - 总长度: {total_length:.2f} 英里")
        print(f"  - 平均交通量: {avg_traffic:,.0f} 辆车/天")
        print(f"  - 最大交通量: {max_traffic:,.0f} 辆车/天")
        print(f"  - 里程碑范围: {route_data['startMilepost'].min():.2f} 到 {route_data['endMilepost'].max():.2f}")
        print(f"  - 路段数: {len(route_data)}\n")
    # 交通模式
    print("3. 交通模式:")
    print("交通量最高的路段:")
    high_traffic = df.nlargest(5, 'Average_daily_traffic_counts_Year_2015')
    for idx, row in high_traffic.iterrows():
        print(f"  - 路线 {row['Route_ID']} 在里程碑 {row['startMilepost']:.2f}-{row['endMilepost']:.2f}: {row['Average_daily_traffic_counts_Year_2015']:,} 辆车/天")
    print("\n交通量最低的路段:")
    low_traffic = df.nsmallest(5, 'Average_daily_traffic_counts_Year_2015')
    for idx, row in low_traffic.iterrows():
        print(f"  - 路线 {row['Route_ID']} 在里程碑 {row['startMilepost']:.2f}-{row['endMilepost']:.2f}: {row['Average_daily_traffic_counts_Year_2015']:,} 辆车/天")

    # 车道分析
    print("\n4. 车道配置:")
    print("车道分布（递减方向）:")
    print(df['Number_of_Lanes_DECR_MP_direction'].value_counts().sort_index())

    print("\n车道分布（递增方向）:")
    print(df['Number_of_Lanes_INCR_MP_direction'].value_counts().sort_index())

    # 交叉路口和特殊位置
    print("\n5. 关键交叉路口和位置:")
    intersections = df[df['Comments'].str.contains('intersection|Intersection|boundary', case=False, na=False)]
    if not intersections.empty:
        print("发现的关键交叉路口:")
        for idx, row in intersections.iterrows():
            print(f"  - 路线 {row['Route_ID']} 在里程碑 {row['startMilepost']:.2f}: {row['Comments']}")
    else:
        print("评论中未发现关键交叉路口。")

    # 基于路线方向的方向分析
    print("\n6. 方向分析:")
    print("对于南北向道路（如I-5）：")
    print("  - 递增方向 = 北行")
    print("  - 递减方向 = 南行")
    print("对于东西向道路（如I-90，I-520）：")
    print("  - 递增方向 = 东行")
    print("  - 递减方向 = 西行")

    # 交通量与车道数的相关性
    print("\n7. 交通量与车道数的相关性:")
    print("分析车道数量与交通量之间的关系...")

    # 创建可视化
    create_visualizations(df)

def create_visualizations(df):
    """创建道路数据的可视化"""
    plt.figure(figsize=(15, 10))

    # 不同路线的交通分布
    plt.subplot(2, 2, 1)
    sns.boxplot(data=df, x='Route_ID', y='Average_daily_traffic_counts_Year_2015')
    plt.title('不同路线的交通分布')
    plt.xlabel('路线ID')
    plt.ylabel('每日平均交通量')

    # 主要路线的交通量与里程碑
    plt.subplot(2, 2, 2)
    for route_id in sorted(df['Route_ID'].unique()):
        route_data = df[df['Route_ID'] == route_id]
        plt.plot(route_data['startMilepost'], route_data['Average_daily_traffic_counts_Year_2015'],
                label=f'路线 {route_id}', alpha=0.7)
    plt.title('路线沿线的交通模式')
    plt.xlabel('里程碑')
    plt.ylabel('每日平均交通量')
    plt.legend()

    # 车道配置
    plt.subplot(2, 2, 3)
    sns.countplot(data=df, x='Number_of_Lanes_DECR_MP_direction')
    plt.title('车道分布（递减方向）')

    plt.subplot(2, 2, 4)
    sns.countplot(data=df, x='Number_of_Lanes_INCR_MP_direction')
    plt.title('车道分布（递增方向）')

    plt.tight_layout()
    plt.savefig('road_data_analysis.png')
    print("\n可视化已保存为 'road_data_analysis.png'")

if __name__ == "__main__":
    # 加载并分析数据
    road_data = load_and_clean_data()
    analyze_data(road_data)