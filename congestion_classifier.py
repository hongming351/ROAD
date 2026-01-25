# 随机森林拥堵分类器 - ML1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import joblib
import warnings

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """加载并准备数据"""
    df = pd.read_csv('2017_MCM_Problem_C_Data.csv')

    # 清理列名
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('"', '').str.replace('(', '').str.replace(')', '')

    # 计算路段长度
    df['length'] = df['endMilepost'] - df['startMilepost']

    # 车道数特征
    df['lanes_incr'] = df['Number_of_Lanes_INCR_MP_direction']
    df['lanes_decr'] = df['Number_of_Lanes_DECR_MP_direction']

    # 平均车道数
    df['avg_lanes'] = (df['lanes_incr'] + df['lanes_decr']) / 2

    # 将路线类型转换为数值
    df['route_type'] = df['RteType_IS=_Interstate,_SR=_State_Route'].apply(lambda x: 1 if 'IS' in x else 0)

    return df

def calculate_traffic_features(df):
    """计算交通特征"""
    # 假设峰值交通量为平均交通量的120%（基于8%峰值比例）
    df['peak_traffic'] = df['Average_daily_traffic_counts_Year_2015'] * 1.2

    # 假设路段容量（基于车道数和标准容量）
    # 标准容量：2000辆/小时/车道（基于60英里/小时和300英尺车头时距）
    df['capacity'] = df['avg_lanes'] * 2000 * 24  # 每日容量

    # 需求容量比
    df['demand_capacity_ratio'] = df['Average_daily_traffic_counts_Year_2015'] / df['capacity']

    # 是否为瓶颈（需求容量比 > 0.8）
    df['is_bottleneck'] = df['demand_capacity_ratio'] > 0.8

    # 计算p值（基于密度和速度的假设值）
    # 使用LWR模型中的参数来估计p值
    df['density'] = df['Average_daily_traffic_counts_Year_2015'] / (df['length'] * 60 * 24 / df['avg_lanes'])
    df['velocity'] = 60 * (1 - df['density'] / 200)  # Greenshields模型
    df['velocity'] = df['velocity'].clip(lower=5)

    # p值：交通流稳定性指标（基于速度和密度的组合）
    df['p_value'] = (df['density'] / df['velocity']) * 100

    return df

def create_congestion_target(df):
    """创建拥堵状态目标变量"""
    # 基于多个特征创建拥堵状态
    # 使用需求容量比和速度作为主要指标
    df['congestion'] = 0  # 0: 自由流

    # 定义拥堵条件
    # 条件1：需求容量比 > 0.9
    # 条件2：速度 < 15英里/小时
    # 条件3：密度 > 500辆车/英里/车道
    congestion_condition = (
        (df['demand_capacity_ratio'] > 0.9) |
        (df['velocity'] < 15) |
        (df['density'] > 500)
    )

    df.loc[congestion_condition, 'congestion'] = 1  # 1: 拥堵

    return df

def prepare_features_and_target(df):
    """准备特征和目标变量"""
    # 特征列表
    features = [
        'length',  # 路段长度
        'lanes_incr',  # 递增方向车道数
        'lanes_decr',  # 递减方向车道数
        'peak_traffic',  # 峰值交通量
        'demand_capacity_ratio',  # 需求容量比
        'p_value',  # 交通流稳定性指标
        'is_bottleneck',  # 是否为瓶颈
        'route_type',  # 路线类型（0:州级公路, 1:州际公路）
        'avg_lanes'  # 平均车道数
    ]

    # 目标变量
    target = 'congestion'

    # 提取特征和目标
    X = df[features]
    y = df[target]

    return X, y, features

def train_random_forest_model(X, y):
    """训练随机森林模型"""
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    # 创建随机森林分类器
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )

    # 训练模型
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # 评估模型
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['自由流', '拥堵'])
    cm = confusion_matrix(y_test, y_pred)

    print("=== 随机森林拥堵分类器训练结果 ===\n")
    print(f"模型准确率：{accuracy:.4f}")
    print("\n分类报告：")
    print(report)
    print("\n混淆矩阵：")
    print(cm)

    return model, scaler, X_train, X_test, y_train, y_test, y_pred, y_pred_proba

def visualize_model_results(model, X_test, y_test, y_pred, y_pred_proba, features):
    """可视化模型结果"""
    plt.figure(figsize=(18, 12))

    # 1. 特征重要性
    plt.subplot(2, 3, 1)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.bar(range(X_test.shape[1]), importances[indices], align='center')
    plt.xticks(range(X_test.shape[1]), [features[i] for i in indices], rotation=45, ha='right')
    plt.title('特征重要性')
    plt.xlabel('特征')
    plt.ylabel('重要性')
    plt.tight_layout()

    # 2. 混淆矩阵可视化
    plt.subplot(2, 3, 2)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['自由流', '拥堵'],
                yticklabels=['自由流', '拥堵'])
    plt.title('混淆矩阵')
    plt.xlabel('预测值')
    plt.ylabel('实际值')

    # 3. 预测概率分布
    plt.subplot(2, 3, 3)
    plt.hist(y_pred_proba[y_test == 0], bins=20, alpha=0.5, label='自由流', color='green')
    plt.hist(y_pred_proba[y_test == 1], bins=20, alpha=0.5, label='拥堵', color='red')
    plt.title('预测概率分布')
    plt.xlabel('拥堵概率')
    plt.ylabel('频数')
    plt.legend()

    # 4. ROC曲线
    plt.subplot(2, 3, 4)
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title('ROC曲线')
    plt.legend(loc="lower right")

    # 5. 特征相关性热图
    plt.subplot(2, 3, 5)
    # 将numpy数组转换为DataFrame
    X_test_df = pd.DataFrame(X_test, columns=features)
    y_test_series = pd.Series(y_test, name='congestion')
    X_with_target = pd.concat([X_test_df, y_test_series], axis=1)
    corr_matrix = X_with_target.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('特征相关性热图')

    # 6. 拥堵预测与实际对比
    plt.subplot(2, 3, 6)
    plt.scatter(range(len(y_test)), y_test, alpha=0.6, label='实际值', color='blue')
    plt.scatter(range(len(y_pred)), y_pred, alpha=0.6, label='预测值', color='orange', marker='x')
    plt.title('拥堵预测与实际对比')
    plt.xlabel('样本索引')
    plt.ylabel('拥堵状态')
    plt.legend()
    plt.ylim(-0.5, 1.5)

    plt.tight_layout()
    plt.savefig('congestion_classifier_results.png')
    print("\n模型结果可视化已保存为 'congestion_classifier_results.png'")

def analyze_model_performance(model, df, features):
    """分析模型性能和特征影响"""
    print("\n=== 模型性能分析 ===\n")

    # 特征重要性分析
    print("1. 特征重要性排名：")
    importances = model.feature_importances_
    feature_importance = pd.DataFrame({
        '特征': features,
        '重要性': importances
    }).sort_values('重要性', ascending=False)

    for idx, row in feature_importance.iterrows():
        print(f"   {row['特征']}: {row['重要性']:.4f}")

    # 模型参数分析
    print("\n2. 模型参数：")
    print(f"   树的数量: {model.n_estimators}")
    print(f"   最大深度: {model.max_depth}")
    print(f"   类权重: {model.class_weight}")

    # 拥堵路段分析
    print("\n3. 拥堵路段分析：")
    congested_segments = df[df['congestion'] == 1]
    print(f"   拥堵路段总数: {len(congested_segments)}")
    print(f"   拥堵路段比例: {len(congested_segments)/len(df)*100:.1f}%")

    if len(congested_segments) > 0:
        print("\n   典型拥堵路段特征：")
        for idx, row in congested_segments.nlargest(3, 'demand_capacity_ratio').iterrows():
            print(f"     - 路线 {row['Route_ID']}，里程碑 {row['startMilepost']:.2f}-{row['endMilepost']:.2f}：")
            print(f"       需求容量比: {row['demand_capacity_ratio']:.3f}")
            print(f"       速度: {row['velocity']:.1f} 英里/小时")
            print(f"       密度: {row['density']:.1f} 辆车/英里/车道")

def save_model(model, scaler):
    """保存模型和标准化器"""
    joblib.dump(model, 'congestion_classifier_model.pkl')
    joblib.dump(scaler, 'congestion_classifier_scaler.pkl')
    print("\n模型和标准化器已保存为:")
    print("  - congestion_classifier_model.pkl")
    print("  - congestion_classifier_scaler.pkl")

def main():
    """主函数"""
    print("=== 随机森林拥堵分类器 (ML1) ===\n")

    # 1. 加载和准备数据
    print("1. 加载和准备数据...")
    df = load_and_prepare_data()

    # 2. 计算交通特征
    print("2. 计算交通特征...")
    df = calculate_traffic_features(df)

    # 3. 创建拥堵状态目标变量
    print("3. 创建拥堵状态目标变量...")
    df = create_congestion_target(df)

    # 4. 准备特征和目标变量
    print("4. 准备特征和目标变量...")
    X, y, features = prepare_features_and_target(df)

    # 显示数据集信息
    print(f"\n数据集信息：")
    print(f"  总样本数: {len(X)}")
    print(f"  特征数: {len(features)}")
    print(f"  拥堵路段比例: {y.sum()/len(y)*100:.1f}%")
    print(f"  自由流路段比例: {(1-y.sum()/len(y))*100:.1f}%")

    # 5. 训练随机森林模型
    print("\n5. 训练随机森林模型...")
    model, scaler, X_train, X_test, y_train, y_test, y_pred, y_pred_proba = train_random_forest_model(X, y)

    # 6. 可视化模型结果
    print("\n6. 可视化模型结果...")
    visualize_model_results(model, X_test, y_test, y_pred, y_pred_proba, features)

    # 7. 分析模型性能
    analyze_model_performance(model, df, features)

    # 8. 保存模型
    save_model(model, scaler)

    print("\n=== 随机森林拥堵分类器搭建完成 ===")
    print("\n主要输出文件：")
    print("  - congestion_classifier_results.png (可视化结果)")
    print("  - congestion_classifier_model.pkl (训练好的模型)")
    print("  - congestion_classifier_scaler.pkl (标准化器)")

if __name__ == "__main__":
    main()