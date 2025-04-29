import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib

# 读取数据
df = pd.read_excel('data/四种材料整合_提取特征_材料波形编码.xlsx')

# 选择特征和目标
features = df[['温度，oC', '频率，Hz', '励磁波形', '材料类型', '平均值', '标准差', '最大值', '峰峰值', '均方根值', '波形因子', '峰值因子', '偏度', '峰度', '最大频率', '能量', '频谱中心', '频谱平坦度', '频谱熵']]
target = df['磁芯损耗，w/m3']

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# 训练随机森林模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 设置字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 负号正常显示

# 获取特征重要性
importances = model.feature_importances_

# 创建特征重要性数据框
feature_importance_df = pd.DataFrame({'Feature': features.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# 打印特征重要性
print(feature_importance_df)

# 可视化特征重要性
plt.figure(figsize=(12, 8))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='lightcoral')
plt.xlabel('重要性', fontsize=14)
plt.ylabel('特征', fontsize=14)
plt.title('特征重要性来自 RF', fontsize=16)
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)  # 添加网格线
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()
