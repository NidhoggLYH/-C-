import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from minepy import MINE
'''
# 读取数据
df = pd.read_excel('data/四种材料整合_提取特征_材料波形编码.xlsx')

# 选择特征和目标
features = df[['温度，oC', '频率，Hz', '励磁波形', '材料类型', '平均值', '标准差', '最大值', '峰峰值', '均方根值', '波形因子', '峰值因子', '偏度', '峰度', '最大频率','能量','频谱中心', '频谱平坦度']]
target = df['磁芯损耗，w/m3']

# 将特征和目标值合并为一个DataFrame
data = pd.concat([features, target], axis=1)

# 计算相关性矩阵
correlation_matrix = data.corr()

# 打印与目标值的相关性系数并排序
correlation_with_target = correlation_matrix['磁芯损耗，w/m3']
sorted_correlation = correlation_with_target.sort_values(ascending=False)
print("与目标值的相关性系数（按大小排序）：")
print(sorted_correlation)

# 设置字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 负号正常显示

# 绘制热力图
plt.figure(figsize=(12, 10))
heatmap = sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.8, linecolor='black',
                      cbar_kws={"shrink": .8}, square=True)
plt.title('皮尔逊相关性热力图', fontsize=16)
plt.xticks(fontsize=12, rotation=45)  # 设置下方字体倾斜
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()
'''


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import spearmanr

# 读取数据
df = pd.read_excel('data/四种材料整合_提取特征_材料波形编码.xlsx')

# 选择特征和目标
features = df[['温度，oC', '频率，Hz', '励磁波形', '材料类型', '平均值', '标准差', '最大值', '峰峰值', '均方根值', '波形因子', '峰值因子', '偏度', '峰度', '最大频率','能量','频谱中心', '频谱平坦度']]
target = df['磁芯损耗，w/m3']

# 将特征和目标值合并为一个DataFrame
data = pd.concat([features, target], axis=1)

# 计算斯皮尔曼相关性矩阵
spearman_corr_matrix = data.corr(method='spearman')

# 打印与目标值的相关性系数并排序
correlation_with_target = spearman_corr_matrix['磁芯损耗，w/m3']
sorted_correlation = correlation_with_target.sort_values(ascending=False)
print("与目标值的相关性系数（按大小排序）：")
print(sorted_correlation)

# 计算MIC
mine = MINE()
mic_scores = {}

for feature in features.columns:
    mine.compute_score(features[feature], target)
    mic = mine.mic()
    mic_scores[feature] = mic

# 输出MIC得分并排序
sorted_mic_scores = sorted(mic_scores.items(), key=lambda x: x[1], reverse=True)
print("\n与目标值的MIC得分（按大小排序）：")
for feature, mic in sorted_mic_scores:
    print(f"{feature} 的MIC: {mic:.4f}")


# 设置字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 负号正常显示

# 绘制斯皮尔曼相关性热力图
plt.figure(figsize=(12, 10))
heatmap = sns.heatmap(spearman_corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.8, linecolor='black',
                      cbar_kws={"shrink": .8}, square=True)
plt.title('斯皮尔曼相关性热力图', fontsize=16)
plt.xticks(fontsize=12, rotation=45)  # 设置下方字体倾斜
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()
