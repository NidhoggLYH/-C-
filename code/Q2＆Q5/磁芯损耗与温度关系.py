import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from minepy import MINE
import matplotlib.pyplot as plt
import matplotlib

# 读取数据
df = pd.read_excel('data/材料1_正弦波_带磁通密度峰值.xlsx')

# 选择两列
col1 = df['温度，oC']
col2 = df['磁芯损耗，w/m3']

# 设置字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 负号正常显示

# 绘制散点图
plt.figure(figsize=(8, 6))
sns.scatterplot(x=col1, y=col2)
plt.title(f' 温度 和 磁芯损耗 散点图')
plt.xlabel('温度（℃）')
plt.ylabel('磁芯损耗（w/m3）')
plt.grid(True)
plt.show()

# 计算皮尔森相关系数
pearson_corr, _ = pearsonr(col1, col2)
print(f"Pearson correlation: {pearson_corr:.4f}")

# 计算斯皮尔曼相关系数
spearman_corr, _ = spearmanr(col1, col2)
print(f"Spearman correlation: {spearman_corr:.4f}")

# 计算MIC
mine = MINE()
mine.compute_score(col1, col2)
mic = mine.mic()
print(f"MIC: {mic:.4f}")
