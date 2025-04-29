import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
# 设置字体，以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 读取数据
data = pd.read_excel('merged_materials.xlsx')

# 1. 可视化单个自变量对因变量的影响（小提琴图）
def plot_single_variable_violin(var):
    plt.figure(figsize=(8, 6))
    sns.violinplot(x=var, y='磁芯损耗', data=data)
    plt.title(f'磁芯损耗与{var}的关系（小提琴图）')
    plt.xlabel(var)
    plt.ylabel('磁芯损耗')
    plt.show()

# 分别可视化每个自变量
plot_single_variable_violin('温度')
plot_single_variable_violin('材料')
plot_single_variable_violin('励磁波形')

# 2. 可视化不同自变量的平均磁芯损耗（条形图）
def plot_average_loss(var):
    avg_loss = data.groupby(var)['磁芯损耗'].mean().reset_index()
    plt.figure(figsize=(8, 6))
    sns.barplot(x=var, y='磁芯损耗', data=avg_loss)
    plt.title(f'不同{var}下的平均磁芯损耗')
    plt.xlabel(var)
    plt.ylabel('平均磁芯损耗')
    plt.show()

# 分别可视化每个自变量的平均磁芯损耗
plot_average_loss('温度')
plot_average_loss('材料')
plot_average_loss('励磁波形')

# 3. 可视化两两自变量的交互影响（热图）
def plot_interaction_heatmap(var1, var2):
    heatmap_data = data.groupby([var1, var2])['磁芯损耗'].mean().unstack()
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlGnBu')
    plt.title(f'{var1}与{var2}的交互作用下的平均磁芯损耗')
    plt.xlabel(var2)
    plt.ylabel(var1)
    plt.show()

# 可视化交互影响的热图
plot_interaction_heatmap('温度', '材料')
plot_interaction_heatmap('温度', '励磁波形')
plot_interaction_heatmap('材料', '励磁波形')

def encode_categories(data, cols):
    mappings = {}
    for col in cols:
        new_col_name = f"{col}_coded"
        data[new_col_name] = data[col].astype('category').cat.codes
        mappings[col] = dict(enumerate(data[col].astype('category').cat.categories))
    return data, mappings

# 对 '材料' 和 '励磁波形' 进行编码
data, mappings = encode_categories(data, ['材料', '励磁波形'])
# 输出编码映射关系
for col, mapping in mappings.items():
    print(f"{col} 的编码映射:")
    for code, category in mapping.items():
        print(f"{code}: {category}")
    print()


def plot_3d(var1, var2, target):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 生成编码后的数据
    if 'coded' in var1:
        original_var1 = var1.replace('_coded', '')
        x_data = data[var1]
        x_labels = list(mappings[original_var1].values())
    else:
        x_data = data[var1]
        x_labels = []

    if 'coded' in var2:
        original_var2 = var2.replace('_coded', '')
        y_data = data[var2]
        y_labels = list(mappings[original_var2].values())
    else:
        y_data = data[var2]
        y_labels = []

    # 绘制三维散点图
    ax.scatter(x_data, y_data, data[target], c='b', marker='o')

    # 设置轴标签
    ax.set_xlabel(original_var1 if 'coded' in var1 else var1)
    ax.set_ylabel(original_var2 if 'coded' in var2 else var2)
    ax.set_zlabel(target)
    ax.set_title(f'三维散点图: {original_var1 if "coded" in var1 else var1}与{original_var2 if "coded" in var2 else var2}对{target}的影响')

    # 设置刻度标签
    if x_labels:
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45)
    if y_labels:
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)

    plt.show()

# 可视化不同自变量的三维影响
plot_3d('温度', '材料_coded', '磁芯损耗')
plot_3d('温度', '励磁波形_coded', '磁芯损耗')
plot_3d('材料_coded', '励磁波形_coded', '磁芯损耗')



