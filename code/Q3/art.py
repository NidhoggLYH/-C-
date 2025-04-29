# 导入必要的库
import pandas as pd
import numpy as np
from scipy.stats import rankdata, shapiro
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# 设置字体，以支持中文显示（可选）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 忽略警告信息
warnings.filterwarnings('ignore')

# 一、数据导入和预处理

# 读取数据
data = pd.read_excel('merged_materials.xlsx')

# 检查数据是否有缺失值
print("缺失值统计：")
print(data.isnull().sum())

# 将分类变量转换为字符串类型
data['Temperature'] = data['温度'].astype(str)
data['Waveform'] = data['励磁波形'].astype(str)
data['Material'] = data['材料'].astype(str)

# 提取响应变量和因素
response = '磁芯损耗'
factors = ['Temperature', 'Waveform', 'Material']

# 二、对齐步骤（Alignment）

def align_data(data, response, factors):
    aligned_data = data.copy()
    effects = []

    for factor in factors:
        effects.append([factor])

    for i in range(len(factors)):
        for j in range(i + 1, len(factors)):
            effects.append([factors[i], factors[j]])

    effects.append(factors)

    for effect_factors in effects:
        effect_terms = [f'C({factor})' for factor in effect_factors]
        formula = f'{response} ~ {" + ".join(effect_terms)}'
        model = ols(formula, data=data).fit()
        residuals = model.resid + model.params['Intercept']
        effect_name = '_'.join(effect_factors)
        aligned_col = f'Aligned_{effect_name}'
        aligned_data[aligned_col] = residuals
        aligned_data[f'Rank_{effect_name}'] = rankdata(aligned_data[aligned_col])

    return aligned_data, effects

aligned_data, effects = align_data(data, response, factors)

# 三、ANOVA 分析

anova_results = {}

for effect_factors in effects:
    effect_name = '_'.join(effect_factors)
    rank_col = f'Rank_{effect_name}'
    formula_terms = [f'C({factor})' for factor in effect_factors]
    formula = f'{rank_col} ~ {"*".join(formula_terms)}'
    model = ols(formula, data=aligned_data).fit()
    anova_table = anova_lm(model, typ=2)
    anova_results[effect_name] = anova_table

    print(f'\nANOVA Results for {effect_name}:')
    print(anova_table)

# 四、结果解释

for effect_factors in effects:
    effect_name = '_'.join(effect_factors)
    anova_table = anova_results[effect_name]
    print(f'\nEffect: {effect_name}')
    print(anova_table[['F', 'PR(>F)']])

# 五、可视化交互效应

plt.figure(figsize=(10, 6))
sns.boxplot(x='Temperature', y=response, data=data)
plt.title('磁芯损耗 vs 温度')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Waveform', y=response, data=data)
plt.title('磁芯损耗 vs 励磁波形')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Material', y=response, data=data)
plt.title('磁芯损耗 vs 材料')
plt.show()

plt.figure(figsize=(10, 6))
sns.pointplot(x='Temperature', y=response, hue='Material', data=data, dodge=True, markers='o', capsize=0.1)
plt.title('Interaction Plot: Temperature and Material')
plt.xlabel('Temperature')
plt.ylabel(response)
plt.legend(title='Material')
plt.show()

plt.figure(figsize=(10, 6))
sns.pointplot(x='Temperature', y=response, hue='Waveform', data=data, dodge=True, markers='o', capsize=0.1)
plt.title('Interaction Plot: Temperature and Waveform')
plt.xlabel('Temperature')
plt.ylabel(response)
plt.legend(title='Waveform')
plt.show()

plt.figure(figsize=(10, 6))
sns.pointplot(x='Waveform', y=response, hue='Material', data=data, dodge=True, markers='o', capsize=0.1)
plt.title('Interaction Plot: Waveform and Material')
plt.xlabel('Waveform')
plt.ylabel(response)
plt.legend(title='Material')
plt.show()

# 六、有效性分析

# 残差分析
residuals = model.resid

# 绘制残差图
plt.figure(figsize=(10, 6))
sns.residplot(x=model.fittedvalues, y=residuals, lowess=True, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.axhline(0, linestyle='--', color='red')
plt.show()

# 正态性检验
shapiro_test = shapiro(residuals)
print(f'\nShapiro-Wilk test for normality: Statistic={shapiro_test.statistic}, p-value={shapiro_test.pvalue}')

# 正态性Q-Q图
sm.qqplot(residuals, line='s')
plt.title('Q-Q Plot of Residuals')
plt.show()

# 同方差性检验
plt.figure(figsize=(10, 6))
sns.scatterplot(x=model.fittedvalues, y=np.square(residuals))
plt.xlabel('Fitted Values')
plt.ylabel('Squared Residuals')
plt.title('Check for Homoscedasticity')
plt.axhline(0, linestyle='--', color='red')
plt.show()
