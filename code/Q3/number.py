import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import glm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # 导入 seaborn

# 设置字体，以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取数据
data = pd.read_excel('merged_materials.xlsx')

# 更新模型列表，确保链接函数适合其分布
models = {
    'Gamma-Log': ('Gamma', sm.families.links.Log),
    'InverseGaussian-Log': ('InverseGaussian', sm.families.links.Log),
    'Normal-Identity': ('Gaussian', sm.families.links.Identity)
}

# 存储结果
results = []

# 循环各个模型
for name, (family, link_func) in models.items():
    try:
        # 选择模型族
        family_obj = {
            'Gamma': sm.families.Gamma(link=link_func()),
            'InverseGaussian': sm.families.InverseGaussian(link=link_func()),
            'Gaussian': sm.families.Gaussian(link=link_func())
        }[family]

        # 拟合模型，包括所有自变量及其交互项
        model = glm(
            '磁芯损耗 ~ C(温度) * C(材料) * C(励磁波形)',  # 包括所有交互项
            data=data,
            family=family_obj).fit()

        # 计算 BIC
        k = model.params.shape[0]  # 参数数量
        n = len(data)  # 样本数量
        bic = -2 * model.llf + k * np.log(n)  # 手动计算 BIC

        # 存储结果
        results.append({
            'Model': name,
            'AIC': model.aic,
            'BIC': bic,
            'LLF': model.llf,
            'ModelFit': model
        })

        # 输出每个模型的影响程度
        print(f"\n模型 {name} 的影响程度（系数及显著性水平）：")
        print(model.summary())

        # 绘制残差图
        plt.figure(figsize=(10, 6))
        plt.scatter(model.fittedvalues, model.resid_response, alpha=0.5, label='残差')
        plt.axhline(0, color='red', linestyle='--', label='零线')

        # 使用 seaborn 拟合平滑曲线
        sns.regplot(x=model.fittedvalues, y=model.resid_response, lowess=True,
                    scatter=False, color='red', label='平滑拟合线')

        plt.title(f'Residuals vs Fitted for {name}')
        plt.xlabel('Fitted values')
        plt.ylabel('Residuals')
        plt.legend()
        plt.show()

    except Exception as e:
        print(f"Error with model {name}: {str(e)}")

# 转换为DataFrame并输出
results_df = pd.DataFrame(results)
best_model_info = results_df.sort_values(by=['AIC', 'BIC']).iloc[0]

# 输出最优模型的信息
print(best_model_info)

# 可视化AIC和BIC
plt.figure(figsize=(10, 6))
colors = ['blue', 'red']
results_df.set_index('Model', inplace=True)
results_df[['AIC', 'BIC']].plot(kind='bar', color=colors)
plt.ylabel('Criteria Value')
plt.title('Comparison of Model Criteria (AIC & BIC)')
plt.xticks(rotation=45)
plt.show()
