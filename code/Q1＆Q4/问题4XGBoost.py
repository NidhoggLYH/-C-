import pandas as pd
import numpy as np
import time  # 导入时间模块

# 导入预处理和模型相关的库
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# 导入绘图库
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# 导入保存模型的库
import joblib
import os

# 设置Seaborn风格为白色背景
sns.set(style='white', font_scale=1.2)

# =========================
# 设置中文字体以避免乱码
# =========================
# 方法一：使用系统已安装的中文字体，例如 SimHei（黑体）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 替换为你系统中存在的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 方法二：如果方法一不起作用，可以指定字体路径
# from matplotlib.font_manager import FontProperties
# font_path = 'C:/Windows/Fonts/simhei.ttf'  # 根据实际情况修改路径
# font_prop = FontProperties(fname=font_path)
# plt.rcParams['font.family'] = font_prop.get_name()
# plt.rcParams['axes.unicode_minus'] = False

# ========== 1. 读取训练集数据 ==========
# 训练集数据路径
train_data_path = r"D:\2024数学建模\解压后\C题\四种材料特征提取.xlsx"
df = pd.read_excel(train_data_path)

# 2. 检查DataFrame的列名
print("训练集 DataFrame 的列名:")
for idx, col in enumerate(df.columns):
    print(f"{idx}: '{col}'")

# 去除列名的前后空格
df.columns = df.columns.str.strip()

# 查找包含 '磁芯损耗' 的列名
target_columns = [col for col in df.columns if '磁芯损耗' in col]
print("包含 '磁芯损耗' 的列名:", target_columns)

if len(target_columns) == 1:
    target = target_columns[0]
    print(f"目标列名设定为: {target}")
    y = df[target]
elif len(target_columns) > 1:
    raise ValueError("找到多个包含 '磁芯损耗' 的列，请手动选择正确的列名。")
else:
    raise KeyError("未找到包含 '磁芯损耗' 的列。")

# 3. 选择指定的特征和目标变量
features = ['温度，oC', '频率，Hz', '励磁波形', '材料类型', '标准差', '最大值', '峰峰值', '最大频率', '偏度']

# 检查特征列是否存在
missing_features = [feat for feat in features if feat not in df.columns]
if missing_features:
    raise KeyError(f"缺少以下特征列: {missing_features}")

X = df[features].copy()  # 使用 .copy() 确保 X 是 df 的副本

# ========== 2. 训练集数据预处理 ==========
# 检查缺失值
if X.isnull().any().any():
    print("训练集数据存在缺失值，进行填充")
    X = X.fillna(X.mean())

# 标准化数值特征，包括所有特征
numeric_features = ['温度，oC', '频率，Hz', '标准差', '最大值', '峰峰值', '最大频率', '偏度']  # 排除非数值特征
categorical_features = ['励磁波形', '材料类型']  # 非数值特征

# 对数值特征进行标准化
scaler_X = StandardScaler()
X[numeric_features] = scaler_X.fit_transform(X[numeric_features])

# 对目标变量进行归一化
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

# 保存标准化器
joblib.dump(scaler_X, 'scaler_X.save')
joblib.dump(scaler_y, 'scaler_y.save')

# ========== 3. 数据集划分 ==========
# 初始化训练集和验证集
X_train = pd.DataFrame()
X_val = pd.DataFrame()
y_train = np.array([])
y_val = np.array([])

# 按照材料类型分组，然后从每组中随机抽取20%作为验证集
for material in X['材料类型'].unique():
    X_material = X[X['材料类型'] == material]
    y_material = y_scaled[X['材料类型'] == material]

    # 使用train_test_split按比例划分
    X_train_m, X_val_m, y_train_m, y_val_m = train_test_split(
        X_material, y_material, test_size=0.1, random_state=42)

    # 合并数据
    X_train = pd.concat([X_train, X_train_m], axis=0)
    X_val = pd.concat([X_val, X_val_m], axis=0)
    y_train = np.concatenate([y_train, y_train_m])
    y_val = np.concatenate([y_val, y_val_m])

# 重置索引
X_train.reset_index(drop=True, inplace=True)
X_val.reset_index(drop=True, inplace=True)

# 转换为适合模型输入的格式
y_train = pd.Series(y_train)
y_val = pd.Series(y_val)

# ========== 4. 超参数调优和模型训练 ==========
# 定义参数调优范围
param_dist = {
    'n_estimators': [400, 500,600,700,800],
    'max_depth': [3,4, 5,6, 7, 8,9],
    'learning_rate': [0.01, 0.05, 0.1, 0.15,0.2],
    'subsample': [0.6,0.7, 0.8,0.9, 1.0],
    'colsample_bytree': [0.6,0.7, 0.8,0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3],
    'reg_alpha': [0, 0.01, 0.05,0.1,0.5, 1],
    'reg_lambda': [0, 0.01, 0.1, 1],
}

# 初始化XGBoost模型
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# 使用RandomizedSearchCV进行超参数调优
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=60,  # 迭代次数，可根据需要调整
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1  # 使用所有可用的CPU核心
)

# 记录训练开始时间
train_start_time = time.time()

# 进行调参
random_search.fit(X_train, y_train)

# 记录训练结束时间
train_end_time = time.time()
train_time = train_end_time - train_start_time

print("最佳参数：", random_search.best_params_)
print("最佳得分（负的MSE）：", -random_search.best_score_)
print(f"训练时间：{train_time:.2f} 秒")

# 使用最佳参数训练模型
best_model = random_search.best_estimator_

# 保存模型
model_filename = 'best_xgboost_model.model'
joblib.dump(best_model, model_filename)

# ========== 5. 模型评估 ==========
# 在验证集上预测
y_pred_scaled = best_model.predict(X_val)

# 反归一化预测值和实际值
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_val_original = scaler_y.inverse_transform(y_val.values.reshape(-1, 1)).flatten()

# 计算评估指标
mse = mean_squared_error(y_val_original, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_val_original, y_pred)

print(f"验证集上的均方误差（MSE）：{mse}")
print(f"验证集上的均方根误差（RMSE）：{rmse}")
print(f"验证集上的R²：{r2}")

# 查看预测值和实际值的对比
results = pd.DataFrame({'实际磁芯损耗': y_val_original, '预测磁芯损耗': y_pred})
print(results.head())

# 绘制验证集上真实值与预测值的对比图
plt.figure(figsize=(10, 8))
sns.scatterplot(x='实际磁芯损耗', y='预测磁芯损耗', data=results, color='steelblue', alpha=0.7, edgecolor=None, s=60)
plt.plot([y_val_original.min(), y_val_original.max()], [y_val_original.min(), y_val_original.max()], 'r--',
         linewidth=2)  # 参考线
plt.xlabel('实际磁芯损耗', fontsize=14)
plt.ylabel('预测磁芯损耗', fontsize=14)
plt.title('验证集上实际值与预测值的对比', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(False)  # 关闭网格线
plt.tight_layout()
plt.show()

# ========== 新增部分：按样本顺序排列的实际值与预测值对比图 ==========
# 为了按样本顺序绘图，首先确保结果按验证集的索引排列
results_sorted = results.reset_index(drop=True)

# 创建一个新的Figure，并设置背景为白色
plt.figure(figsize=(10, 8), facecolor='white')  # 设置Figure背景为白色
ax = plt.gca()
ax.set_facecolor('white')  # 设置Axes背景为白色

plt.plot(results_sorted.index.values, results_sorted['实际磁芯损耗'].values, label='实际值', color='green', linewidth=2)
plt.plot(results_sorted.index.values, results_sorted['预测磁芯损耗'].values, label='预测值', color='orange', linewidth=2)
plt.xlabel('样本索引', fontsize=14)
plt.ylabel('磁芯损耗', fontsize=14)
plt.title('验证集上实际值与预测值对比', fontsize=16)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)  # 添加网格线
plt.tight_layout()
plt.show()
# ========== 新增部分结束 ==========

# ========== 6. 测试集预测 ==========
# 测试集数据路径
test_data_path = r"D:\2024数学建模\解压后\C题\附件三特征1（测试集）.xlsx"
test_df = pd.read_excel(test_data_path)

# 检查测试集的列名
print("测试集 DataFrame 的列名:")
for idx, col in enumerate(test_df.columns):
    print(f"{idx}: '{col}'")

# 去除列名的前后空格
test_df.columns = test_df.columns.str.strip()

# 选择与训练集相同的特征
test_X = test_df[features].copy()

# 检查缺失值
if test_X.isnull().any().any():
    print("测试集数据存在缺失值，进行填充")
    test_X = test_X.fillna(test_X.mean())

# 使用训练集的标准化器对测试集进行标准化
test_X[numeric_features] = scaler_X.transform(test_X[numeric_features])

# 记录测试开始时间
test_start_time = time.time()

# 使用训练好的模型进行预测（归一化后）
test_predictions_scaled = best_model.predict(test_X)

# 记录测试结束时间
test_end_time = time.time()
test_time = test_end_time - test_start_time

print(f"测试集预测时间：{test_time:.2f} 秒")

# 反归一化预测结果
test_predictions = scaler_y.inverse_transform(test_predictions_scaled.reshape(-1, 1)).flatten()

# 四舍五入到一位小数
test_predictions = np.round(test_predictions, 1)  # 方法一
# 或者使用 pandas 的 round 方法
# test_predictions = pd.Series(test_predictions).round(1)  # 方法二

# 打印 test_predictions 的类型和形状以确保正确
print("test_predictions 类型：", type(test_predictions))
print("test_predictions 形状：", test_predictions.shape)

# 将预测结果添加到测试集 DataFrame 中
test_df['预测的磁芯损耗'] = test_predictions

# 保存结果到新的 Excel 文件
output_path = r"D:\2024数学建模\解压后\C题\测试集预测结果XGBoost.xlsx"

try:
    test_df.to_excel(output_path, index=False)
    print(f"预测结果已保存到 {output_path}")
except PermissionError:
    print(f"无法保存文件到 {output_path}。请确保文件没有被其他程序占用，并且你有写入权限。")

# ========== 完整代码结束 ==========
