import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

sns.set(style='white', font_scale=1.2)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取训练集数据
train_data_path = r"D:\2024数学建模\解压后\C题\四种材料特征提取.xlsx"
df = pd.read_excel(train_data_path)

print("训练集 DataFrame 的列名:")
for idx, col in enumerate(df.columns):
    print(f"{idx}: '{col}'")

df.columns = df.columns.str.strip()
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

features = ['温度，oC', '频率，Hz', '励磁波形', '材料类型', '标准差', '最大值', '峰峰值', '最大频率', '偏度']
missing_features = [feat for feat in features if feat not in df.columns]
if missing_features:
    raise KeyError(f"缺少以下特征列: {missing_features}")

X = df[features].copy()

# 训练集数据预处理
if X.isnull().any().any():
    print("训练集数据存在缺失值，进行填充")
    X = X.fillna(X.mean())

numeric_features = ['温度，oC', '频率，Hz', '标准差', '最大值', '峰峰值', '最大频率', '偏度']
categorical_features = ['励磁波形', '材料类型']

scaler_X = StandardScaler()
X[numeric_features] = scaler_X.fit_transform(X[numeric_features])

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

X = pd.get_dummies(X, columns=categorical_features)
joblib.dump(scaler_X, 'scaler_X_RF.save')
joblib.dump(scaler_y, 'scaler_y_RF.save')

# 数据集划分
X_train = pd.DataFrame()
X_val = pd.DataFrame()
y_train = np.array([])
y_val = np.array([])

for material in df['材料类型'].unique():
    X_material = X[df['材料类型'] == material]
    y_material = y_scaled[df['材料类型'] == material]

    X_train_m, X_val_m, y_train_m, y_val_m = train_test_split(X_material, y_material, test_size=0.1, random_state=42)

    X_train = pd.concat([X_train, X_train_m], axis=0)
    X_val = pd.concat([X_val, X_val_m], axis=0)
    y_train = np.concatenate([y_train, y_train_m])
    y_val = np.concatenate([y_val, y_val_m])

X_train.reset_index(drop=True, inplace=True)
X_val.reset_index(drop=True, inplace=True)

y_train = pd.Series(y_train)
y_val = pd.Series(y_val)

# 超参数调优和模型训练
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}

rf_model = RandomForestRegressor(random_state=42)
random_search = RandomizedSearchCV(estimator=rf_model, param_distributions=param_dist, n_iter=50,
                                   scoring='neg_mean_squared_error', cv=5, verbose=1, random_state=42, n_jobs=-1)

train_start_time = time.time()
random_search.fit(X_train, y_train)
train_end_time = time.time()
train_time = train_end_time - train_start_time

print("最佳参数：", random_search.best_params_)
print("最佳得分（负的MSE）：", -random_search.best_score_)
print(f"训练时间：{train_time:.2f} 秒")

best_model = random_search.best_estimator_

# 模型评估
y_pred_scaled = best_model.predict(X_val)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_val_original = scaler_y.inverse_transform(y_val.values.reshape(-1, 1)).flatten()

mae = mean_absolute_error(y_val_original, y_pred)
rmse = np.sqrt(mean_squared_error(y_val_original, y_pred))
r2 = r2_score(y_val_original, y_pred)

print(f"验证集上的平均绝对误差（MAE）：{mae}")
print(f"验证集上的均方根误差（RMSE）：{rmse}")
print(f"验证集上的R²：{r2}")

results = pd.DataFrame({'实际磁芯损耗': y_val_original, '预测磁芯损耗': y_pred})
print(results.head())

plt.figure(figsize=(10, 8))
sns.scatterplot(x='实际磁芯损耗', y='预测磁芯损耗', data=results, color='steelblue', alpha=0.7, edgecolor=None, s=60)
plt.plot([y_val_original.min(), y_val_original.max()], [y_val_original.min(), y_val_original.max()], 'r--', linewidth=2)
plt.xlabel('实际磁芯损耗', fontsize=14)
plt.ylabel('预测磁芯损耗', fontsize=14)
plt.title('验证集上实际值与预测值的对比（随机森林）', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(False)
plt.tight_layout()
plt.show()

results_sorted = results.reset_index(drop=True)
plt.figure(figsize=(10, 8), facecolor='white')
ax = plt.gca()
ax.set_facecolor('white')

plt.plot(results_sorted.index.values, results_sorted['实际磁芯损耗'].values, label='实际值', color='green', linewidth=2)
plt.plot(results_sorted.index.values, results_sorted['预测磁芯损耗'].values, label='预测值', color='orange', linewidth=2)
plt.xlabel('样本索引', fontsize=14)
plt.ylabel('磁芯损耗', fontsize=14)
plt.title('验证集上实际值与预测值对比（随机森林）', fontsize=16)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 测试集预测
test_data_path = r"D:\2024数学建模\解压后\C题\附件三特征1（测试集）.xlsx"
test_df = pd.read_excel(test_data_path)

print("测试集 DataFrame 的列名:")
for idx, col in enumerate(test_df.columns):
    print(f"{idx}: '{col}'")

test_df.columns = test_df.columns.str.strip()
test_X = test_df[features].copy()

if test_X.isnull().any().any():
    print("测试集数据存在缺失值，进行填充")
    test_X = test_X.fillna(test_X.mean())

test_X = pd.get_dummies(test_X, columns=categorical_features)
test_X[numeric_features] = scaler_X.transform(test_X[numeric_features])

missing_cols = set(X_train.columns) - set(test_X.columns)
for col in missing_cols:
    test_X[col] = 0
test_X = test_X[X_train.columns]

test_start_time = time.time()
test_predictions_scaled = best_model.predict(test_X)
test_end_time = time.time()
test_time = test_end_time - test_start_time

print(f"测试集预测时间：{test_time:.2f} 秒")

test_predictions = scaler_y.inverse_transform(test_predictions_scaled.reshape(-1, 1)).flatten()
test_predictions = np.round(test_predictions, 1)
test_df['预测的磁芯损耗'] = test_predictions

output_path = r"D:\2024数学建模\解压后\C题\测试集预测结果随机森林1.xlsx"

try:
    test_df.to_excel(output_path, index=False)
    print(f"预测结果已保存到 {output_path}")
except PermissionError:
    print(f"无法保存文件到 {output_path}。请确保文件没有被其他程序占用，并且你有写入权限。")
