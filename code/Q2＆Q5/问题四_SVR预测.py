import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, root_mean_squared_error, max_error, r2_score, mean_absolute_error,mean_absolute_percentage_error
import matplotlib.pyplot as plt
import matplotlib

# 读取数据
df = pd.read_excel('data/问题提取的特征.xlsx')

# 选择特征和目标
features = df[['温度，oC', '频率，Hz', '励磁波形', '标准差', '最大值', '峰峰值', '均方根值', '最大频率', '能量']]
target = df['磁芯损耗，w/m3']

# 划分数据集
X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.3, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# 创建支持向量机模型
model = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
model.fit(X_train_scaled, y_train)

# 预测验证集
val_predictions = model.predict(X_val_scaled)

mse_test = mean_squared_error(y_val, val_predictions)
rmse_test = root_mean_squared_error(y_val, val_predictions)
max_test = max_error(y_val, val_predictions)
r2_test = r2_score(y_val, val_predictions)
mae_test = mean_absolute_error(y_val, val_predictions)
mape_test = mean_absolute_percentage_error(y_val, val_predictions)
# 输出误差
#print(f"修正训练集 MSE: {mse_train:.4f}")
#print(f"修正训练集 RMSE: {rmse_train:.4f}")
#print(f"修正训练集 MAX: {max_train:.4f}")

print(f"修正测试集 MSE: {mse_test:.4f}")
print(f"修正测试集 RMSE: {rmse_test:.4f}")
print(f"修正测试集 MAX: {max_test:.4f}")
print(f"修正测试集 R2: {r2_test:.4f}")
print(f"修正测试集 MAE: {mae_test:.4f}")
print(f"修正测试集 MAPE: {mape_test:.4f}")
# 设置字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 负号正常显示

# 可视化结果
plt.figure(figsize=(10,6))
#plt.scatter(range(len(y_pred_test_ori)), y_pred_test_ori, label='传统预测磁芯损耗', color='g')
plt.scatter(range(len(y_val)), y_val, label='实际磁芯损耗', color='b')
plt.scatter(range(len(val_predictions)), val_predictions, label='修正预测磁芯损耗', color='r')
plt.title('测试集表现')
plt.xlabel('采样点')
plt.ylabel('磁芯损耗')
plt.legend()
plt.grid(True)
plt.show()

# 残差
residuals = y_val - val_predictions
#residuals_ori = y_test - y_pred_test_ori
plt.figure(figsize=(10,6))
plt.scatter(range(len(residuals)), residuals, label='修正方程', color='r')
#plt.scatter(range(len(residuals)), residuals_ori, label='传统方程', color='g')
plt.title('预测残差')
plt.xlabel('采样点')
plt.ylabel('磁芯损耗')
plt.legend()
plt.grid(True)
plt.show()
