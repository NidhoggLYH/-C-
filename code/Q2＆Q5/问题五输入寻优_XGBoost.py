import numpy as np
import pandas as pd
import joblib
from pyswarm import pso
from sklearn.metrics import mean_squared_error, root_mean_squared_error, max_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import matplotlib

def objective(params):
    # 将参数解包为多个特征值
    temperature, frequency, B_max, W, Q, bzc, ffz, zdpl, pd = params

    # 确保将参数转换为二维数据结构（例如 DataFrame）
    input_features = pd.DataFrame({
        '温度，oC': [float(temperature)],
        '频率，Hz': [float(frequency)],
        '最大值': [float(B_max)],
        '励磁波形': [int(W)],
        '材料类型': [int(Q)],
        '标准差': [float(bzc)],
        '峰峰值': [float(ffz)],
        '最大频率': [float(zdpl)],
        '偏度': [float(pd)]
    })

    # 对特征进行标准化
    input_features[features] = scaler.transform(input_features[features])

    # 使用模型进行预测
    pred = best_model.predict(input_features)

    # 计算能量损耗
    energy_m = energy(frequency, B_max)

    # 计算目标值
    final_objective = pred[0] - energy_m

    return final_objective if final_objective > 0 else float('inf')  # 确保目标值非负

def energy(frequency, B_max):
    return frequency * B_max

def pso_optimize(bounds, num_particles=80, max_iter=300):
    particles = np.random.rand(num_particles, len(bounds))
    velocities = np.random.rand(num_particles, len(bounds))

    for i, (min_val, max_val) in enumerate(bounds):
        particles[:, i] = particles[:, i] * (max_val - min_val) + min_val

    personal_best = np.copy(particles)
    personal_best_scores = np.array([objective(p) for p in personal_best])
    global_best = personal_best[np.argmin(personal_best_scores)]

    for iteration in range(max_iter):
        for i in range(num_particles):
            r1, r2 = np.random.rand(2)
            velocities[i] = (
                    0.5 * velocities[i] +
                    r1 * (personal_best[i] - particles[i]) +
                    r2 * (global_best - particles[i])
            )
            particles[i] += velocities[i]

            for j, (min_val, max_val) in enumerate(bounds):
                particles[i, j] = np.clip(particles[i, j], min_val, max_val)

            score = objective(particles[i])
            if score < personal_best_scores[i]:
                personal_best[i] = particles[i]
                personal_best_scores[i] = score

        if np.min(personal_best_scores) < objective(global_best):
            global_best = personal_best[np.argmin(personal_best_scores)]

    return global_best

# 加载模型和标准化器
model_filename = 'best_xgboost_model.model'
scaler_filename = 'scaler.save'
scaler = joblib.load(scaler_filename)
best_model = joblib.load(model_filename)

# 测试集数据路径
test_data_path = 'data/四种材料整合_提取特征_材料波形编码.xlsx'
test_df = pd.read_excel(test_data_path)

# 去除列名的前后空格
test_df.columns = test_df.columns.str.strip()

features = ['温度，oC', '频率，Hz', '励磁波形', '材料类型', '标准差', '最大值', '峰峰值', '最大频率', '偏度']
test_X = test_df[features].copy()
y_test = test_df['磁芯损耗，w/m3'].copy()

# 检查缺失值并进行填充
if test_X.isnull().any().any():
    print("测试集数据存在缺失值，进行填充")
    test_X = test_X.fillna(test_X.mean())

# 使用训练集的标准化器对测试集进行标准化
test_X[features] = scaler.transform(test_X[features])

# 优化输入参数
bounds = [(test_X[feature].min(), test_X[feature].max()) for feature in features]
best_params = pso_optimize(bounds)
print("PSO优化的最佳参数:", best_params)

# 使用优化后的参数进行预测
y_pred_test = best_model.predict(pd.DataFrame({
    '温度，oC': [best_params[0]],
    '频率，Hz': [best_params[1]],
    '最大值': [best_params[2]],
    '励磁波形': [best_params[3]],
    '材料类型': [best_params[4]],
    '标准差': [best_params[5]],
    '峰峰值': [best_params[6]],
    '最大频率': [best_params[7]],
    '偏度': [best_params[8]]
}))

# 计算误差
mse_test = mean_squared_error(y_test, y_pred_test)
rmse_test = root_mean_squared_error(y_test, y_pred_test)
max_test = max_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
mape_test = mean_absolute_percentage_error(y_test, y_pred_test)

print(f"修正测试集 MSE: {mse_test:.4f}")
print(f"修正测试集 RMSE: {rmse_test:.4f}")
print(f"修正测试集 MAX: {max_test:.4f}")
print(f"修正测试集 R2: {r2_test:.4f}")
print(f"修正测试集 MAE: {mae_test:.4f}")
print(f"修正测试集 MAPE: {mape_test:.4f}")
