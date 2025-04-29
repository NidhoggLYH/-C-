import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_error, max_error, r2_score, mean_absolute_error,mean_absolute_percentage_error
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib
import random

# 读取数据
df1 = pd.read_excel('data/问题五训练集_按材料随机采样.xlsx')
df2 = pd.read_excel('data/问题五验证集_按材料随机采样.xlsx')

# 准备拟合数据
x_train = df1[['温度，oC', '频率，Hz', '励磁波形', '材料类型', '标准差', '最大值', '峰峰值', '最大频率', '偏度']]
y_train = df1['磁芯损耗，w/m3']

x_test = df2[['温度，oC', '频率，Hz', '励磁波形', '材料类型', '标准差', '最大值', '峰峰值', '最大频率', '偏度']]
y_test = df2['磁芯损耗，w/m3']


def steinmetz_equation(temprature, frequency, B_max):
    P = 4.9259 * (frequency ** 1.4585) * (B_max ** 2.4432) * (temprature ** -0.3986)
    return P

# 定义最小平方根误差损失函数
def objjective(params, X, Y):
    temperature, frequency, B_max = params
    temperature = X['温度，oC']
    frequency = X['频率，Hz']
    B_max = X['最大值']
    target = Y
    pred = steinmetz_equation(temperature, frequency, B_max)
    rmse = np.sqrt(np.mean((target - pred) ** 2))
    return rmse

def energy(frequency, B_max):
    return frequency * B_max

def objective(params, X, l1, l2):
    #k, alpha, beta, gamma, h = params
    temperature = X['温度，oC']
    frequency = X['频率，Hz']
    B_max = X['最大值']
    W = X['励磁波形']
    Q = X['材料类型']
    #target = Y
    pred = steinmetz_equation(temperature, frequency, B_max, W, Q, params)
    energy_m = energy(frequency, B_max)
    final_objective = l1 * pred - l2 * energy_m
    return final_objective

class Particle1:
    def __init__(self, bounds):
        self.position = np.random.uniform(bounds[0][0], bounds[0][1], size=len(bounds))  # 初始位置
        self.velocity = np.random.uniform(-1, 1, size=len(bounds))  # 初始速度
        self.best_position = self.position.copy()  # 个体最好位置
        self.best_value = float('inf')  # 个体最好值

def pso1(X, Y, num_particles=80, max_iter=300,
        bounds=((0.0, 6.0), (1.0, 3.0), (2.0, 3.0), (-1.0, 1.0), (-10.0, 10.0), (-10.0, 10.0)),  # 四个参数的范围
        w=0.5, c1=1.5, c2=1.5):  # w: 惯性权重, c1: 认知因子, c2: 社会因子
    particles = [Particle1(bounds) for _ in range(num_particles)]
    global_best_position = None
    global_best_value = float('inf')

    for _ in range(max_iter):
        for particle in particles:
            value = rmse_loss(particle.position, X, Y)  # 适应度函数，使用四个参数
            if value < particle.best_value:
                particle.best_value = value
                particle.best_position = particle.position.copy()

            if value < global_best_value:
                global_best_value = value
                global_best_position = particle.position.copy()

        for particle in particles:
            inertia = w * particle.velocity
            cognitive = c1 * np.random.rand(len(bounds)) * (particle.best_position - particle.position)
            social = c2 * np.random.rand(len(bounds)) * (global_best_position - particle.position)
            particle.velocity = inertia + cognitive + social
            particle.position += particle.velocity

            # 确保位置在范围内
            particle.position = np.clip(particle.position, *zip(*bounds))

    return global_best_position

# 运行粒子群优化 修正
bounds = ((0.001, 10.0), (1.0, 3.0), (2.0, 3.0), (-1.0, 1.0), (-10.0, 10.0), (-10.0, 10.0),(-10.0, 10.0),(-10.0, 10.0))  # 对 k, alpha, beta, gamma, h, p, l1, l2的范围设置
best_params = pso1(x_train, y_train, bounds=bounds)
print("PSO 修正 最佳拟合参数:")
print(f"k: {best_params[0]:.4f}, alpha: {best_params[1]:.4f}, beta: {best_params[2]:.4f}， gamma: {best_params[3]:.4f}, h: {best_params[3]:.4f}, p: {best_params[3]:.4f}")

# 预测
k_best, alpha_best, beta_best, gamma_best, h_best, p_best = best_params
y_pred_train = steinmetz_equation(x_train['温度，oC'], x_train['频率，Hz'], x_train['最大值'], x_train['励磁波形'], x_train['材料类型'], k_best,
                                          alpha_best, beta_best, gamma_best, h_best, p_best)
y_pred_test = steinmetz_equation(x_test['温度，oC'], x_test['频率，Hz'], x_test['最大值'], x_test['励磁波形'], x_test['材料类型'], k_best,
                                         alpha_best, beta_best, gamma_best, h_best, p_best)

# 计算误差
mse_train = mean_squared_error(y_train, y_pred_train)
rmse_train = root_mean_squared_error(y_train, y_pred_train)
max_train = max_error(y_train, y_pred_train)

mse_test = mean_squared_error(y_test, y_pred_test)
rmse_test = root_mean_squared_error(y_test, y_pred_test)
max_test = max_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
mape_test = mean_absolute_percentage_error(y_test, y_pred_test)
# 输出误差
print(f"修正训练集 MSE: {mse_train:.4f}")
print(f"修正训练集 RMSE: {rmse_train:.4f}")
print(f"修正训练集 MAX: {max_train:.4f}")

print(f"修正测试集 MSE: {mse_test:.4f}")
print(f"修正测试集 RMSE: {rmse_test:.4f}")
print(f"修正测试集 MAX: {max_test:.4f}")
print(f"修正测试集 R2: {r2_test:.4f}")
print(f"修正测试集 MAE: {mae_test:.4f}")
print(f"修正测试集 MAPE: {mape_test:.4f}")