import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib
import random

# 设置随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)

# 定义斯坦麦茨方程
def steinmetz_equation_ori(temperature, frequency, B_max, k, alpha, beta):
    P = k * (frequency ** alpha) * (B_max ** beta)
    return P
def steinmetz_equation(temperature, frequency, B_max, k, alpha, beta, omiga, sigma, gamma, lamda):
    P = k * (frequency ** alpha) * (B_max ** beta) * (((temperature**3) * omiga) + ((temperature**2) * sigma) + (temperature * gamma) + lamda)
    return P


# 定义RMSE损失函数
def rmse_loss_ori(params, X, Y):
    k, alpha, beta = params
    temperature = X['温度，oC']
    frequency = X['频率，Hz']
    B_max = X['磁通密度峰值']
    target = Y
    pred = steinmetz_equation_ori(temperature, frequency, B_max, k, alpha, beta)
    return np.sqrt(np.mean((target - pred) ** 2))

def rmse_loss(params, X, Y):
    k, alpha, beta, omiga, sigma, gamma, lamda = params
    temperature = X['温度，oC']
    frequency = X['频率，Hz']
    B_max = X['磁通密度峰值']
    target = Y
    pred = steinmetz_equation(temperature, frequency, B_max, k, alpha, beta, omiga, sigma, gamma, lamda)
    rmse = np.sqrt(np.mean((target - pred) ** 2))
    return rmse

# 粒子群优化算法
class Particle:
    def __init__(self, bounds):
        self.position = np.random.uniform(bounds[0][0], bounds[0][1], size=3)  # 初始位置
        self.velocity = np.random.uniform(-1, 1, size=3)  # 初始速度
        self.best_position = self.position.copy()  # 个体最好位置
        self.best_value = float('inf')  # 个体最好值

def pso(X, Y, num_particles=80, max_iter=500, bounds = ((0.0, 10.0), (1.0, 3.0), (2.0, 3.0))):
    particles = [Particle(bounds) for _ in range(num_particles)]
    global_best_position = None
    global_best_value = float('inf')

    for _ in range(max_iter):
        for particle in particles:
            value = rmse_loss_ori(particle.position, X, Y)
            if value < particle.best_value:
                particle.best_value = value
                particle.best_position = particle.position

            if value < global_best_value:
                global_best_value = value
                global_best_position = particle.position

        for particle in particles:
            inertia = 0.5 * particle.velocity
            cognitive = np.random.rand(3) * (particle.best_position - particle.position)
            social = np.random.rand(3) * (global_best_position - particle.position)
            particle.velocity = inertia + cognitive + social
            particle.position += particle.velocity

            # 确保位置在范围内
            particle.position = np.clip(particle.position, *zip(*bounds))

    return global_best_position


class Particle1:
    def __init__(self, bounds):
        self.position = np.random.uniform(bounds[0][0], bounds[0][1], size=len(bounds))  # 初始位置
        self.velocity = np.random.uniform(-1, 1, size=len(bounds))  # 初始速度
        self.best_position = self.position.copy()  # 个体最好位置
        self.best_value = float('inf')  # 个体最好值


def pso1(X, Y, num_particles=100, max_iter=100,
        bounds=((0.0, 10.0), (1.0, 3.0), (2.0, 3.0), (-1.0, 1.0), (-20.0, 20.0), (-20.0, 20.0), (-20.0, 20.0)),  # 四个参数的范围
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


# 创建数据框
df = pd.read_excel('data/材料1_正弦波_带磁通密度峰值.xlsx')
X = df[['温度，oC', '频率，Hz', '磁通密度峰值']]
Y = df['磁芯损耗，w/m3']

# 数据集划分
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=14)



# 运行粒子群优化 修正
bounds = ((0.001, 10.0), (1.0, 3.0), (2.0, 3.0), (-1.0, 1.0), (-20.0, 20.0), (-20.0, 20.0), (-20.0, 20.0))
best_params = pso1(x_train, y_train, bounds=bounds)
print("PSO 修正 最佳拟合参数:")
print(f"k: {best_params[0]:.4f}, alpha: {best_params[1]:.4f}, beta: {best_params[2]:.4f}, omiga: {best_params[3]:.4f}, sigma: {best_params[4]:.4f}, gamma: {best_params[5]:.4f}, lamda: {best_params[6]:.4f}")

# 预测
k_best, alpha_best, beta_best, omiga_best, sigma_best, gamma_best, lamda_best = best_params
y_pred_train = steinmetz_equation(x_train['温度，oC'], x_train['频率，Hz'], x_train['磁通密度峰值'], k_best,
                                          alpha_best, beta_best, omiga_best, sigma_best,gamma_best, lamda_best)
y_pred_test = steinmetz_equation(x_test['温度，oC'], x_test['频率，Hz'], x_test['磁通密度峰值'], k_best,
                                         alpha_best, beta_best, omiga_best, sigma_best,gamma_best, lamda_best)

# 计算误差
mse_train = mean_squared_error(y_train, y_pred_train)
rmse_train = np.sqrt(mse_train)
mse_test = mean_squared_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)

# 输出误差
print(f"修正 训练集 MSE: {mse_train:.4f}")
print(f"修正 训练集 RMSE: {rmse_train:.4f}")
print(f"修正 测试集 MSE: {mse_test:.4f}")
print(f"修正 测试集 RMSE: {rmse_test:.4f}")




# 运行粒子群优化
bounds = ((0.001, 10.0), (1.0, 3.0), (2.0, 3.0))  # 对 k, alpha, beta 的范围设置
best_params = pso(x_train, y_train, bounds=bounds)
print("PSO 原始 最佳拟合参数:")
print(f"k: {best_params[0]:.4f}, alpha: {best_params[1]:.4f}, beta: {best_params[2]:.4f}")

# 预测
k_best, alpha_best, beta_best = best_params
y_pred_train_ori = steinmetz_equation_ori(x_train['温度，oC'], x_train['频率，Hz'], x_train['磁通密度峰值'], k_best,
                                          alpha_best, beta_best)
y_pred_test_ori = steinmetz_equation_ori(x_test['温度，oC'], x_test['频率，Hz'], x_test['磁通密度峰值'], k_best,
                                         alpha_best, beta_best)

# 计算误差
mse_train = mean_squared_error(y_train, y_pred_train_ori)
rmse_train = np.sqrt(mse_train)
mse_test = mean_squared_error(y_test, y_pred_test_ori)
rmse_test = np.sqrt(mse_test)

# 输出误差
print(f"原始 训练集 MSE: {mse_train:.4f}")
print(f"原始 训练集 RMSE: {rmse_train:.4f}")
print(f"原始 测试集 MSE: {mse_test:.4f}")
print(f"原始 测试集 RMSE: {rmse_test:.4f}")







# 设置字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 负号正常显示

# 可视化结果
plt.figure(figsize=(10,6))
plt.scatter(range(len(y_pred_test_ori)), y_pred_test_ori, label='原始预测磁芯损耗', color='y')
plt.scatter(range(len(y_test)), y_test, label='实际磁芯损耗', color='b')
plt.scatter(range(len(y_pred_test)), y_pred_test, label='修正预测磁芯损耗', color='r')
plt.title('测试集表现')
plt.xlabel('采样点')
plt.ylabel('磁芯损耗')
plt.legend()
plt.grid(True)
plt.show()


# 残差
residuals = y_test - y_pred_test
residuals_ori = y_test - y_pred_test_ori
plt.figure(figsize=(10,6))
plt.scatter(range(len(residuals)), residuals, label='修正残差', color='green')
plt.scatter(range(len(residuals_ori)), residuals_ori, label='原始残差', color='brown')
plt.title('预测残差')
plt.xlabel('采样点')
plt.ylabel('磁芯损耗')
plt.legend()
plt.grid(True)
plt.show()
