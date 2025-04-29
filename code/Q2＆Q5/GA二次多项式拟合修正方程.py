import numpy as np
import random
import pandas as pd
from deap import base, creator, tools, algorithms
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_error, max_error
import matplotlib.pyplot as plt
import matplotlib

# 设置随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)

# 定义斯坦麦茨方程函数
def steinmetz_equation(temprature, frequency, B_max, k, alpha, beta, gamma):
    """
    计算斯坦麦茨方程的磁芯损耗。

    参数:
    temprature: 温度（℃）
    frequency: 工作频率 (Hz)
    B_max: 最大磁通密度 (T)
    k: 材料常数
    alpha: 频率指数
    beta: 磁通密度指数
    gamma：温度指数

    返回:
    磁芯损耗 (W/m³)
    """
    P = k * (frequency ** alpha) * (B_max ** beta) * (temprature ** gamma)
    return P

def steinmetz_equation_ori(temprature, frequency, B_max, k, alpha, beta):
    P = k * (frequency ** alpha) * (B_max ** beta)
    return P

def steinmetz_equation_2(temperature, frequency, B_max, k, alpha, beta, sigma, gamma, lamda):
    P = k * (frequency ** alpha) * (B_max ** beta) * (((temperature**2) * sigma) + (temperature * gamma) + lamda)
    return P


# 定义最小平方根误差损失函数
def rmse_loss(params, X, Y):
    k, alpha, beta, sigma, gamma, lamda = params
    temperature = X['温度，oC']
    frequency = X['频率，Hz']
    B_max = X['磁通密度峰值']
    target = Y
    pred = steinmetz_equation_2(temperature, frequency, B_max, k, alpha, beta, sigma, gamma, lamda)
    rmse = np.sqrt(np.mean((target - pred) ** 2))
    return rmse

def rmse_loss_ori(params, X, Y):
    k, alpha, beta = params
    temperature = X['温度，oC']
    frequency = X['频率，Hz']
    B_max = X['磁通密度峰值']
    target = Y
    pred = steinmetz_equation_ori(temperature, frequency, B_max, k, alpha, beta)
    rmse = np.sqrt(np.mean((target - pred) ** 2))
    return rmse

def eval_individual(individual):
    # 确保参数在范围内
    #individual[0] = np.clip(individual[0], 0.001, 3.0)
    #individual[1] = np.clip(individual[1], 1.0, 3.0)
    #individual[2] = np.clip(individual[2], 2.0, 3.0)
    #individual[3] = np.clip(individual[3], -1.0, -0.1)
    k, alpha, beta, sigma, gamma, lamda = individual
    params = [k, alpha, beta, sigma, gamma, lamda]
    return (rmse_loss(params, x_train, y_train),)

def eval_individual_ori(individual):
    # 确保参数在范围内
    #individual[0] = np.clip(individual[0], 0.001, 3.0)
    #individual[1] = np.clip(individual[1], 1.0, 3.0)
    #individual[2] = np.clip(individual[2], 2.0, 3.0)
    k, alpha, beta = individual
    params = [k, alpha, beta]
    return (rmse_loss_ori(params, x_train, y_train),)

# 创建数据框
df = pd.read_excel('data/材料1_正弦波_带磁通密度峰值.xlsx')

# 准备拟合数据
X = df[['温度，oC','频率，Hz','磁通密度峰值']]
Y = df['磁芯损耗，w/m3']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=14)

#初始话遗传算法参数
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))#最小化误差
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float_k", np.random.uniform, 0.001, 6.0)  # k 的范围
toolbox.register("attr_float_alpha", np.random.uniform, 1.0, 3.0)   # alpha 的范围
toolbox.register("attr_float_beta", np.random.uniform, 2.0, 3.0)    # beta 的范围
toolbox.register("attr_float_gamma", np.random.uniform, -100.0, 100.0)   # gamma 的范围
toolbox.register("attr_float_lamda", np.random.uniform, -200.0, 200.0)   # lamda 的范围
toolbox.register("attr_float_sigma", np.random.uniform, -10.0, 10.0) # sigma 的范围



# 初始化个体
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_float_k, toolbox.attr_float_alpha, toolbox.attr_float_beta, toolbox.attr_float_sigma, toolbox.attr_float_gamma, toolbox.attr_float_lamda), n=1)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", eval_individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

#初始化种群、参数
population = toolbox.population(n=300)
num_generations = 500
crossover_prob = 0.6
mutation_prob = 0.3

result, logbook = algorithms.eaSimple(population, toolbox, cxpb=crossover_prob, mutpb=mutation_prob,
                                      ngen=num_generations, verbose=True, stats=None)

#获取最优解
best_individual = tools.selBest(result, 1)[0]
print("最佳拟合参数:")
print(f"k: {best_individual[0]:.4f}, alpha: {best_individual[1]:.4f}, beta: {best_individual[2]:.4f}, sigma: {best_individual[3]:.4f}, gamma: {best_individual[4]:.4f}, , lamda: {best_individual[5]:.4f}")

k_best, alpha_best, beta_best, sigma_best, gamma_best, lamda_best = best_individual
y_pred_train = steinmetz_equation_2(x_train['温度，oC'],x_train['频率，Hz'],x_train['磁通密度峰值'],k_best, alpha_best, beta_best, sigma_best, gamma_best, lamda_best)
y_pred_test = steinmetz_equation_2(x_test['温度，oC'],x_test['频率，Hz'],x_test['磁通密度峰值'],k_best, alpha_best, beta_best, sigma_best, gamma_best, lamda_best)

mse_train = mean_squared_error(y_train, y_pred_train)
rmse_train = root_mean_squared_error(y_train, y_pred_train)
max_train = max_error(y_train, y_pred_train)

mse_test = mean_squared_error(y_test, y_pred_test)
rmse_test = root_mean_squared_error(y_test, y_pred_test)
max_test = max_error(y_test, y_pred_test)
# 输出误差
print(f"修正训练集 MSE: {mse_train:.4f}")
print(f"修正训练集 RMSE: {rmse_train:.4f}")
print(f"修正训练集 MAX: {max_train:.4f}")

print(f"修正测试集 MSE: {mse_test:.4f}")
print(f"修正测试集 RMSE: {rmse_test:.4f}")
print(f"修正测试集 MAX: {max_test:.4f}")








# 检查并删除已有的类
if "FitnessMin" in creator.__dict__:
    del creator.FitnessMin

if "Individual" in creator.__dict__:
    del creator.Individual
#初始话遗传算法参数
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))#最小化误差
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float_k", np.random.uniform, 0.001, 3.0)  # k的范围
toolbox.register("attr_float_alpha", np.random.uniform, 1.0, 3.0)   # alpha的范围
toolbox.register("attr_float_beta", np.random.uniform, 2.0, 3.0)     # beta的范围

# 初始化个体，分别为k, alpha, beta
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_float_k, toolbox.attr_float_alpha, toolbox.attr_float_beta), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", eval_individual_ori)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

#初始化种群、参数
population = toolbox.population(n=300)
num_generations = 500
crossover_prob = 0.6
mutation_prob = 0.3

result_ori, logbook_ori = algorithms.eaSimple(population, toolbox, cxpb=crossover_prob, mutpb=mutation_prob,
                                      ngen=num_generations, verbose=True, stats=None)

#获取最优解
best_individual_ori = tools.selBest(result_ori, 1)[0]
print("最佳拟合参数:")
print(f"k: {best_individual_ori[0]:.4f}, alpha: {best_individual_ori[1]:.4f}, beta: {best_individual_ori[2]:.4f}")

k_best_ori, alpha_best_ori, beta_best_ori = best_individual_ori
y_pred_train_ori = steinmetz_equation_ori(x_train['温度，oC'],x_train['频率，Hz'],x_train['磁通密度峰值'],k_best_ori, alpha_best_ori, beta_best_ori)
y_pred_test_ori = steinmetz_equation_ori(x_test['温度，oC'],x_test['频率，Hz'],x_test['磁通密度峰值'],k_best_ori, alpha_best_ori, beta_best_ori)

mse_train = mean_squared_error(y_train, y_pred_train_ori)
rmse_train = root_mean_squared_error(y_train, y_pred_train_ori)
max_train = max_error(y_train, y_pred_train_ori)

mse_test = mean_squared_error(y_test, y_pred_test_ori)
rmse_test = root_mean_squared_error(y_test, y_pred_test_ori)
max_test = max_error(y_test, y_pred_test_ori)

# 输出误差
print(f"原始训练集 MSE: {mse_train:.4f}")
print(f"原始训练集 RMSE: {rmse_train:.4f}")
print(f"原始训练集 MAX: {max_train:.4f}")

print(f"原始测试集 MSE: {mse_test:.4f}")
print(f"原始测试集 RMSE: {rmse_test:.4f}")
print(f"原始测试集 MAX: {max_test:.4f}")









# 设置字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 负号正常显示

# 可视化结果
plt.figure(figsize=(10,6))
plt.scatter(range(len(y_pred_test_ori)), y_pred_test_ori, label='传统预测磁芯损耗', color='g')
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
plt.scatter(range(len(residuals)), residuals, label='修正方程', color='r')
plt.scatter(range(len(residuals)), residuals_ori, label='传统方程', color='g')
plt.title('预测残差')
plt.xlabel('采样点')
plt.ylabel('磁芯损耗')
plt.legend()
plt.grid(True)
plt.show()






