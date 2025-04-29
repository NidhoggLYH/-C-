import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib

# 设置随机种子
seed = 12
random.seed(seed)
np.random.seed(seed)

def steinmetz_equation(temperature, frequency, B_max, W, Q):
    P = 4.9259 * (frequency ** 1.4585) * (B_max ** 2.4432) * (temperature ** -0.3986) * (1 - 0.2896 * W) * (1 - 0.2893 * Q)
    return P

# 定义变量范围
temperature_min, temperature_max = 25, 90  # 温度范围改为25到90的连续值
materials = [1, 2, 3, 4]
waveforms = [1, 2, 3]
freq_min, freq_max = 49990, 501180
B_max_min, B_max_max = 0.00963815, 0.313284469

# 初始化模拟退火参数
T_init = 1000  # 初始温度
T_min = 1  # 最小温度
alpha = 0.9  # 温度衰减率
max_iter = 1000  # 每个温度的迭代次数

# 目标函数
def objective_function(temperature, mat_idx, wave_idx, freq, B_max):
    # 确保索引是整数
    mat_idx = int(round(mat_idx))
    wave_idx = int(round(wave_idx))

    # 检查温度是否为零或负数
    if temperature <= 0:
        return float('inf')  # 返回一个很大的值，表示不可行的解

    # 模拟磁芯损耗的计算
    core_loss = steinmetz_equation(temperature, freq, B_max, waveforms[wave_idx], materials[mat_idx])

    # 确保损耗非负
    if core_loss < 0:
        return float('inf')  # 返回不可行解

    transmitted_energy = freq * B_max  # 传输磁能是频率和磁通密度峰值的乘积

    # 定义目标函数
    epsilon = 1e-6
    objective = core_loss / (transmitted_energy + epsilon)
    return objective

# 随机生成初始解
def random_solution():
    temperature = random.uniform(temperature_min, temperature_max)
    mat_idx = random.randint(0, len(materials) - 1)
    wave_idx = random.randint(0, len(waveforms) - 1)
    freq = random.uniform(freq_min, freq_max)
    B_max = random.uniform(B_max_min, B_max_max)
    return [temperature, mat_idx, wave_idx, freq, B_max]

# 生成邻域解
def neighbor_solution(current_sol):
    new_sol = current_sol.copy()
    index = random.randint(0, 4)  # 随机选择要变动的变量
    if index == 0:
        new_sol[0] = random.uniform(temperature_min, temperature_max)
    elif index == 1:
        new_sol[1] = random.randint(0, len(materials) - 1)
    elif index == 2:
        new_sol[2] = random.randint(0, len(waveforms) - 1)
    elif index == 3:
        new_sol[3] = random.uniform(freq_min, freq_max)
    elif index == 4:
        new_sol[4] = random.uniform(B_max_min, B_max_max)
    return new_sol

# 模拟退火算法
def simulated_annealing():
    current_solution = random_solution()
    current_value = objective_function(*current_solution)
    best_solution = current_solution
    best_value = current_value
    T = T_init

    values = []  # 用于记录每次迭代的目标函数值

    while T > T_min:
        for _ in range(max_iter):
            new_solution = neighbor_solution(current_solution)
            new_value = objective_function(*new_solution)

            # 如果新解更好，直接接受
            if new_value < current_value:
                current_solution = new_solution
                current_value = new_value
                if new_value < best_value:
                    best_solution = new_solution
                    best_value = new_value
            # 如果新解更差，以一定概率接受
            else:
                delta = new_value - current_value
                acceptance_prob = math.exp(-delta / T)
                if random.random() < acceptance_prob:
                    current_solution = new_solution
                    current_value = new_value

        # 降低温度
        T *= alpha
        values.append(best_value)

    return best_solution, best_value, values

# 运行模拟退火算法
best_solution, best_value, values = simulated_annealing()

# 设置字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 负号正常显示

# 输出最优解
optimal_conditions = {
    '温度，oC': best_solution[0],
    '磁芯材料': materials[int(best_solution[1])],
    '励磁波形': waveforms[int(best_solution[2])],
    '频率，Hz': best_solution[3],
    '磁通密度峰值': best_solution[4],
    '磁芯损耗': steinmetz_equation(
        best_solution[0],
        best_solution[3],
        best_solution[4],
        waveforms[int(best_solution[2])],
        materials[int(best_solution[1])]
    ),
    '最大传输磁能': best_solution[3] * best_solution[4]
}

print("最优条件：")
for key, value in optimal_conditions.items():
    print(f"{key}: {value}")

print(f"最优目标函数值: {best_value}")

# 绘制优化过程的收敛图
plt.figure(figsize=(10, 6))
plt.plot(values, marker='o')
plt.title('模拟退火算法优化过程')
plt.xlabel('迭代次数')
plt.ylabel('目标函数值')
plt.grid(True)
plt.show()
