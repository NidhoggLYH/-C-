import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib

seed = 42
random.seed(seed)
np.random.seed(seed)

def steinmetz_equation(temperature, frequency, B_max, W, Q):
    P = 3.0134 * (frequency ** 1.4112) * (B_max ** 2.1536) * (temperature ** -0.2458) * (1 - 0.2458 * W) * (1 - 0.2458 * Q)
    return P


data = pd.read_excel('data/四种材料整合_提取特征_材料波形编码.xlsx')

temperatures = [25, 50, 70, 90]
materials = [1, 2, 3, 4]
waveforms = [1, 2, 3]
freq_min, freq_max = 49990, 501180
B_max_min, B_max_max = 0.00963815, 0.313284469

population_size = 100
generations = 50
mutation_rate = 0.1


def objective_function(temp_idx, mat_idx, wave_idx, freq, B_max):
    temperature = temperatures[temp_idx]
    if temperature <= 0:
        return float('inf')

    core_loss = steinmetz_equation(temperature, freq, B_max, waveforms[wave_idx], materials[mat_idx])
    if core_loss < 0:
        return float('inf')

    transmitted_energy = freq * B_max
    epsilon = 1e-6
    objective = core_loss / (transmitted_energy + epsilon)
    return objective


def random_solution():
    return [
        random.randint(0, len(temperatures) - 1),
        random.randint(0, len(materials) - 1),
        random.randint(0, len(waveforms) - 1),
        random.uniform(freq_min, freq_max),
        random.uniform(B_max_min, B_max_max)
    ]


def mutate(solution):
    index = random.randint(0, 4)
    if index == 0:
        solution[0] = random.randint(0, len(temperatures) - 1)
    elif index == 1:
        solution[1] = random.randint(0, len(materials) - 1)
    elif index == 2:
        solution[2] = random.randint(0, len(waveforms) - 1)
    elif index == 3:
        solution[3] = random.uniform(freq_min, freq_max)
    elif index == 4:
        solution[4] = random.uniform(B_max_min, B_max_max)
    return solution


def crossover(parent1, parent2):
    child = parent1[:3]  # 继承温度、材料、波形
    child.append(random.uniform(freq_min, freq_max))  # 随机生成频率
    child.append(random.uniform(B_max_min, B_max_max))  # 随机生成磁通密度
    return child


def genetic_algorithm():
    population = [random_solution() for _ in range(population_size)]
    best_solution = None
    best_value = float('inf')

    for generation in range(generations):
        population.sort(key=lambda sol: objective_function(*sol))
        new_population = population[:population_size // 2]  # 保留前50%

        while len(new_population) < population_size:
            parent1, parent2 = random.sample(new_population[:20], 2)  # 从前20个中随机选择父母
            child = crossover(parent1, parent2)
            if random.random() < mutation_rate:
                child = mutate(child)
            new_population.append(child)

        population = new_population
        current_best = population[0]
        current_value = objective_function(*current_best)

        if current_value < best_value:
            best_solution = current_best
            best_value = current_value

    return best_solution, best_value


# 运行遗传算法
best_solution, best_value = genetic_algorithm()

# 设置字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 输出最优解
optimal_conditions = {
    '温度，oC': temperatures[int(best_solution[0])],
    '磁芯材料': materials[int(best_solution[1])],
    '励磁波形': waveforms[int(best_solution[2])],
    '频率，Hz': best_solution[3],
    '磁通密度峰值': best_solution[4],
    '磁芯损耗': steinmetz_equation(
        temperatures[int(best_solution[0])],
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
