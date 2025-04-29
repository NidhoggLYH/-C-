import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# 设置随机种子
seed = 13
random.seed(seed)
np.random.seed(seed)

# 斯坦麦茨方程
def steinmetz_equation(temperature, frequency, B_max, W, Q):
    P = 4.9259 * (frequency ** 1.4535) * (B_max ** 2.1796) * (temperature ** -0.2893) * (1 - 0.2896 * W) * (1 - 0.2893 * Q)
    return P

# 读取数据
data = pd.read_excel('data/四种材料整合_提取特征_材料波形编码.xlsx')

# 定义变量范围
temperatures = np.linspace(25, 90, num=100)  # 从25到90，共100个温度值
materials = [1, 2, 3, 4]
waveforms = [1, 2, 3]
freq_min, freq_max = 49990, 501180
B_max_min, B_max_max = 0.00963815, 0.313284469

# 目标函数
def objective_function(temp_idx, mat_idx, wave_idx, freq, B_max):
    temperature = temperatures[temp_idx]
    W = waveforms[wave_idx]
    Q = materials[mat_idx]

    if temperature <= 0:
        return float('inf')

    core_loss = steinmetz_equation(temperature, freq, B_max, W, Q)

    if core_loss < 0:
        return float('inf')

    transmitted_energy = freq * B_max
    epsilon = 1e-6
    objective = core_loss / (transmitted_energy + epsilon)
    return objective

# 随机生成初始解
def random_solution():
    temp_idx = random.randint(0, len(temperatures) - 1)
    mat_idx = random.randint(0, len(materials) - 1)
    wave_idx = random.randint(0, len(waveforms) - 1)
    freq = random.uniform(freq_min, freq_max)
    B_max = random.uniform(B_max_min, B_max_max)
    return [temp_idx, mat_idx, wave_idx, freq, B_max]

# 粒子类
class Particle:
    def __init__(self):
        self.position = random_solution()
        self.velocity = [0.0] * len(self.position)
        self.best_position = self.position
        self.best_value = objective_function(*self.position)
        while self.best_value is None:
            self.position = random_solution()
            self.best_value = objective_function(*self.position)

# 粒子群优化算法
def particle_swarm_optimization(num_particles=30, max_iter=100):
    particles = [Particle() for _ in range(num_particles)]
    global_best_position = particles[0].best_position
    global_best_value = particles[0].best_value

    w = 0.5
    c1 = 1.5
    c2 = 1.5

    for _ in range(max_iter):
        for particle in particles:
            if particle.best_value is not None and particle.best_value < global_best_value:
                global_best_value = particle.best_value
                global_best_position = particle.best_position

            for i in range(len(particle.position)):
                r1 = random.random()
                r2 = random.random()
                particle.velocity[i] = (w * particle.velocity[i] +
                                        c1 * r1 * (particle.best_position[i] - particle.position[i]) +
                                        c2 * r2 * (global_best_position[i] - particle.position[i]))

                particle.position[i] += particle.velocity[i]

                if i == 0:  # 温度索引
                    particle.position[i] = max(0, min(len(temperatures) - 1, int(particle.position[i])))
                elif i == 1:  # 材料索引
                    particle.position[i] = max(0, min(len(materials) - 1, int(particle.position[i])))
                elif i == 2:  # 波形索引
                    particle.position[i] = max(0, min(len(waveforms) - 1, int(particle.position[i])))
                elif i == 3:  # 频率
                    particle.position[i] = max(freq_min, min(freq_max, particle.position[i]))
                elif i == 4:  # 磁通密度峰值
                    particle.position[i] = max(B_max_min, min(B_max_max, particle.position[i]))

            current_value = objective_function(*particle.position)

            if current_value is not None and current_value < particle.best_value:
                particle.best_value = current_value
                particle.best_position = particle.position

    return global_best_position, global_best_value

# 运行PSO算法
best_solution, best_value = particle_swarm_optimization()

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

# 可选：在这里添加收敛图的代码
