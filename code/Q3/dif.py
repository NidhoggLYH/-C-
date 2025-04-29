import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools

# 设置字体，以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
data = pd.read_excel('merged_materials.xlsx')

# 对因变量进行对数变换
data['log_loss'] = np.log1p(data['磁芯损耗'])

# 自变量的唯一值
temperatures = data['温度'].unique()
waveforms = data['励磁波形'].unique()
materials = data['材料'].unique()

# 适应度函数
def evaluate(individual):
    temp = temperatures[individual[0]]
    waveform = waveforms[individual[1]]
    material = materials[individual[2]]

    subset = data[(data['温度'] == temp) &
                  (data['励磁波形'] == waveform) &
                  (data['材料'] == material)]

    if subset.empty:
        return (np.inf,)

    return (subset['log_loss'].mean(),)

# 遗传算法
def genetic_algorithm():
    if "FitnessMin" not in creator.__dict__:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     lambda: [np.random.randint(0, len(temperatures)),
                               np.random.randint(0, len(waveforms)),
                               np.random.randint(0, len(materials))])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxUniform, indpb=0.7)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.5)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=200)
    NGEN = 1000  # 将遗传算法的代数改为1000
    best_fit_history = []

    for gen in range(NGEN):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.rand() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if np.random.rand() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring

        fits = [ind.fitness.values[0] for ind in population]
        best_fit_history.append((np.min(fits), np.mean(fits), np.max(fits)))

    best_ind = population[np.argmin([ind.fitness.values[0] for ind in population])]
    best_fit = np.min([ind.fitness.values[0] for ind in population])

    return best_ind, best_fit, best_fit_history

# 模拟退火
def simulated_annealing():
    best_solution = None
    best_loss = np.inf
    fitness_history = []

    current_solution = [
        np.random.randint(0, len(temperatures)),
        np.random.randint(0, len(waveforms)),
        np.random.randint(0, len(materials))
    ]
    current_loss = evaluate(current_solution)[0]

    for _ in range(1000):  # 迭代次数
        new_solution = current_solution.copy()
        for i in range(3):
            new_solution[i] = np.random.randint(0, [len(temperatures), len(waveforms), len(materials)][i])

        new_loss = evaluate(new_solution)[0]

        if new_loss < current_loss:
            current_solution = new_solution
            current_loss = new_loss
        else:
            acceptance_probability = np.exp((current_loss - new_loss) / 1)  # 温度设为1
            if np.random.rand() < acceptance_probability:
                current_solution = new_solution
                current_loss = new_loss

        if current_loss < best_loss:
            best_solution = current_solution
            best_loss = current_loss

        fitness_history.append((current_loss, current_loss, current_loss))  # 用当前损失替代

    return best_solution, best_loss, fitness_history

# 联合优化
def combined_optimization():
    ga_best, ga_loss, ga_history = genetic_algorithm()
    sa_best, sa_loss, sa_history = simulated_annealing()

    # 综合最优
    combined_best = ga_best.copy()
    combined_best_loss = ga_loss if ga_loss < sa_loss else sa_loss

    # 记录联合优化历史
    combined_history = [(ga_loss + sa_loss) / 2] * max(len(ga_history), len(sa_history))

    return {
        'GA': (ga_best, ga_loss, ga_history),
        'SA': (sa_best, sa_loss, sa_history),
        'Combined': (combined_best, combined_best_loss, combined_history)
    }

# 执行优化
results = combined_optimization()

# 输出结果
ga_best, ga_loss, ga_history = results['GA']
sa_best, sa_loss, sa_history = results['SA']
combined_best, combined_loss, combined_history = results['Combined']

print("遗传算法结果:", ga_best, "损失:", ga_loss)
print("模拟退火结果:", sa_best, "损失:", sa_loss)
print("联合优化结果:", combined_best, "损失:", combined_loss)

# 可视化优化过程
ga_history = np.array(ga_history)
sa_history = np.array(sa_history)
combined_history = np.array(combined_history)

plt.figure(figsize=(12, 6))
plt.plot(ga_history[:, 0], label='遗传算法最小损耗', color='blue')
plt.plot(ga_history[:, 1], label='遗传算法平均损耗', color='green')
plt.plot(ga_history[:, 2], label='遗传算法最大损耗', color='red')
plt.plot(sa_history[:, 0], label='模拟退火损耗', color='orange', alpha=0.5)

# 绘制联合优化的平均损耗
plt.plot(combined_history, label='联合优化平均损耗', color='purple', linestyle='--')

plt.title('优化损耗过程')
plt.xlabel('迭代次数')
plt.ylabel('损耗 (对数变换的磁芯损耗)')
plt.legend()
plt.grid()
plt.show()

# 输出最佳自变量选择
best_temperature = temperatures[combined_best[0]]
best_waveform = waveforms[combined_best[1]]
best_material = materials[combined_best[2]]

print("最佳自变量选择:")
print("温度:", best_temperature)
print("励磁波形:", best_waveform)
print("材料:", best_material)
