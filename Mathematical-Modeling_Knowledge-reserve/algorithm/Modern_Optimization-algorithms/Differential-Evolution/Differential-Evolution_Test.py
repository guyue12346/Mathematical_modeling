import random

# 目标函数，这里以一个简单的二维函数为例
def objective_function(x):
    return x[0]**2 + x[1]**4

# 差分进化算法
def differential_evolution(objective_function, bounds, population_size, max_generations, F, CR):
    population = [[random.uniform(bounds[i][0], bounds[i][1]) for i in range(len(bounds))] for _ in range(population_size)]

    for generation in range(max_generations):
        new_population = []

        for i, target in enumerate(population):
            a, b, c = random.sample(population, 3)
            j = random.randint(0, len(target) - 1)
            trial = [target[k] if random.random() < CR or k == j else a[k] + F * (b[k] - c[k]) for k in range(len(target))]
            if objective_function(trial) < objective_function(target):
                new_population.append(trial)
            else:
                new_population.append(target)

        population = new_population

    # 找到最优解
    best_solution = min(population, key=objective_function)
    best_fitness = objective_function(best_solution)
    return best_solution, best_fitness

# 设置参数和搜索空间范围
bounds = [(-5, 5), (-5, 5)]
population_size = 50
max_generations = 100
F = 0.5  # 差异权重因子
CR = 0.8  # 交叉概率

# 运行差分进化算法
best_solution, best_fitness = differential_evolution(objective_function, bounds, population_size, max_generations, F, CR)

print("最优解：", best_solution)
print("最优解的适应度：", best_fitness)
