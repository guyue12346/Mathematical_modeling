import random

# 目标函数，这里以一个简单的二维函数为例
def objective_function(x):
    return x[0]**2 + x[1]**6

# 粒子类
class Particle:
    def __init__(self, bounds):
        self.position = [random.uniform(bounds[i][0], bounds[i][1]) for i in range(len(bounds))]
        self.velocity = [random.uniform(-1, 1) for _ in range(len(bounds))]
        self.best_position = self.position.copy()
        self.best_fitness = objective_function(self.position)

# 粒子群类
class ParticleSwarm:
    def __init__(self, bounds, num_particles, max_generations, w, c1, c2):
        self.particles = [Particle(bounds) for _ in range(num_particles)]
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.max_generations = max_generations
        self.w = w  # 惯性权重
        self.c1 = c1  # 个体认知因子
        self.c2 = c2  # 群体社会因子

    def optimize(self):
        for generation in range(self.max_generations):
            for particle in self.particles:
                fitness = objective_function(particle.position)
                if fitness < particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = particle.position.copy()

                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle.position.copy()

            for particle in self.particles:
                for i in range(len(particle.position)):
                    r1, r2 = random.random(), random.random()
                    cognitive = self.c1 * r1 * (particle.best_position[i] - particle.position[i])
                    social = self.c2 * r2 * (self.global_best_position[i] - particle.position[i])
                    particle.velocity[i] = self.w * particle.velocity[i] + cognitive + social
                    particle.position[i] += particle.velocity[i]

# 设置参数和搜索空间范围
bounds = [(-5, 5), (-5, 5)]
num_particles = 20
max_generations = 100
w = 0.5
c1 = 1.5
c2 = 1.5

# 运行粒子群算法
pso = ParticleSwarm(bounds, num_particles, max_generations, w, c1, c2)
pso.optimize()

print("最优解：", pso.global_best_position)
print("最优解的适应度：", pso.global_best_fitness)
