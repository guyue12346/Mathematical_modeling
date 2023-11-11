import random
import math

# 生成随机城市坐标
def generate_random_cities(num_cities):
    return [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(num_cities)]

# 计算路径长度
def calculate_path_length(path, cities):
    total_length = 0
    for i in range(len(path) - 1):
        city1 = cities[path[i]]
        city2 = cities[path[i + 1]]
        total_length += math.dist(city1, city2)
    return total_length

# 模拟退火算法
def simulated_annealing(cities, initial_path, initial_temperature, cooling_rate, num_iterations):
    current_path = initial_path
    current_length = calculate_path_length(current_path, cities)
    best_path = current_path
    best_length = current_length
    temperature = initial_temperature

    for _ in range(num_iterations):
        new_path = current_path.copy()

        # 随机选择两个不同的城市并交换它们的顺序
        i, j = random.sample(range(len(new_path)), 2)
        new_path[i], new_path[j] = new_path[j], new_path[i]

        new_length = calculate_path_length(new_path, cities)
        delta_length = new_length - current_length

        if delta_length < 0 or random.random() < math.exp(-delta_length / temperature):
            current_path = new_path
            current_length = new_length

            if current_length < best_length:
                best_path = current_path
                best_length = current_length

        temperature *= cooling_rate

    return best_path, best_length

if __name__ == "__main__":
    num_cities = 20
    cities = generate_random_cities(num_cities)
    initial_path = list(range(num_cities))
    random.shuffle(initial_path)
    initial_temperature = 1000.0
    cooling_rate = 0.995
    num_iterations = 10000

    best_path, best_length = simulated_annealing(cities, initial_path, initial_temperature, cooling_rate, num_iterations)
    print("Best Path:", best_path)
    print("Best Path Length:", best_length)
