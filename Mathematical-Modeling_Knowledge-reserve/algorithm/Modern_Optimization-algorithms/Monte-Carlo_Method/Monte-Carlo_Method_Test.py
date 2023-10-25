import random

# 模拟点的数量
num_points = 1000000

# 在正方形内随机生成点，并计算它们与圆心的距离
points_inside_circle = 0

for _ in range(num_points):
    x = random.uniform(0, 1)  # 在正方形内生成随机x坐标
    y = random.uniform(0, 1)  # 在正方形内生成随机y坐标
    distance = x**2 + y**2  # 计算点到原点的距离的平方

    if distance <= 1:  # 如果点在单位圆内
        points_inside_circle += 1

# 估算圆的面积
estimated_area = points_inside_circle / num_points

# 因为我们模拟的是一个四分之一的圆，所以需要乘以4来估算整个圆的面积
estimated_area *= 4

print("估计的圆的面积:", estimated_area)
