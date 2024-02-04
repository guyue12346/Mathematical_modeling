import matplotlib.pyplot as plt
import numpy as np

# 定义函数Q(t)
def Q(t):

    return t**2 - 5*t + 6  

# 生成时间范围
t_values = np.linspace(0, 5, 100)  # 适当调整时间范围和点的数量

# 计算函数值
Q_values = Q(t_values)

# 创建图表
plt.figure(figsize=(8, 6))

# 根据势能正负选择线的颜色
colors = ['blue' if q >= 0 else 'red' for q in Q_values]

# 绘制函数图像
for i in range(len(t_values) - 1):
    plt.plot(t_values[i:i+2], Q_values[i:i+2], color=colors[i])

plt.title('Function Q(t) Graph')
plt.xlabel('time')
plt.ylabel('momentum')

plt.grid(True)
plt.show()


