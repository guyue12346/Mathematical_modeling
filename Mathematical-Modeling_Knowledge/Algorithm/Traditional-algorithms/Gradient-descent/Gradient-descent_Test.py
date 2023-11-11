import numpy as np

# 生成随机数据
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.rand(100, 1)

# 添加一列偏置项到特征矩阵 X
X_b = np.c_[np.ones((100, 1)), X]

# 初始化模型参数
theta = np.random.rand(2, 1)  # 随机初始化参数

# 学习率和迭代次数
learning_rate = 0.1
n_iterations = 1000

# 梯度下降迭代
for iteration in range(n_iterations):
    # 计算预测值
    y_pred = X_b.dot(theta)
    
    # 计算梯度
    gradient = X_b.T.dot(y_pred - y) / len(y)
    
    # 更新参数
    theta = theta - learning_rate * gradient

# 打印最终参数值
print("tained(theta):")
print(theta)
