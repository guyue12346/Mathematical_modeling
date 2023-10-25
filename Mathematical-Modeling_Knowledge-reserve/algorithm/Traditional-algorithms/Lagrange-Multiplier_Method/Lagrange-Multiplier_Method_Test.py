import sympy as sp

# 定义变量
x, y, lambda_ = sp.symbols('x y lambda')

# 定义目标函数
f = x**2 + 2*y**2

# 定义约束条件
g = x + y - 1

# 构建拉格朗日函数
L = f + lambda_ * g

# 计算拉格朗日函数的偏导数
grad_L = [sp.diff(L, var) for var in (x, y, lambda_)]

# 解拉格朗日方程
solutions = sp.solve(grad_L, (x, y, lambda_))

# 提取最优解
optimal_x = solutions[x]
optimal_y = solutions[y]
optimal_lambda = solutions[lambda_]

# 打印最优解
print("最优解 x:", optimal_x)
print("最优解 y:", optimal_y)
print("拉格朗日乘子 lambda:", optimal_lambda)

# 验证约束条件
constraint_value = g.subs({x: optimal_x, y: optimal_y})
print("约束条件值 (g(x, y)): ", constraint_value)
