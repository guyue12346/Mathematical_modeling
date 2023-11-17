'''
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# 定义微分方程组
def model(y, t):
    y1, y2 = y
    dy1dt = -2*y1 + y2
    dy2dt = -y1 - 2*y2
    return [dy1dt, dy2dt]

# 初始条件
y0 = [1, 0]

# 时间范围
t = np.linspace(0, 5, 100)

# 求解微分方程组
solution = odeint(model, y0, t)

# 绘制结果
plt.plot(t, solution[:, 0], label='y1(t)')
plt.plot(t, solution[:, 1], label='y2(t)')
plt.xlabel('Time')
plt.ylabel('y(t)')
plt.legend()
plt.show()
'''

'''
import numpy as np
from scipy.integrate import odeint, quad
import matplotlib.pyplot as plt

# 定义微分方程组
def model(y, t):
    y1, y2 = y
    dy1dt = -2 * y1 + y2
    dy2dt = -y1 - 2 * y2
    return [dy1dt, dy2dt]

# 初始条件
y0 = [1, 0]

# 时间范围
t = np.linspace(0, 5, 100)

# 求解微分方程组
solution = odeint(model, y0, t)

# 定义要积分的解
def integrate_solution(t):
    # 取得解在时间 t 的值
    y_values = odeint(model, y0, [0, t])[1]
    # 对每个分量进行积分
    integral_y1, _ = quad(lambda x: y_values[0], 0, t)
    integral_y2, _ = quad(lambda x: y_values[1], 0, t)
    return integral_y1, integral_y2

# 对时间范围进行积分
integral_results = np.array([integrate_solution(ti) for ti in t])

# 绘制积分结果
plt.plot(t, integral_results[:, 0], label='Integral of y1(t)')
plt.plot(t, integral_results[:, 1], label='Integral of y2(t)')
plt.xlabel('Time')
plt.ylabel('Integral')
plt.legend()
plt.show()
'''



'''
from sympy import symbols, Function

x, y, z = symbols('x y z')
v1 = Function('v1')(x, y, z)
v2 = Function('v2')(x, y, z)
v3 = Function('v3')(x, y, z)
p = Function('p')(x, y, z)

partial_derivative_v1_x = v1.diff(x)

partial_derivative_v2_y = v2.diff(y)

partial_derivative_v3_z = v3.diff(z)

partial_derivative_v1_x+partial_derivative_v2_y+partial_derivative_v3_z == 0

p+1.225*(v1*v1+v2*v2+v3*v3)/2 == 1.225/2
'''

'''
from sympy import symbols, Function, Eq, solve

# 定义符号变量和函数
x, y, z = symbols('x y z')
v1 = Function('v1')(x, y, z)
v2 = Function('v2')(x, y, z)
v3 = Function('v3')(x, y, z)
p = Function('p')(x, y, z)

# 方程组
equations = [
    Eq(v1.diff(x) + v2.diff(y) + v3.diff(z), 0),
    Eq(p + 1.225 * (v1**2 + v2**2 + v3**2) / 2, 1.225 / 2)
]

# 求解方程组
solution = solve(equations, (v1, v2, v3, p))

# 打印解
print(solution)
'''


from sympy import symbols, Function, Eq, nsolve

# 定义符号变量和函数
x, y, z = symbols('x y z')
v1 = Function('v1')(x, y, z)
v2 = Function('v2')(x, y, z)
v3 = Function('v3')(x, y, z)
p = Function('p')(x, y, z)

# 方程组
equations = [
    Eq(v1.diff(x) + v2.diff(y) + v3.diff(z), 0),
    Eq(p + 1.225 * (v1**2 + v2**2 + v3**2) / 2, 1.225 / 2)
]

# 使用 nsolve 进行数值求解
initial_guess = {v1: 0, v2: 0, v3: 0, p: 0}  # 初始猜测
solution = nsolve(equations, [v1, v2, v3, p], initial_guess)

# 打印结果
print(solution)
