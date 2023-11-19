from scipy.optimize import fsolve
import numpy as np
from scipy.integrate import odeint


def equation(x):
    return 30.051*x**3 - 295.35*x**2 + 1581.7*x - 1128.6 - 25000

# 初值设定
initial_guess = [0.0]

# 使用fsolve求解方程
solution = fsolve(equation, initial_guess)

#参数设定
K_r = 0.1
rho = 1.23
A = 10.0
V_tip = 5.0
C_p = 0.05
C_h = 0.02
q = 100.0
S = 8.0
y = 2.0
I_yy = 100.0
initial_conditions = [0.0]  

def pitch_dynamics(theta, t):
    M_r = K_r * rho * A * V_tip
    T_p = C_p
    M_h = C_h * q * S * y

    # 俯仰角变化微分方程
    dtheta_dt = (M_r + T_p + M_h) / I_yy

    return dtheta_dt

time_points = np.linspace(0, 20, 1000)

#使用odeint求解方程
result = odeint(pitch_dynamics, initial_conditions, time_points)

pitch_angles = result[:, 0]