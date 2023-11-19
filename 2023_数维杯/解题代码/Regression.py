
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# 已知的数据点
known_x = np.array([0, 0.1, 0.2, 0.3])
known_y = np.array([-0.00002,0.00017,0.00034,0.00043])  # 用实际的值替换

# 创建插值函数
interp_function = interp1d(known_x, known_y, kind='cubic', fill_value='extrapolate')

# 需要插值的点
x_to_interpolate = 80/180

# 进行插值
y_interpolated = interp_function(x_to_interpolate)

print(f'在 x = {x_to_interpolate} 处的插值 y 值: {y_interpolated}')

# 绘制插值结果
x_values = np.linspace(0, 0.5, 100)
plt.plot(known_x, known_y, 'o', label='Known Data Points')
plt.plot(x_values, interp_function(x_values), '-', label='Interpolation Curves')
plt.xlabel('R_4')
plt.ylabel('R=F/R')
plt.legend()
plt.show()
