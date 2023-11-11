import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

# 生成一个示例时间序列的残差序列，你应该替换为你的实际数据
np.random.seed(0)
time_series = np.random.randn(166)  # 这里使用随机生成的残差序列作为示例

# 进行DW分析
dw_test = sm.stats.stattools.durbin_watson(time_series)

# 输出DW统计量的值
print("DW统计量:", dw_test)

# 绘制时间序列残差图
plt.figure(figsize=(8, 4))
plt.plot(time_series, label='Residuals')
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.title('Time Series Residuals')
plt.legend()
plt.grid(True)

# 绘制DW统计量图
plt.figure(figsize=(8, 4))
plt.plot(dw_test, marker='o', linestyle='-', color='b')
plt.axhline(y=2, color='r', linestyle='--', label='DW Critical Values')
plt.xlabel('Observations')
plt.ylabel('DW Statistic')
plt.title('Durbin-Watson Statistic')
plt.legend()
plt.grid(True)

# 显示图形
plt.show()
