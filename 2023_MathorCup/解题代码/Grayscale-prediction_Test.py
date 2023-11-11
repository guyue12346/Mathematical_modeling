import pandas as pd
import numpy as np
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt

# 创建示例销售额时间序列数据
np.random.seed(42)
sales_data = pd.Series(100 + np.cumsum(np.random.randn(40)))

# 拆分数据集为训练集和测试集
train_size = int(len(sales_data) * 0.8)
train_data, test_data = sales_data[:train_size], sales_data[train_size:]

# 使用auto_arima选择ARIMA模型的参数
model = auto_arima(train_data, seasonal=False, stepwise=True, suppress_warnings=True)

# 获取选择的ARIMA模型的p、d、q值
p, d, q = model.order

# 拟合ARIMA模型
arima_model = ARIMA(train_data, order=(p, d, q))
arima_fit = arima_model.fit(disp=0)

# 预测未来数据点
n_forecast = len(test_data)
forecast, stderr, conf_int = arima_fit.forecast(steps=n_forecast, alpha=0.05)

# 可视化预测结果
plt.figure(figsize=(12, 6))
plt.plot(sales_data, label='销售额数据')
plt.plot(test_data.index, forecast, label='预测数据', color='red')
plt.fill_between(test_data.index, conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.3, label='95% 置信区间')
plt.legend()
plt.title('销售额灰度预测')
plt.show()



