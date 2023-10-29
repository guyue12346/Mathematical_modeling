import numpy as np
import statsmodels.api as sm

residuals=np.array([0.1, -0.2, 0.05, -0.1, 0.3, -0.15, 0.2, -0.05])

dw_statistic, dw_p_value = sm.stats.durbin_watson(residuals)

#dw_statistic 评估时间序列残差的一阶自相关性 2(小于1.5的值可能表示正自相关性，而大于2.5的值可能表示负自相关性)
#dw_p_value 验证Durbin-Watson统计量的显著性 (如果 dw_p_value 很小（通常小于显著性水平，如0.05），则可以拒绝零假设，表示存在一阶自相关性 如果 dw_p_value 较大，不能拒绝零假设，表示没有一阶自相关性)

