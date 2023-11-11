import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from MMtools import get_data_dict
from MMtools import get_keys_by_value
from datetime import datetime, timedelta
from openpyxl import Workbook


time_series_dict=get_data_dict()

time_series_lists=list(time_series_dict.values())

output_folder = 'E:/plot_1'

j=0

# 创建一个空的列表用于保存每次循环的均方误差
mse_values = []

# 在循环外部创建工作簿和工作表
workbook = Workbook()
worksheet = workbook.active

# 设置列标题
worksheet['A1'] = 'seller_no'
worksheet['B1'] = 'product_no'
worksheet['C1'] = 'warehouse_no'
worksheet['D1'] = 'timestamp'
worksheet['E1'] = 'forecast_qty'

start_row = 2  # 从第二行开始写入数据


#time_series_lists_test=time_series_lists[:5]


for time_series in time_series_lists:
    # 创建一个模拟的时间序列列表
    time_series_length = len(time_series)
    # 创建特征矩阵
    window_size = 10  # 用于特征提取的滑动窗口大小
    X = []
    y = []
    for i in range(time_series_length - window_size):
        X.append(time_series[i:i+window_size])
        y.append(time_series[i+window_size])

    # 划分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # 使用随机森林回归模型
    rf_model = RandomForestRegressor(n_estimators=100, random_state=10, max_depth = 100, min_samples_split = 2)
    rf_model.fit(X_train, y_train)

    # 进行预测
    y_pred = rf_model.predict(X_test)

    # 计算均方误差
    mse = mean_squared_error(y_test, y_pred)

    #记录循环次数 输出并保留均方误差
    j+=1
    print(f"第{j}个维度的mse为：{mse}")

    mse_values.append(mse)

    # 使用训练好的模型预测未来时间步
    future_steps = 15
    last_window = time_series[-window_size:]  # 最后一个已知时间窗口
    forecast = []
    for _ in range(future_steps):
        # 预测下一个时间步
        next_step_pred = rf_model.predict([last_window])[0]
        forecast.append(next_step_pred)

        # 更新时间窗口
        last_window = np.roll(last_window, -1)  # 移除第一个值
        last_window[-1] = next_step_pred  # 添加预测的值


    time_series_keys=get_keys_by_value(time_series_dict, time_series)#取得时间序列对应的xyz元组 并保存在列表中
    file_name_str = '_'.join(str(item) for item in time_series_keys[0])#将该元组所有元素转化为字符串并用_连接作为文件名
    filename = f'{output_folder}/{file_name_str}.png'#确定文件名字和保存位置

    # 创建三个子图
    plt.figure(figsize=(12, 18))

    # 第一个子图：整个训练集的拟合结果
    plt.subplot(3, 1, 1)
    plt.plot(range(len(y_train)), y_train, label='Real World Value Training Set')
    plt.plot(range(len(y_train)), rf_model.predict(X_train), label='Predicted Value Training Set', linestyle='--', color='red')
    plt.title('The Fitting Result Of The Entire Training Set', fontsize=11)
    plt.xlabel('Time Step', fontsize=11)
    plt.ylabel('Value', fontsize=11)
    plt.legend()

    # 第二个子图：测试集的拟合结果
    plt.subplot(3, 1, 2)
    plt.plot(range(len(y_test)), y_test, label='Real World Test Set')
    plt.plot(range(len(y_test)), rf_model.predict(X_test), label='Predicted Value Test Set', linestyle='--', color='red')
    plt.title('The Fitting Result Of The Test Set', fontsize=11)
    plt.xlabel('Time Step', fontsize=11)
    plt.ylabel('Value', fontsize=11)
    plt.legend()


    # 第三个子图：未来十五步的预测结果
    plt.subplot(3, 1, 3)
    plt.plot(range(len(time_series)), time_series, label='Known Time Series')
    plt.plot(range(len(time_series), len(time_series) + future_steps), forecast, label='Future Predictions', linestyle='--', color='red')
    plt.title('Predicted Results For The Next Fifteen Steps', fontsize=11)
    plt.xlabel('Time Step', fontsize=11)
    plt.ylabel('Value', fontsize=11)
    plt.legend()

    plt.subplots_adjust(hspace=0.3)#调整垂直间距 防止标签重叠
    
    plt.savefig(filename)#保存图片并关闭
    plt.close()

    # 预测结果保存
    

    # 设置预测的开始日期和结束日期
    start_date = datetime(2023, 5, 16)
    end_date = datetime(2023, 5, 30)

    # 生成日期范围
    date_range = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]

    #取出xyz和预测结果
    seller_no, product_no, warehouse_no = time_series_keys[0]
    forecast_qty = forecast

    # 创建一个新的数据帧以写入 Excel

    for i, date in enumerate(date_range):
        row = start_row + i
    
        worksheet[f'A{row}'] = seller_no
        worksheet[f'B{row}'] = product_no
        worksheet[f'C{row}'] = warehouse_no
        worksheet[f'D{row}'] = date
        worksheet[f'E{row}'] = forecast_qty[i]
    
    start_row+=15

    # 打印第15行数据作为参考
    print(f"Row {row-1}: Seller No: {seller_no}, Product No: {product_no}, Warehouse No: {warehouse_no}, Timestamp: {date}, Forecast Qty: {forecast_qty[i]},结束行数：{start_row-1}")


# 保存工作簿为 Excel 文件
workbook.save('E:/Data_storage/result1.xlsx')


plt.figure(figsize=(8, 6))
plt.plot(range(1, len(mse_values) + 1), mse_values, marker='o', linestyle='-', color='b')
plt.title('Change In Mean Square Error')
plt.xlabel('Number Of Cycles')
plt.ylabel('Mean Square Error')
plt.grid(True)
plt.show()


