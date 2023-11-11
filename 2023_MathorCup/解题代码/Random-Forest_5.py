from MMtools import get_data_dict_5
from MMtools import get_data_dict
from fastdtw import fastdtw
import pandas as pd
from MMtools import Similarity_test_function1
from MMtools import get_keys_by_value
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from openpyxl import Workbook


time_series_dict,start_time_lists=get_data_dict_5()#附件五的时间序列字典
time_series_dict_1=get_data_dict()#附件一的时间序列字典

# 将时间戳字符串转换为datetime对象
date_objects = [pd.to_datetime(timestamp) for timestamp in start_time_lists]

min_date =pd.to_datetime("2022-12-1")

days_since_start = [(date - min_date).days for date in date_objects]#包含了和附件1起始时间的差值


time_series_lists=list(time_series_dict.values())
time_series_lists_1=list(time_series_dict_1.values())

#dtw筛选
i=0

dict_dtw={}
dict_dtw_1_5=[]


#取出对应的时间序列
for time_series in time_series_lists:
    distance=days_since_start[i]
    length=len(time_series)
    for time_series_1 in time_series_lists_1:
        new_list = time_series_1[distance:distance+length-1]
        dtw,_=fastdtw(time_series,new_list)
        dict_dtw_1_5.append(dtw)
    dict_dtw[i]=dict_dtw_1_5
    dict_dtw_1_5=[]
    i+=1
    print(f'i为{i}')


#dict_dtw是一个字典，key从0~209 value是210个列表 每个列表存储了对应时间序列和1996个时间序列重合部分的dtw距离

dtw_lists = list(dict_dtw.values())

other_list =time_series_lists_1

min_indices = []

for sublist in dtw_lists:
    # 使用enumerate函数获取每个元素的索引
    indexed_sublist = list(enumerate(sublist))
    
    # 使用sorted函数找到最小的二十个个元素的索引
    sorted_indices = sorted(indexed_sublist, key=lambda x: x[1])[:20]
    
    # 将最小的两个元素的索引添加到min_indices列表中
    min_indices.append(sorted_indices)

# 提取数据
min_elements = []

for indices in min_indices:
    elements = [other_list[index] for index, _ in indices]
    min_elements.append(elements)

print(min_elements)

#min_elements是一个1列表 里面包含210个2列表 每个2列表包含20个3列表 每个3列表里面包含166个元素
#现在我们要筛选每个2列表中的20个3列表 对20个3列表依次做相似度检验 将结果输出


#相似度检验筛选

l=0

similarity_lists=[]
similarity_lists_1_5=[]

xyz_keys_5=list(time_series_dict.keys())

for time_series in min_elements:#依次取出210个列表
    for time_series_1 in min_elements[l]:#依次取出20个时间序列
        xyz_key_1=(get_keys_by_value(time_series_dict_1,time_series_1)[0])#取出时间序列对应的key 也就是xyz组合
        similarity=Similarity_test_function1(list(xyz_keys_5[l]),xyz_key_1)#计算相似度
        similarity_lists.append(similarity)
    similarity_lists_1_5.append(similarity_lists)
    similarity_lists=[]#similarity_lists_1_5是一个1列表 里面有210个2列表 每个2列表有20个元素 表示相似度数值
    l+=1
    print(f'l为{l}')


#similarity_lists_1_5是一个列表，有210个2列表 每个列表存储了对应时间序列和20个时间序列的相似度数值

other_list =time_series_lists_1

min_indices_2 = []

for sublist in similarity_lists_1_5:
    # 使用enumerate函数获取每个元素的索引
    indexed_sublist = list(enumerate(sublist))
    
    # 使用sorted函数找到最小的十个个元素的索引
    sorted_indices = sorted(indexed_sublist, key=lambda x: x[1])[:10]
    
    # 将最小的两个元素的索引添加到min_indices_2列表中
    min_indices_2.append(sorted_indices)

# 提取数据
min_elements_2 = []

for indices in min_indices_2:
    elements = [other_list[index] for index, _ in indices]
    min_elements_2.append(elements)

#min_elements_2为一个1列表 包含210个2列表 每个2列表包含10个三列表 每个3列表包含长为166的时间序列

averaged_min_elements_2 = []

# 遍历每个2列表
for sublist in min_elements_2:
    # 初始化用于存储平均值的2列表
    averaged_sublist = [0] * 166
    
    # 遍历每个3列表
    for subsublist in sublist:
        # 累加每个3列表的对应元素到平均值2列表
        averaged_sublist = [avg + val for avg, val in zip(averaged_sublist, subsublist)]
    
    # 计算平均值，将总和除以10
    averaged_sublist = [avg / 10 for avg in averaged_sublist]
    
    # 将平均值2列表添加到结果列表中
    averaged_min_elements_2.append(averaged_sublist)

#averaged_min_elements_2包含210个平均后长为166的时间序列
#time_series_lists包含210个原始残缺时间序列
#days_since_start包含了和附件1起始时间的差值

#残缺时间序列补全
n=0

for list_1 in averaged_min_elements_2:
    start_index=days_since_start[n]
    for element in time_series_lists[n]:
        averaged_min_elements_2[n][start_index] = element
        start_index += 1
    n+=1

#averaged_min_elements_2是一个列表 这个列表包含210个补全时间序列 每个时间序列长166

#迁移学习

time_series_lists=averaged_min_elements_2

output_folder = 'E:/plot_5'

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

m=0

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


    time_series_keys=list(time_series_dict.keys())#取得时间序列对应的xyz元组 并保存在列表中
    time_series_key=time_series_keys[m]
    m+=1
    file_name_str = '_'.join(str(item) for item in time_series_key)#将该元组所有元素转化为字符串并用_连接作为文件名
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
workbook.save('E:/Data_storage/result5.xlsx')


plt.figure(figsize=(8, 6))
plt.plot(range(1, len(mse_values) + 1), mse_values, marker='o', linestyle='-', color='b')
plt.title('Change In Mean Square Error')
plt.xlabel('Number Of Cycles')
plt.ylabel('Mean Square Error')
plt.grid(True)
plt.show()

