import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from MMtools import get_LSTM_Data
from MMtools import Prepare_time_data
from MMtools import deal_X
from tensorflow.keras.layers import Dropout
from MMtools import deal_X_2

# 处理数据集

data_all=get_LSTM_Data(15)

data = np.array(data_all[0])#data是一个列表 里边包含若干个长为166的时间序列列表

X_train,X_test,X_val=Prepare_time_data(data)#训练集  测试集 验证集 矩阵


x_train,y_train=deal_X_2(X_train)

# 调整输入数据的形状，LSTM需要3D输入数据，维度为 (samples, time_steps, features)
re_x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
re_y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)


# 构建LSTM模型
model = Sequential()

# 添加第一层LSTM层

model.add(LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))#时间步长 特征数
model.add(Dropout(0.2))  # 添加丢弃层

# 添加第二层LSTM层
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.2))  # 添加丢弃层

# 添加第三层LSTM层
model.add(LSTM(100))
model.add(Dropout(0.2))  # 添加丢弃层

# 添加输出层
model.add(Dense(1))


# 编译模型
model.compile(loss='mean_squared_error', optimizer='adam') # 使用均方误差损失函数

model.summary()

x_train,y_train=deal_X(X_train)

# 训练模型
model.fit(re_x_train, re_y_train, epochs=100, batch_size=16)  # 可根据需要调整训练时期数和批量大小

# 使用模型进行验证
x_val,y_val=deal_X(X_val)
re_x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)


predicted_values = model.predict(re_x_val)

print(predicted_values)#打印预测结果

print(y_val)

mse = np.mean(np.square(predicted_values - y_val))
print(f"均方误差 (MSE): {mse}")

x_test,y_test=deal_X(X_test)#x_test 测试集前165步二维矩阵

print(x_test[0])

for _ in range(50):

    re_x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)#重塑为三维

    predicted_output = model.predict(re_x_test)#放入训练好的模型输出166步的二维矩阵

    x_test=re_x_test.reshape(x_test.shape[0], x_test.shape[1])

    x_test=np.delete(x_test, 0, axis=1)

    x_test=np.concatenate((x_test,predicted_output), axis=1)

print(x_test[:,-50:])

