import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 定义数据集
data = np.array([
    [1, 2, 3, 5, 7],
    [2, 4, 6, 7, 8],
    [5, 7, 8, 9, 10]
])

# 准备数据，将数据集拆分成输入和输出序列
input_sequence = data[:, :-1]  # 输入序列，去掉最后一列
output_sequence = data[:, 1:]  # 输出序列，去掉第一列

# 构建LSTM模型
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(input_sequence.shape[1], 1)))
model.add(Dense(1))  # 输出层，可以根据需要调整神经元数量和激活函数

# 编译模型
model.compile(optimizer='adam', loss='mse')  # 使用均方误差损失函数

# 调整输入数据的形状，LSTM需要3D输入数据，维度为 (samples, time_steps, features)
input_sequence = input_sequence.reshape(input_sequence.shape[0], input_sequence.shape[1], 1)

# 训练模型
model.fit(input_sequence, output_sequence, epochs=100, batch_size=1)  # 可根据需要调整训练时期数和批量大小

# 进行预测
test_input = np.array([[3, 5, 7, 8]])  # 用于预测的输入序列

predicted_outputs = []

for _ in range(100):
    test_input = test_input.reshape(1, test_input.shape[1], 1)
    predicted_output = model.predict(test_input)
    predicted_output=predicted_output.reshape(1,predicted_output.shape[1],1)
    predicted_outputs.append(predicted_output[0, -1, 0])

    value_to_append=predicted_output[0, -1, 0]

    test_input = test_input.reshape(test_input.shape[0], test_input.shape[1])

    test_input = np.append(test_input[:, 1:], [[value_to_append]], axis=1)


print("预测输出:", predicted_outputs)
