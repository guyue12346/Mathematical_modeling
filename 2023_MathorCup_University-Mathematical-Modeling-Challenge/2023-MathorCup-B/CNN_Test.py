import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# 创建一个简单的卷积神经网络
model = Sequential()

# 添加一个卷积层
model.add(Conv1D(filters=16, kernel_size=3, input_shape=(3, 5), activation='relu'))

# 添加一个最大池化层
model.add(MaxPooling1D(pool_size=2))

# 将卷积层的输出展平
model.add(Flatten())

# 添加一个全连接层
model.add(Dense(10, activation='relu'))

# 添加输出层
model.add(Dense(1, activation='linear'))

# 编译模型
model.compile(optimizer='adam', loss='mean_squared_error')

# 准备时间序列数据
data = np.array([
    [1, 2, 3, 5, 7],
    [2, 4, 6, 7, 8],
    [5, 7, 8, 9, 10]
])

# 将数据整形成适合输入的形状
data = data.reshape((3, 5, 1))

# 准备目标数据（这里只是示例，你需要提供实际目标数据）
target = np.array([10, 15, 20])

# 训练模型
model.fit(data, target, epochs=100, verbose=2)

# 使用模型进行预测
predictions = model.predict(data)
print("预测结果：", predictions)
