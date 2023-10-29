import numpy as np

# 定义激活函数（这里使用Sigmoid函数）
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 定义神经网络的参数
input_size = 5
hidden_size = 4
output_size = 1
learning_rate = 0.1
epochs = 1000

# 准备训练数据
data = np.array([
    [1, 2, 3, 5, 7],
    [2, 4, 6, 7, 8],
    [5, 7, 8, 9, 10]
])
target = np.array([10, 15, 20])

# 初始化权重和偏差
input_layer_weights = np.random.rand(input_size, hidden_size)
input_layer_bias = np.random.rand(1, hidden_size)
output_layer_weights = np.random.rand(hidden_size, output_size)
output_layer_bias = np.random.rand(1, output_size)

# 训练神经网络
for epoch in range(epochs):
    # 前向传播
    input_layer_output = sigmoid(np.dot(data, input_layer_weights) + input_layer_bias)
    output_layer_output = np.dot(input_layer_output, output_layer_weights) + output_layer_bias
    
    # 计算损失
    loss = 0.5 * np.mean((output_layer_output - target) ** 2)
    
    # 反向传播
    output_layer_delta = (output_layer_output - target) / len(data)
    input_layer_delta = np.dot(output_layer_delta, output_layer_weights.T) * input_layer_output * (1 - input_layer_output)
    
    # 更新权重和偏差
    output_layer_weights -= learning_rate * np.dot(input_layer_output.T, output_layer_delta)
    output_layer_bias -= learning_rate * np.sum(output_layer_delta, axis=0, keepdims=True)
    input_layer_weights -= learning_rate * np.dot(data.T, input_layer_delta)
    input_layer_bias -= learning_rate * np.sum(input_layer_delta, axis=0, keepdims=True)
    
    # 打印每一轮的损失
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# 输出训练后的模型预测
input_layer_output = sigmoid(np.dot(data, input_layer_weights) + input_layer_bias)
final_predictions = np.dot(input_layer_output, output_layer_weights) + output_layer_bias
print("最终预测结果：", final_predictions)
