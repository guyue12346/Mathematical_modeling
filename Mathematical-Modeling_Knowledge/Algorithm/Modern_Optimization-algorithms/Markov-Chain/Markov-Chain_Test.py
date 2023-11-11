import random

# 定义马尔可夫链的状态空间
states = ["Heads", "Tails"]

# 定义状态转移概率矩阵
# 例如，从Heads到Heads的概率为0.5，从Tails到Tails的概率也为0.5
transition_matrix = {
    "Heads": {"Heads": 0.5, "Tails": 0.5},
    "Tails": {"Heads": 0.5, "Tails": 0.5}
}

# 初始状态
current_state = random.choice(states)

# 模拟状态转移
num_steps = 10  # 模拟的步数
trajectory = [current_state]

for _ in range(num_steps):
    next_state = random.choices(list(transition_matrix[current_state].keys()), 
                                weights=list(transition_matrix[current_state].values()))[0]
    trajectory.append(next_state)
    current_state = next_state

# 打印状态序列
print("马尔可夫链状态序列:")
print(" -> ".join(trajectory))
