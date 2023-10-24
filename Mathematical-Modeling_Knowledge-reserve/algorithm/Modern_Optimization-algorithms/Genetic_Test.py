import random

graph_dict_Size:10

Crossover_Probability:0.8

Mutation_Probability:0.05



def select_Polpulation(dictionary):
    keys = list(dictionary.keys())

    # 从字典中随机选择两个键
    selected_keys = random.sample(keys, 2)

    # 获取选中键对应的值
    selected_values = [dictionary[key] for key in selected_keys]

    return selected_values


def compare_and_replace(strings):
    result = ''

    for i in range(len(strings[0])):
        if strings[0][i] == strings[1][i]:
            result += strings[0][i]
        else:
            result += random.choice("01")

    return result



def mutate_bit(input_string, mutation_rate=0.05):
    if not (0 <= mutation_rate <= 1):
        raise ValueError("Mutation rate should be between 0 and 1.")

    mutated_string = ""
    for bit in input_string:
        if random.random() < mutation_rate:
            # 随机决定是否变异，以mutation_rate的概率
            mutated_bit = "0" if bit == "1" else "1"
            mutated_string += mutated_bit
        else:
            mutated_string += bit

    return mutated_string

def classify_nodes(graph_str):
    class1 = []  # 存储分类为1的节点
    class2 = []  # 存储分类为2的节点
    edges = []   # 存储图的边关系

    # 根据输入字符串构建节点分类和边关系
    for i, char in enumerate(graph_str):
        if char == '0':
            class1.append(i + 1)  # 节点编号从1开始
        else:
            class2.append(i + 1)

    # 定义图的连接关系
    graph = {
        1: [2, 5, 7],
        2: [3, 6,1],
        3: [4, 7,5],
        4: [5,3],
        5: [1,4,3],
        6: [7,2],
        7: [1,3]
    }

    # 构建图的边关系
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            if node < neighbor:
                edges.append((node, neighbor))

    # 寻找属于不同类别的节点之间的连接关系
    connected_pairs = 0
    for edge in edges:
        if (edge[0] in class1 and edge[1] in class2) or (edge[0] in class2 and edge[1] in class1):
            connected_pairs += 1

    return connected_pairs

def replace_lowest_value_with_graph(input_string, graph_dict):
    # 初始化最低值和对应的键
    lowest_value = float('inf')
    lowest_key = None

    # 遍历字典，计算各值并找到最低值的键
    for key, value in graph_dict.items():
        result = classify_nodes(value)
        if result < lowest_value:
            lowest_value = result
            lowest_key = key

    # 替换最低值的键对应的值为输入的01字符串
    graph_dict[lowest_key] = input_string

    return graph_dict

def find_max_classification_result(graph_dict):
    max_result = float('-inf')  # 初始化最大结果为负无穷

    for value in graph_dict.values():
        result = classify_nodes(value)
        max_result = max(max_result, result)

    return max_result


def main():
    graph_dict = {}

# 遍历随机生成10个个体 也可以使用贪心算法生成 但要先写适应度函数
    for i in range(1, 11):
        key = f'graph{i}'
        value = ''.join(random.choice('01') for _ in range(7))
        graph_dict[key] = value

    num_iterations = 1000

    for _ in range(num_iterations):
        selected_values=select_Polpulation(graph_dict)
        result=compare_and_replace(selected_values)
        mutated_string=mutate_bit(result, mutation_rate=0.05)
        graph_dict=replace_lowest_value_with_graph(mutated_string, graph_dict)

    max_result=find_max_classification_result(graph_dict)
    print(max_result)

if __name__ == "__main__":
    main()










