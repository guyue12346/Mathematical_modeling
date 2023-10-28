import pandas as pd
from fastdtw import fastdtw


file_path = 'E:\Data_storage\Data_1_analyse.xlsx'
df = pd.read_excel(file_path)

# 获取唯一的组合
unique_combinations = df[['seller_no', 'product_no', 'warehouse_no']].drop_duplicates()

# 创建一个空字典来存储每个唯一组合的数据
data_dict = {}

# 遍历唯一组合
for index, row in unique_combinations.iterrows():
    seller_no = row['seller_no']
    product_no = row['product_no']
    warehouse_no = row['warehouse_no']
    
    # 选择对应唯一组合的数据
    subset = df[(df['seller_no'] == seller_no) & (df['product_no'] == product_no) & (df['warehouse_no'] == warehouse_no)]
    
    # 根据时间戳排序数据
    subset = subset.sort_values(by='date')
    
    # 将qty值保存为一个列表
    qty_list = subset['qty'].tolist()
    
    # 将该唯一组合的数据保存到字典中
    data_dict[(seller_no, product_no, warehouse_no)] = qty_list

time_series_list=list(data_dict.values()) # 包含1996个时间序列的数据

distance_matrix = [[0] * 1996 for _ in range(1996)]

total_iterations = 0


# 创建一个空的距离矩阵
distance_matrix = [[0.0] * 1996 for _ in range(1996)]

# 指定保存文件的绝对路径
file_path = "E:\Data_storage\distance.txt"  

# 打开文件以写入数据
with open(file_path, "w") as file:
    # 填充距离矩阵
    for i in range(1996):
        row_values = []  # 用于存储每一行的距离值
        for j in range(i, 1996):
            total_iterations += 1
            time_series1 = time_series_list[i]
            time_series2 = time_series_list[j]
            distance, _ = fastdtw(time_series1, time_series2)
            row_values.append(distance)

            print(f"已完成 {total_iterations} 次循环")

        # 将一行的距离值写入文件并用空格分隔
        file.write(" ".join(map(str, row_values)) + "\n")

print("总循环次数:", total_iterations)


'''
file_path = "E:\Data_storage\distance.txt"  

# 读取文本文件中的距离值
with open(file_path, "r") as file:
    lines = file.readlines()

# 初始化一个空的距离矩阵
distance_matrix = np.zeros((1996, 1996), dtype=float)

# 解析文本文件中的距离值并还原成矩阵
for i, line in enumerate(lines):
    row_values = [float(value) for value in line.split()]
    for j in range(i, 1996):
        distance_matrix[i][j] = row_values[j - i]
        distance_matrix[j][i] = row_values[j - i]

# 现在 distance_matrix 变量包含了还原后的距离矩阵
print(distance_matrix)

'''
