import numpy as np
import random

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

#distance_matrix 是一个 有1996行，每一行有一个长为1996的列表 的矩阵

distance_matrix[i][j]#是指第i个时间序列和第j个时间序列的DTM距离 0~1995


data = list(range(1996))

# 定义K值（簇的数量）
k = 10

# 随机选择k个数据点作为初始聚类中心
centroids = random.sample(data, k)

# 最大迭代次数
max_iterations = 100

for _ in range(max_iterations):
    # 初始化簇字典，将每个簇初始化为空
    clusters = {i: [] for i in range(k)}

    centroids_clusters={}
    
    for key in clusters:
        centroids_clusters[centroids[key]]=key
        

    # 分配数据点到最近的簇
    for point in data:
        distance = []
        for i in centroids:
            distance_value = distance_matrix[i][point]
            distance.append(distance_value)
        argmin=np.argmin(distance)

        key_add=centroids[argmin]

        value_add=point

        key_add_2=centroids_clusters[key_add]

        clusters[key_add_2].append(value_add)


    # 计算新的簇中心
    new_centroids = []
    for i in range(k):
        distance_2=[]
        for j in clusters[i]:
            sum_2=0
            for l in clusters[i]:
                sum_2+=distance_matrix[l][j]
            distance_2.append(sum_2)
        argmin_2=np.argmin(distance_2)
        new_add=clusters[i][argmin_2]

        new_centroids.append(new_add)


    # 判断是否收敛
    if new_centroids == centroids:
        break
    centroids = new_centroids

# 输出最终的簇中心和分配结果
for i, centroid in enumerate(centroids):
    print(f'Cluster {i + 1} Center: {centroid}')
    print(f'Cluster {i + 1} Points: {clusters[i]}')