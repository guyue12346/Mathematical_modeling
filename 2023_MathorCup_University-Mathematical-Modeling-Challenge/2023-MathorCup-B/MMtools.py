import numpy as np
import random
import statsmodels.api as sm
import pandas as pd

def DTM_k(k):

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


    # 随机选择k个数据点作为初始聚类中心
    centroids = random.sample(data, k)

    # 最大迭代次数
    max_iterations = 10000

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
    '''
    # 输出最终的簇中心和分配结果
    for i, centroid in enumerate(centroids):
        print(f'Cluster {i + 1} Center: {centroid}')
        print(f'Cluster {i + 1} Points: {clusters[i]}')
    '''

    return clusters

def Enhance_wmape(list1,list2):#list1为真实值
    diff_sum = 0
    for i in range(len(list1)):
        diff_sum += abs(list1[i] - list2[i])

    sum_list1 = sum(list1)
    wmape=(1-(diff_sum/sum_list1))*100

    formatted_wmape = "{:.2f}%".format(wmape)

    return formatted_wmape

def DW(list):

    residuals=np.array(list)

    dw_statistic, dw_p_value = sm.stats.durbin_watson(residuals)

    return dw_statistic, dw_p_value

def DTM_k_6(k):

    file_path = "E:\Data_storage\distance_6.txt"  
    # 读取文本文件中的距离值
    with open(file_path, "r") as file:
        lines = file.readlines()

    # 初始化一个空的距离矩阵
    distance_matrix = np.zeros((1957, 1957), dtype=float)

    # 解析文本文件中的距离值并还原成矩阵
    for i, line in enumerate(lines):
        row_values = [float(value) for value in line.split()]
        for j in range(i, 1957):
            distance_matrix[i][j] = row_values[j - i]
            distance_matrix[j][i] = row_values[j - i]

    #distance_matrix 是一个 有1996行，每一行有一个长为1996的列表 的矩阵

    distance_matrix[i][j]#是指第i个时间序列和第j个时间序列的DTM距离 0~1995


    data = list(range(1957))


    # 随机选择k个数据点作为初始聚类中心
    centroids = random.sample(data, k)

    # 最大迭代次数
    max_iterations = 200

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

def Creat_Excel_1(list1):
    # 创建一个空的DataFrame
    df = pd.DataFrame(columns=list1, index=list1)

    # 填充对角线为1
    for i in list1:
        df.loc[i, i] = 1

    excel_file = "table.xlsx"
    df.to_excel(excel_file, index=True)

def Similarity_test_function1(list1,list2):

    file_path_2 = 'E:/Data_storage/Data_2_analyse.xlsx'

    df_2 = pd.read_excel(file_path_2)

    file_path_3 = 'E:/Data_storage/Data_3_analyse.xlsx'
    df_3 = pd.read_excel(file_path_3)

    file_path_4 = 'E:/Data_storage/Data_4_analyse.xlsx'
    df_4 = pd.read_excel(file_path_4)


    def read_xlsx_as_dict(df):
        D={}
        for index, row in df.iterrows():
            add_key = row.iloc[0]  # 获取第一列的值作为键
            add_values = row.iloc[1:].tolist()  # 获取剩余三列的值并转换为列表
            D[add_key] = add_values
        return D


    L2=read_xlsx_as_dict(df_2)
    L3=read_xlsx_as_dict(df_3)
    L4=read_xlsx_as_dict(df_4)


    A=[[] for _ in range(8)]#按序储存所有的性质

    for X in list(L2.values()):
        if X[0] not in A[0]:
            A[0].append(X[0])
        if X[1] not in A[1]:
            A[1].append(X[1])
        if X[2] not in A[2]:
            A[2].append(X[2])
            
    for X1 in list(L3.values()):
        if X1[0] not in A[3]:
            A[3].append(X1[0])
        if X1[1] not in A[4]:
            A[4].append(X1[1])
        if X1[2] not in A[5]:
            A[5].append(X1[2])

    for X2 in list(L4.values()):
        if X2[0] not in A[6]:
            A[6].append(X2[0])
        if X2[1] not in A[7]:
            A[7].append(X2[1])


    lis=[]#类元数组XYZ
    def trans(lis):
        lisA=[L2[lis[0]][0],L2[lis[0]][1],L2[lis[0]][2],L3[lis[1]][0],L3[lis[1]][1],L3[lis[1]][2],L4[lis[2]][0],L4[lis[2]][1]]
        #好像可以切片
        lisA_arrow=[]
        #lisA储存了性质字符串，lisA_arrow储存输出的数阵
        for i in range(8):
            lisA_arrow.append([0 for _ in range(len(A[i]))])#初始化输出数阵
        #    print(i,len(A[i]))
        #    print(i,len(lisA_arrow[i]))
        #print(lisA_arrow)
        for j in range(8):
            #print(j,A[j].index(lisA[j]))
            lisA_arrow[j][A[j].index(lisA[j])]=1
        #print(lisA_arrow)
        return lisA_arrow


    def evaluate(A,B):
        s=0
        for i in range(8):
            for j in range(len(A[i])):
                s+=A[i][j]*B[i][j]
        return s


    list01=[list1[1],list1[0],list1[2]]
    list02=[list2[1],list2[0],list2[2]]

    return evaluate(trans(list01),trans(list02))

def get_data_dict():
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
    return data_dict

def get_xyz(k):

    data_dict=get_data_dict()

    key_list=DTM_k(k)

    xyz_list=[]
    xyz_lists=[]

    xyz_keys=list(data_dict.keys())

    for i in range(len(key_list)):
        for j in key_list[i]:
            xyz_key=xyz_keys[j]
            xyz_list.append(xyz_key)
        xyz_lists.append(xyz_list)
        xyz_list=[]
    #xyz_lists包含k个元素 每个元素是一个列表 每个列表里有n个元组

    xyz_list_list = []
    xyz_list_lists = []

    for i in range(len(xyz_lists)):
        for tpl in xyz_lists[i]:
            lst = list(tpl)
            xyz_list_list.append(lst)
        xyz_list_lists.append(xyz_list_list)
        xyz_list_list=[]

    #xyz_list_lists 包含k个元素 每个元素是一个列表 每个列表里有n个列表

    return xyz_list_lists

def get_xyz_tpl(k):
    data_dict=get_data_dict()

    key_list=DTM_k(k)

    xyz_list=[]
    xyz_lists=[]

    xyz_keys=list(data_dict.keys())

    for i in range(len(key_list)):
        for j in key_list[i]:
            xyz_key=xyz_keys[j]
            xyz_list.append(xyz_key)
        xyz_lists.append(xyz_list)
        xyz_list=[]
    #xyz_lists包含k个元素 每个元素是一个列表 每个列表里有n个元组

    return xyz_lists



