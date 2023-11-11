from MMtools import get_xyz
from MMtools import Similarity_test_function1
import pandas as pd

list_xyz=get_xyz(15)#list_xyz包含了聚类分为十类的xyz列表组合 得到列表只需要15秒(分10类)


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


def Similarity_test_function1(list1,list2):


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


#内部相似度检验

sum0=0
sum1=0

for i in range(len(list_xyz)):#进行10次循环
    list1=list_xyz[i]#list1为第i类的列表
    sum1=0
    for j in range(len(list1)):#依次取出列表中的每一个列表
        for k in range(len(list1)):#对每一个列表 依次取出列表中的每一个列表求其相似度并求和
            sum0+=Similarity_test_function1(list1[j],list1[k])
        '''
            print(f'第{i}类中第{j}个维度正在与第{k}个维度进行相似度检验')
        '''
        sum1+=sum0
        sum0=0
    print(f'第{i}聚类共{len(list1)}个维度')
    print(f'第{i}聚类的相似度均值为{sum1/(len(list1)*len(list1))}')

sum3=0
sum4=0

for i in range(len(list_xyz)):
    list1=list_xyz[i]
    list2=list_xyz[i-2]
    sum4=0
    for j in range(len(list1)):
        for k in range(len(list2)):
            sum3+=Similarity_test_function1(list1[j],list2[k])
        sum4+=sum3
        sum3=0
    print(f'第{i}聚类和第{i-1}聚类的相似度均值为{sum4/(len(list1)*len(list2))}')

