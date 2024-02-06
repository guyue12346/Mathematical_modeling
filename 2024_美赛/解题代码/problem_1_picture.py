import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


file_path = '/Users/guyue/知识模块/Mathematical_modeling/2024_美赛/解题数据/problem_data/output_file_2023-wimbledon-1701.csv'
data = pd.read_csv(file_path)

target_column_name = ['server','point_victor','p1_sets','p1_games','p1_score','p2_sets','p2_games','p2_score','p1_ace','p2_ace','serve_no']
Parameters = {}
num_data=[]

for i in range(1000):
    row_index = i
    if row_index < len(data):
        for j in target_column_name:
            col_name = j
            element = data.at[i, j]
            if(element=='0'):
                element=0
            elif(element=='15'):
                element=1
            elif(element=='30'):
                element=2
            elif(element=='40'):
                element=3
            elif(element=='AD'):
                element=4
            num_data.append(element)
        Parameters[row_index]=num_data
        num_data=[]

#Parameters字典包含了每个点的所需数据
player1_target = {}
player2_target = {}

# 目标数据：球权（1，2）x_r 得失分（1，2）N 连续得分 x_1 发球方失误次数（0，1，2）x_2 局点（0，1）x_g 盘点（0，1）x_s 发球直接得分 x_3
for i in range(1000):
    value_num = Parameters.get(i)
    if value_num is not None:

        player1_num = [0, 0, 0, 0, 0, 0, 0]  # 初始化 player1_num 列表
        player2_num = [0, 0, 0, 0, 0, 0, 0]  # 初始化 player2_num 列表

        con_num = 1

        if value_num[1] == 1:
            player1_num[0] = 1
            player2_num[0] = 0
        else:
            player1_num[0] = 0
            player2_num[0] = 1

        if value_num[1] == 1:
            for t in range(i - 1, 0, -1):
                value_before = Parameters.get(t)
                if value_before[1] == 1:
                    con_num += 1
                else:
                    break
        else:
            for t in range(i - 1, 0, -1):
                value_before = Parameters.get(t)
                if value_before[1] == 2:
                    con_num += 1
                else:
                    break
        player1_num[1] = player2_num[1] = con_num

        if value_num[10] == 1:
            player1_num[2] = player2_num[2] = 0
        elif value_num[10] == 2 and value_num[1] == 1:
            player1_num[2] = 1
        else:
            player2_num[2] = 1

        if value_num[8] == 1 and value_num[9] == 0:
            player1_num[3] = 1
            player2_num[3] = 0
        elif value_num[8] == 0 and value_num[9] == 1:
            player1_num[3] = 0
            player2_num[3] = 1
        else:
            player1_num[3] = 0
            player2_num[3] = 0

        if value_num[0] == 1:
            player1_num[4] = 1
            player2_num[4] = 0
        else:
            player1_num[4] = 0
            player2_num[4] = 1

        if (int(value_num[4]) == 3 and int(value_num[7]) < 3) or (int(value_num[4]) < 3 and int(value_num[7]) == 3) or (int(value_num[4]) == 4 or int(value_num[7]) == 4):
            player1_num[5] = player2_num[5] = 1
        else:
            player1_num[5] = player2_num[5] = 0

        if ((int(value_num[3]) == 6 and (int(value_num[6]) == 6 or int(value_num[6]) == 5)) or
            (int(value_num[3]) == 5 and int(value_num[6]) == 4)) or ((int(value_num[6]) == 6 and (int(value_num[3]) == 6 or int(value_num[3]) == 5)) or (int(value_num[6]) == 5 and int(value_num[3]) == 4)):
            player1_num[6] = player2_num[6] = 1
        else:
            player1_num[6] = player2_num[6] = 0

        player1_target[i] = player1_num.copy() 
        player2_target[i] = player2_num.copy()

def F_fun(initial_parameters,n,t):
            q=0.6731191652937946
            if t==1:
                player=player1_target[n]
            else:
                player=player2_target[n]
            k1, k2, k3, k4, k5, k6 = initial_parameters
            F=((1-player[4])*(q/(1-q))+player[4])*player[0]*(1+k1*(player[1])-k2*player[2]*player[4]+k3*(player[5]+player[6]))-k4*((q/(1-q))*player[4]+(1-player[4]))*(1-player[0])*(1+k1*(player[1]-1)+k5*player[2]*player[4]+k3*(player[5]+player[6])+k6*player[3])
            return F

y=[]
yy=[]

for i in range(len(Parameters)):
    y.append(F_fun([56.46144093  ,-26.96010033  ,195.48414152 ,2 , 20,20],i,1)/1000)
    yy.append(F_fun([56.46144093  ,-26.96010033  ,195.48414152 ,2 , 20,20],i,2)/1000)

y1=[]
y2=[]
for i in range(len(y)):
    a = 0
    b = 0
    for j in range(i+1):
        a+=y[j]
        b+=yy[j]
    y1.append(a)
    y2.append(b)
# 横坐标为整数序列1、2、3...
x = list(range(1, len(y) + 1))

dy=[]
for i in range(len(y)):
    dy.append(y1[i]-y2[i])
'''
degree = 10
coefficients_y = np.polyfit(x, y, degree)
coefficients_yy = np.polyfit(x, yy, degree)

fit_y = np.polyval(coefficients_y, x)
fit_yy = np.polyval(coefficients_yy, x)
'''

plt.plot(x, y, label='Player 1',linewidth=1)
plt.plot(x, yy, label='Player 2',linewidth=1)

#plt.plot(x, fit_y, label='Player 1')
#plt.plot(x, fit_yy, label='Player 2')


# 绘制折线图
#plt.plot(x, y, label='Player 1')
#plt.plot(x, yy, label='Player 2')
#plt.plot(x, dy, label='Momentum Trend')

# 设置图表标题和坐标轴标签
#plt.title('Momentum Vision')
plt.xlabel('Points')
plt.ylabel('performance evaluation')

# 添加图例
plt.legend()

# 显示图表
plt.show()
