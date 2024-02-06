import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#计算发球方胜率
'''
q_num=0
for k in range(7284):
    value_data=Parameters.get(k)
    if(value_data[0]==value_data[1]):
        q_num+=1
q_goal=q_num/7284
print(q_goal)
'''

#考虑最速下降法构成一个BP参数优化模型

#server为球权 pointer_victor为当局获胜者 在当局比赛中寻找连续得分或失分 分数差距在p1_sets games scores 是否为局点 看scores 是否为盘点 看games 
#是否发球直接得分 看p1_ace/p2_ace 

# 指定文件夹路径
folder_path = '/Users/guyue/知识模块/Mathematical_modeling/2024_美赛/解题数据/problem_data'

# 获取文件夹内所有文件
all_files = os.listdir(folder_path)

# 筛选出CSV文件
csv_files = [file for file in all_files if file.endswith('.csv')]

all_Parameters=[]
all_player1_target=[]
all_player2_target=[]

# 循环读取CSV文件
for csv_file in csv_files:
    # 构建完整的文件路径
    file_path = os.path.join(folder_path, csv_file)
    
    # 使用pandas读取CSV文件
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
    all_Parameters.append(Parameters)
    all_player1_target.append(player1_target)
    all_player2_target.append(player2_target)

def T_fun(initial_parameters):
    all_sum=0
    for p in range(len(all_Parameters)):
        player1_target=all_player1_target[p]
        player2_target=all_player2_target[p]
        Parameters=all_Parameters[p]

        # 目标函数F
        def F_fun(initial_parameters,n,t):
            q=0.6731191652937946
            if t==1:
                player=player1_target[n]
            else:
                player=player2_target[n]
            k1, k2, k3, k4, k5, k6 = initial_parameters
            F=((1-player[4])*(q/(1-q))+player[4])*player[0]*(1+k1*(player[1])-k2*player[2]*player[4]+k3*(player[5]+player[6]))-k4*((q/(1-q))*player[4]+(1-player[4]))*(1-player[0])*(1+k1*(player[1]-1)+k5*player[2]*player[4]+k3*(player[5]+player[6])+k6*player[3])
            return F

        sum=0
        for l in range(len(player1_target)):
            if l in Parameters:
                value_list=Parameters[l]
                if value_list[1]==1:
                    N=1
                else:
                    N=0
                sum+=((F_fun(initial_parameters, l, 1) - F_fun(initial_parameters, l, 2)) / (F_fun(initial_parameters, l, 1) + F_fun(initial_parameters, l, 2)) - N) ** 2
            else:
                break
        all_sum+=sum
    return all_sum


def gradient(initial_parameters):
    del_initial_parameters_1=[0,0,0,0,0,0]
    del_initial_parameters_2=[0,0,0,0,0,0]
    del_initial_parameters_3=[0,0,0,0,0,0]
    del_initial_parameters_4=[0,0,0,0,0,0]
    del_initial_parameters_5=[0,0,0,0,0,0]
    del_initial_parameters_6=[0,0,0,0,0,0]
    i=1
    for a in initial_parameters:
        if i ==1:
            del_initial_parameters_1[0]=a+0.001

            del_initial_parameters_2[0]=a
        
            del_initial_parameters_3[0]=a
    
            del_initial_parameters_4[0]=a

            del_initial_parameters_5[0]=a
    
            del_initial_parameters_6[0]=a

        elif i==2:
            del_initial_parameters_1[1]=a

            del_initial_parameters_2[1]=a+0.001
        
            del_initial_parameters_3[1]=a
    
            del_initial_parameters_4[1]=a

            del_initial_parameters_5[1]=a
    
            del_initial_parameters_6[1]=a

        elif i==3:
            del_initial_parameters_1[2]=a

            del_initial_parameters_2[2]=a
        
            del_initial_parameters_3[2]=a+0.001
    
            del_initial_parameters_4[2]=a

            del_initial_parameters_5[2]=a
    
            del_initial_parameters_6[2]=a

        elif i==4:
            del_initial_parameters_1[3]=a

            del_initial_parameters_2[3]=a
        
            del_initial_parameters_3[3]=a
    
            del_initial_parameters_4[3]=a+0.001

            del_initial_parameters_5[3]=a
    
            del_initial_parameters_6[3]=a

        elif i==5:
            del_initial_parameters_1[4]=a

            del_initial_parameters_2[4]=a
        
            del_initial_parameters_3[4]=a
    
            del_initial_parameters_4[4]=a

            del_initial_parameters_5[4]=a+0.001
    
            del_initial_parameters_6[4]=a
        
        elif i==6:
            del_initial_parameters_1[5]=a

            del_initial_parameters_2[5]=a
        
            del_initial_parameters_3[5]=a
    
            del_initial_parameters_4[5]=a

            del_initial_parameters_5[5]=a
    
            del_initial_parameters_6[5]=a+0.001
        i+=1

    df_k1=(T_fun(del_initial_parameters_1)-T_fun(initial_parameters))*1000
    df_k2=(T_fun(del_initial_parameters_2)-T_fun(initial_parameters))*1000
    df_k3=(T_fun(del_initial_parameters_3)-T_fun(initial_parameters))*1000
    df_k4=(T_fun(del_initial_parameters_4)-T_fun(initial_parameters))*1000
    df_k5=(T_fun(del_initial_parameters_5)-T_fun(initial_parameters))*1000
    df_k6=(T_fun(del_initial_parameters_6)-T_fun(initial_parameters))*1000
    gradients = []
    gradients.append(df_k1)
    gradients.append(df_k2)
    gradients.append(df_k3)
    gradients.append(df_k4)
    gradients.append(df_k5)
    gradients.append(df_k6)
    return np.array(gradients)


# 最速下降法
def gradient_descent(initial_params, learning_rate, max_iterations, tolerance):
    params = np.array(initial_params)
    iteration = 0

    while iteration < max_iterations:
        grad = gradient(params)

        params = params - learning_rate * grad

        # 计算梯度的范数，用作停止条件
        gradient_norm = np.linalg.norm(grad)

        print(f"Iteration {iteration + 1}, Parameters: {params}, Gradient Norm: {gradient_norm}")

        if gradient_norm < tolerance:
            break

        iteration += 1

        print(T_fun(params))
    
    return params

# 初始参数、学习率、最大迭代次数和停止条件
initial_params = [ 10,  26,  23, 75,   20,   20]
learning_rate = 100
max_iterations = 200
tolerance = 1e-6

# 运行最速下降法
result = gradient_descent(initial_params, learning_rate, max_iterations, tolerance)

print("最优参数:", result)
print("最优函数值:", T_fun(result))

