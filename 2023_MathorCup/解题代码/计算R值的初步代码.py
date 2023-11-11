import numpy as np
#import matplotlib.pyplot as plt
import random
#from scipy import stats
#from distfit import distfit
#from fitter import Fitter

def QT2DT(lisQ):
    lisD0=[]
    lisD=[]
    lisd=[]
    for i in range(len(lisQ)-1):
        lisD0.append(lisQ[i+1]-lisQ[i])#获得步长序列

    for aa in lisD0:#分别统计正步长和负步长
        if aa>=0:
            lisD.append(aa)
        else:
            lisd.append(aa)
    return [lisD0, lisD, lisd]
'''
listD0_sort=sorted(lisD0)

for i in listD0_sort:
    print(i)
'''
def data_to_probability(data):#根据步长数据lisD0返回离散概率分布，每种概率对应的采样值
    m = min(data)
    M = max(data)
    d = (M-m)/20

    idex=[]
    for i in range(20):
        idex.append(m+(i+0.5)*d-sum(data)/len(data))
    
    pinglv=[0 for _ in range(20)]
    for i in range(len(data)):
        for j in range(20):
            if data[i] >= m+j*d and data[i] <= m+(j+1)*d:
                pinglv[j] += 1
                break
    
    gailv=[]
    for i in range(20):
        gailv.append(pinglv[i]/len(data))
    return [gailv,idex]

'''
lis_sample=data_to_probability(lisD0)
list_probability = lis_sample[0] #[0.005, 0.015, 0.08, 0.25, 0.3, 0.25, 0.08, 0.015, 0.005]
idex = lis_sample[1]

print(idex)
probability_index = random.choices(idex, weights=list_probability, k=1)[0]#从idex中单次抽样
'''
def eva_fluc(lis_fluc,Q0):#仅当不三次同号且波动不导致负出货量时返回True
    #print('####',sum(lis_fluc),Q0)
    if len(lis_fluc)<=3:
        return True
    elif sum(lis_fluc)+Q0 >= 0 and (lis_fluc[-1]*lis_fluc[-2] <= 0 or lis_fluc[-2]*lis_fluc[-3] <= 0):
        return True
    else:
        return False
    
def fluctuate(lis_prob,idex,n,Q0):#根据步长及概率分布返回模拟的波动
    lis_fluc=[]
    for i in range(n-1):
        while True:
            a = random.choices(idex,weights=lis_prob)
            if eva_fluc(lis_fluc+a,Q0):
                lis_fluc.append(a[0])
                break
    return lis_fluc

def fluctuate_model(lisQ,n):
    DDd = QT2DT(lisQ)
    if len(DDd[1])*len(DDd[2])==0:
        #print(DDd)
        return [0 for _ in range(n)]
    else:
        lis1 = data_to_probability(DDd[0])
        #print('lis1',lis1)
        return fluctuate(lis1[0],lis1[1],n,lisQ[-1])
'''
ll = fluctuate_model(lisQ,15)
for i in ll:
    print(i)
for j in lisQ:
    print(j)
'''
def ftongji3(lisQ):#输入一个时间序列即可
    data = QT2DT(lisQ)[0]
    jicha = max(data)-min(data)
    #data_sort=sorted(data)
    bzcha = np.std(data,ddof=1)
    a = 0
    aa = 2#调整此参数可调整极端波动出现次数，aa变大，次数变少
    for i in data:
        m=np.mean(data)
        if i > m+aa*bzcha or i < m-aa*bzcha:
            a+=1

    #bz = np.std(lisQ,ddof=1)
    return [jicha,bzcha,a,len(lisQ)*bzcha/sum(lisQ)]#极差，标准差，极端波动出现次数，日常相对波动指数g(与aa无关)

def cx_fluc(lisQ,lis1111):#返回促销相对指数f 日常相对波动指数g
    q = 0.99#衰减率，越大则618促销波动指数越依赖于靠近六月的数据，比如五月份数据越大则最终比值越大
    f0 = ftongji3(lis1111)[3]
    lisQ_618 = []
    lisQ_1111 = []
    for i in range(len(lisQ)):
        lisQ_618.append(lisQ[i]*(q**i))
        lisQ_1111.append(lisQ[-1-i]*(q**i))
    f = f0*sum(lisQ_618)/sum(lisQ_1111) - 1
    g = ftongji3(lisQ)[3]
    return [f,g]


    