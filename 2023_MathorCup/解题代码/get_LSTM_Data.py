from MMtools import get_xyz_tpl
from MMtools import get_data_dict


list_xyz=get_xyz_tpl(10)#list_xyz包含了聚类分为十类的xyz列表组合


data_dict=get_data_dict()#时间序列数据集


LSTM_data={}#用于存放数据集的字典
LSTM_data_value=[]


for i in range(len(list_xyz)):
    key_list=list_xyz[i]
    LSTM_data_value = []
    for tpl in key_list:
        add_value=data_dict[tpl]
        LSTM_data_value.append(add_value)
    LSTM_data[i]=LSTM_data_value

#LSTM_data是一个字典 key为0~i-1的整数 value为对应类的时间序列的列表
