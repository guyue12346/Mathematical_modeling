from MMtools import get_xyz_tpl
from MMtools import get_data_dict


list_xyz=get_xyz_tpl(10)#list_xyz包含了聚类分为十类的xyz列表组合


data_dict=get_data_dict()#时间序列数据集


LSTM_data={}#用于存放数据集的字典

for i in range(len(list_xyz)):
    
