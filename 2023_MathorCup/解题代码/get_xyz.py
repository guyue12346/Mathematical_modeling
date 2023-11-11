from MMtools import DTM_k
from MMtools import get_data_dict


data_dict=get_data_dict()

key_list=DTM_k()

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

