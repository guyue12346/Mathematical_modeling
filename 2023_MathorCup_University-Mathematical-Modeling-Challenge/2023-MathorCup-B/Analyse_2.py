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

for key in data_dict:
    print(key)


list1=list(data_dict.values())

for value in list1:
    print(value)


