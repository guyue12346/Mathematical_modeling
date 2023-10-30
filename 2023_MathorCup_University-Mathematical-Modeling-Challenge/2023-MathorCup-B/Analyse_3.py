import pandas as pd

'''
file_path = 'E:\Data_storage\Data_5.xlsx'
df = pd.read_excel(file_path)

df['seller_no']=df['seller_no'].str.replace('[^\d]','',regex=True)
df['product_no']=df['product_no'].str.replace('[^\d]','',regex=True)
df['warehouse_no']=df['warehouse_no'].str.replace('[^\d]','',regex=True)

df.to_excel('Data_5_analyse.xlsx',index=False)
'''

file_path = 'E:\Data_storage\Data_5_analyse.xlsx'
df = pd.read_excel(file_path)

# 获取唯一的组合
unique_combinations = df[['seller_no', 'product_no', 'warehouse_no']].drop_duplicates()

# 创建一个空字典来存储每个唯一组合的数据
data_dict_5 = {}

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
    data_dict_5[(seller_no, product_no, warehouse_no)] = qty_list

time_series_list_5=list(data_dict_5.values()) # 包含210个时间序列的数据 每个时间序列长不一样


