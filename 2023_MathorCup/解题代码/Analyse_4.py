import pandas as pd
import matplotlib.pyplot as plt
import os

file_path = 'E:\Data_storage\Data_6_analyse.xlsx'
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

output_folder = 'E:/plot6' 
os.makedirs(output_folder, exist_ok=True)


result_df = pd.DataFrame(columns=['seller_no', 'product_no', 'warehouse_no', 'qty_data'])

# 遍历 data_dict，并将数据添加到 DataFrame
for (seller_no, product_no, warehouse_no), qty_list in data_dict.items():
    result_df = pd.concat([result_df, pd.DataFrame({'seller_no': [seller_no],
                                                    'product_no': [product_no],
                                                    'warehouse_no': [warehouse_no],
                                                    'qty_data': [qty_list]})], ignore_index=True)


    plt.figure()
    plt.plot(qty_list)
    plt.title(f'Qty Data for {seller_no}, {product_no}, {warehouse_no}')
    plt.xlabel('Time')
    plt.ylabel('Quantity')



    # 为图像文件命名
    image_filename = os.path.join(output_folder, f'{seller_no}_{product_no}_{warehouse_no}.png')
    
    # 保存图像文件
    plt.savefig(image_filename)
    plt.close()  # 关闭图形以释放内存


'''
# 保存 DataFrame 到 Excel 文件
result_df.to_excel('output_data_6.xlsx', index=False)
'''