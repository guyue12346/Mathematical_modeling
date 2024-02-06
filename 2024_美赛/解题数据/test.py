import pandas as pd

# 文件路径
file_path = "//Users/guyue/知识模块/Mathematical_modeling/2024_美赛/题目C及其数据/Wimbledon_featured_matches.csv"

# 读取CSV文件
original_data = pd.read_csv(file_path)

# 根据第一列的不同值进行分组
grouped_data = original_data.groupby('match_id')

# 遍历分组并将每个分组保存到不同的CSV文件
for group_name, group_df in grouped_data:
    # 构造输出文件名
    output_filename = f'output_file_{group_name}.csv'

    # 保存分组数据到CSV文件
    group_df.to_csv(output_filename, index=False)

    print(f"Group '{group_name}' saved to '{output_filename}'")
