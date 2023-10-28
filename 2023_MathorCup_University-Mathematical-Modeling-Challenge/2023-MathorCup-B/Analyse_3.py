import pandas as pd

file_path = 'E:\Data_storage\Data_1_analyse_2.xlsx'
df = pd.read_excel(file_path)

# 将时间戳列解析为日期时间对象
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d-%H')


# 获取每个时间戳的第一行信息
unique_timestamps = df['date'].unique()  # 获取唯一的时间戳值

for timestamp in unique_timestamps:
    # 获取该时间戳的第一行信息
    first_row = df[df['date'] == timestamp].iloc[0]

    # 输出第一行信息
    print(first_row)

