import pandas as pd

file_path = 'E:/Data_1.xlsx'
df = pd.read_excel(file_path)

df=df.drop_duplicates()

df['seller_no']=df['seller_no'].str.replace('[^\d]','',regex=True)
df['product_no']=df['product_no'].str.replace('[^\d]','',regex=True)
df['warehouse_no']=df['warehouse_no'].str.replace('[^\d]','',regex=True)


df.to_excel('Data_1_analyse.xlsx',index=False)
