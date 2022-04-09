import pandas as pd

if __name__=='__main__':
    df=pd.read_csv('database.csv')
    df.drop(columns=['Metal'])
    df.drop(columns=['Ceramic'])
    df2=pd.read_csv('metal_ceramics_data_all.csv',index_col=0)
    df3=pd.concat([df2,df],join='outer',axis=1)
    df3.to_csv('metal_ceramic_data_all_with_A_T.csv')