from data_found_for_one_mental_one_ceramic import data_found
import pandas as pd
import numpy as np

if __name__ == '__main__':
    data_test_composition = pd.read_csv('database.csv')
    data  = np.array(data_test_composition)

    for i in range(0,1039):
        if i:
            data_found(data[i][0],data[i][1]).to_csv('metal_ceramics_data_all.csv',mode='a',index=True,header=False)
        else:
            data_found(data[i][0],data[i][1]).to_csv('metal_ceramics_data_all.csv', mode='a')

    df=pd.read_csv('metal_ceramics_data_all.csv')
    df.reset_index(drop=True)
    dff=df.loc[:, ~df.columns.str.match('Unnamed')]
    dff.to_csv('metal_ceramics_data_all.csv')



