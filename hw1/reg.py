import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from math import floor

# filename = sys.argv[1]
# train_csv = pandas.read_csv(filename, encoding='big5')


def csvToDf(filePath):

    # read csv to create a dataframe without 測站
    #            日期          測項     0     1     2     3     4     5     6     7  ... 23
    # 0       2014/1/1    AMB_TEMP    14    14    14    13    12    12    12    12   
    # 1       2014/1/1         CH4   1.8   1.8   1.8   1.8   1.8   1.8   1.8   1.8   
    # 2       2014/1/1          CO  0.51  0.41  0.39  0.37  0.35   0.3  0.37  0.47   
    # 3       2014/1/1        NMHC   0.2  0.15  0.13  0.12  0.11  0.06   0.1  0.13 
    # ...
    df = pd.read_csv(
    filePath, 
    encoding='big5',
    usecols = ['日期','測項','0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23']
    )

    # massage a DataFrame into a format where one or more columns are identifier variables (id_vars), 
    # while all other columns, considered measured variables (value_vars), are “unpivoted” to the row axis, 
    # leaving just two non-identifier columns, ‘variable’ and ‘value’ 
    #             日期          測項  小時    數值
    # 0         2014/1/1    AMB_TEMP   0    14
    # 1         2014/1/1         CH4   0   1.8
    # 2         2014/1/1          CO   0  0.51
    # 3         2014/1/1        NMHC   0   0.2
    # ...
    df = pd.melt(df, id_vars=['日期','測項'], var_name='小時', value_name='數值')
  
    # convert 日期 and merge 小時 together in python's format
    # delete column of 小時 at the end
    df[['小時']] = df[['小時']].apply(pd.to_numeric)
    df['日期'] = pd.to_datetime(df['日期']) + pd.to_timedelta(df['小時'], unit='h')
    del df['小時']

    # set 日期 as index of the dataframe
    df = df.set_index(['日期'])
    
    # use 日期 & 小時 as index and 測項 as columns to create a new dataframe
    df = df.pivot(columns='測項', values='數值')
    return df



def dfToListDf(df):
    
    lst_dt = []  # list of datetimes
    for month in range(1, 10):
        dt1 = datetime(2014, month, 1, 0, 0, 0)  # start datetime
        dt2 = datetime(2014, month, 20, 14, 0, 0)  # end datetime
        delta = dt2 - dt1         # timedelta
        for i in range(int(delta.total_seconds() / 3600) + 1):
            # print(dt1 + timedelta(hours=i))
            lst_dt.append(dt1 + timedelta(hours=i))

    lst_df = []
    for dt in lst_dt:
            lst_df.append((df.loc[dt : dt + timedelta(hours=8)], df.loc[dt + timedelta(hours=9)]['PM2.5']))
    return lst_df

def dfScale(df):
    # convert object to float
    df = df.apply(pd.to_numeric)
    # mean-normalization
    df=(df-df.mean())/df.std()
    # min-max normalization
    # normalized_df=(df-df.min())/(df.max()-df.min())
    return df

def rainfall(df, NRvalue):
    # Another approach with defect: leave null value for unassigned value
    # df_trn[('數值', 'RAINFALL')] = df_trn[('數值', 'RAINFALL')].map({'NR': 0})
    df['RAINFALL'] = df['RAINFALL'].replace('NR', NRvalue)
    return df

# def chunks(lst, n):
#     """Yield successive n-sized chunks from l."""
#     for i in range(0, len(lst), n):
#         yield lst[i:i + n]

def split(lstDf, validRate):
    # np_df = np.array(lstDf)
    # np.random.shuffle(df)
    # Randomness has not been implemented yet
    split_idx = floor(len(lstDf) * validRate)
    valid, train = lstDf[:split_idx], lstDf[split_idx:]
    return valid, train

df = csvToDf('/home/weichenyeh/Documents/train.csv')
df = rainfall(df, 0)
df = dfScale(df)
lst_df = dfToListDf(df)
lst_df_vld, lst_df_trn = split(lst_df, 0.1)

# print(lst_df_X)



# df.to_csv('pivoted_train.csv', encoding='big5')

# df_test = testCsvToDataframe('/home/weichenyeh/Documents/test.csv')
# print(df_test)
# df_test.to_csv('pivoted_test.csv', encoding='big5')



# x_data = [ 338.,  333.,  328. , 207. , 226.  , 25. , 179. ,  60. , 208.,  606.]
# y_data = [  640.  , 633. ,  619.  , 393.  , 428. ,   27.  , 193.  ,  66. ,  226. , 1591.]

# # ydata = b + w * xdata 
# b = -120 # initial b
# w = -4 # initial w
# lr = 1 # learning rate
# iteration = 100000

# b_lr = 0.0
# w_lr = 0.0

# # Store initial values for plotting.
# b_history = [b]
# w_history = [w]

# # Iterations
# for i in range(iteration):
    
#     b_grad = 0.0
#     w_grad = 0.0
#     for n in range(len(x_data)):        
#         b_grad = b_grad  - 2.0*(y_data[n] - b - w*x_data[n])*1.0
#         w_grad = w_grad  - 2.0*(y_data[n] - b - w*x_data[n])*x_data[n]
    
#     b_lr = b_lr + b_grad**2
#     w_lr = w_lr + w_grad**2
    
#     # Update parameters.
#     b = b - lr/np.sqrt(b_lr) * b_grad 
#     w = w - lr/np.sqrt(w_lr) * w_grad
    
#     # Store parameters for plotting
#     b_history.append(b)
#     w_history.append(w)