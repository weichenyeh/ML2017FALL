import csv 
import numpy as np
from numpy.linalg import inv
import random
import math
import sys
import pandas as pd


def scale(np_arr):
    
    # np_arr = np.array(lst)
    np_trans = np.transpose(np_arr)
    df = pd.DataFrame(np_trans)

    df_max = df.max()
    df_min = df.min()

    df = (df-df.min())/(df.max()-df.min())
    
    np_trans_scaled = df.values
    np_trans_scaled_trans = np.transpose(np_trans_scaled)

    

    max_trans_scaled = df_max.values
    max_trans_scaled_trans = np.transpose(max_trans_scaled)
    # print(max_trans_scaled_trans)
    min_trans_scaled = df_min.values
    min_trans_scaled_trans = np.transpose(min_trans_scaled)
    # print(min_trans_scaled_trans)
    return np_trans_scaled_trans, max_trans_scaled_trans, min_trans_scaled_trans

def testScale(np_arr, maxi, mini):
    # convert object to float
    # mean-normalization
    # df=(df-df.mean())/df.std()
    # min-max normalization
    # print(mini)
    # print(maxi - mini)
    # np_arr = np.array(lst)
    
    for i in range(0, 18):
        np_arr[:, i*9 : (i+1)*9] = (np_arr[:, i*9 : (i+1)*9 ] - mini[i]) / (maxi[i] - mini[i])
    

    return np_arr

data = []
# 每一個維度儲存一種污染物的資訊
for i in range(18):
	data.append([])

n_row = 0
text = open('/home/weichenyeh/Documents/train.csv', 'r', encoding='big5') 
row = csv.reader(text , delimiter=",")
for r in row:
    # 第0列沒有資訊
    if n_row != 0:
        # 每一列只有第3-27格有值(1天內24小時的數值)
        for i in range(3,27):
            if r[i] != "NR":
                data[(n_row-1)%18].append(float(r[i]))
            else:
                data[(n_row-1)%18].append(float(0))	
    n_row = n_row+1
text.close()

data = np.array(data)
sqr_data = np.power(data, 2)
# print(data.shape)
# print(sqr_data.shape)
# xx = pd.DataFrame(data)
# xx.to_csv('xx.csv')
# sqr_xx = pd.DataFrame(sqr_data)
# sqr_xx.to_csv('sqr_xx.csv')
scaled_data, max_x, min_x = scale(data)
scaled_sqr_data, max_sqr_x, min_sqr_x = scale(sqr_data)

np.save('hw1_scaling_info.npy', [max_x, min_x, max_sqr_x, min_sqr_x])
# np.save('min_x.npy', min_x)
# np.save('max_sqr_x.npy', max_sqr_x)
# np.save('min_sqr_x.npy', min_sqr_x)

# print(scaled_data.shape)
# print(scaled_sqr_data.shape)

# sc_xx = pd.DataFrame(scaled_x)
# sc_xx.to_csv('sc_xx_t.csv')

x = []
y = []
sqr_x = []

# 每 12 個月
for i in range(12):
    # 一個月取連續10小時的data可以有471筆
    for j in range(471):
        x.append([])
        sqr_x.append([])
        # 18種污染物
        for t in range(18):
            # 連續9小時
            for s in range(9):
                x[471*i+j].append(scaled_data[t][480*i+j+s] )
                sqr_x[471*i+j].append(scaled_sqr_data[t][480*i+j+s] )

        y.append(data[9][480*i+j+9])


x = np.array(x)
sqr_x = np.array(sqr_x)
# df_x = pd.DataFrame(x)
# df_x.to_csv('df_x.csv')

# df_y = pd.DataFrame(y)
# df_y.to_csv('df_y.csv')

# add square term
x = np.concatenate((x,sqr_x), axis=1)
# x = np.concatenate((x,x**2), axis=1)

# add bias
x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)

# yooo = pd.DataFrame(x)
# yooo.to_csv('sample_1.csv')


w = np.zeros(len(x[0]))
l_rate = 10
repeat = 20000

x_t = x.transpose()
s_gra = np.zeros(len(x[0]))

# print(x.shape[0])

for i in range(repeat):
    hypo = np.dot(x,w)
    loss = hypo - y

    cost = np.sum(loss**2) / len(x)
    cost_a  = math.sqrt(cost)
    
    gra = np.dot(x_t,loss)
    # print(gra.shape)
    s_gra += gra**2
    # print(s_gra.shape)
    ada = np.sqrt(s_gra)
    w = w - l_rate * gra/ada
    print ('iteration: %d | Cost: %f  ' % ( i,cost_a))


# save model
np.save('sample_model.npy',w)
# read model
w = np.load('hw1_model.npy')


test_x = []
n_row = 0
text = open('/home/weichenyeh/Documents/test.csv' ,"r")
row = csv.reader(text , delimiter= ",")

for r in row:
    if n_row %18 == 0:
        test_x.append([])
        for i in range(2,11):
            test_x[n_row//18].append(float(r[i]) )
    else :
        for i in range(2,11):
            if r[i] !="NR":
                test_x[n_row//18].append(float(r[i]))
            else:
                test_x[n_row//18].append(0)
    n_row = n_row+1
text.close()

# test_xx = pd.DataFrame(test_x)
# test_xx.to_csv('test_xx.csv')

# test_x = np.array(test_x)

test_x = np.array(test_x)
test_sqr_x = np.power(test_x, 2)
test_x_scaled = testScale(test_x, max_x, min_x)
test_sqr_x_scaled = testScale(test_sqr_x, max_sqr_x, min_sqr_x)
# test_xx_scaled = pd.DataFrame(test_x_scaled)
# test_xx_scaled.to_csv('test_scaled_xx.csv')
# print(test_x_scaled)
# add square term

# print(test_x_scaled.shape)

# print(test_sqr_x_scaled.shape)
test_x_scaled = np.concatenate((test_x_scaled,test_sqr_x_scaled), axis=1)


# add bias
test_x_scaled = np.concatenate((np.ones((test_x_scaled.shape[0],1)),test_x_scaled), axis=1)
# df_test_x = pd.DataFrame(test_x_scaled)
# df_test_x.to_csv('df_test_x.csv')



ans = []
for i in range(len(test_x_scaled)):
    ans.append(["id_"+str(i)])
    a = np.dot(w,test_x_scaled[i])
    ans[i].append(a)

filename = "sample_1_predict.csv"
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()