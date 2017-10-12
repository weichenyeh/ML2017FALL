import csv 
import numpy as np
from numpy.linalg import inv
import random
import math
import sys
import pandas as pd

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation

import sys

inputFile = sys.argv[1]
outputFile = sys.argv[2]

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
# read model
[max_x, min_x, max_sqr_x, min_sqr_x] = np.load('hw1_best_scaling_info.npy')

w = load_model('hw1_best_model')


test_x = []
n_row = 0
text = open(inputFile ,"r")
# text = open('/home/weichenyeh/Documents/test.csv' ,"r")
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
# test_x_scaled = np.concatenate((np.ones((test_x_scaled.shape[0],1)),test_x_scaled), axis=1)
# df_test_x = pd.DataFrame(test_x_scaled)
# df_test_x.to_csv('df_test_x.csv')


pred = w.predict(test_x_scaled)

ans = []
for i in range(len(pred)):
    ans.append(["id_"+str(i)])
    a = pred[i].item(0)
    ans[i].append(a)

# print(ans)

filename = outputFile
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()