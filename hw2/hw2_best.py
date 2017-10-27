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

xTrnPath = sys.argv[3]
yTrnPath = sys.argv[4]
xTstPath = sys.argv[5]
outputPath = sys.argv[6]


def prepX(path):

    def _dfScale(df):
        # convert object to float
        # df = df.apply(pd.to_numeric)
        # mean-normalization
        # df=(df-df.mean())/df.std()
        # min-max normalization
        df=(df-df.min())/(df.max()-df.min())
        return df
    
    def _dfFeatureSelection(df, lstOfFeature):
        selected_df = df[lstOfFeature]
        return selected_df

    df = pd.read_csv(path)
    # print(len(df.columns.values))
    continuous = ['age', 'fnlwgt', 'capital_gain', 'capital_loss', 'hours_per_week']

    df_continuous = df[continuous]

    # df_continuous['age_sqr'] = df_continuous['age'] ** 2
    # df_continuous['capital_gain_sqr'] = df_continuous['capital_gain'] ** 2
    # df_continuous['hours_per_week_sqr'] = df_continuous['hours_per_week'] ** 2

    df_continuous = _dfScale(df_continuous)
    for i in range(len(continuous)):
        del df[continuous[i]]
    
    df = pd.concat([df_continuous, df], axis=1)
    # lst_feature = ['age', 'fnlwgt', 'capital_gain', 'capital_loss', 'hours_per_week', ' 1st-4th', ' 5th-6th', ' Bachelors', ' Masters', ' Private',' Husband', ' Exec-managerial', ' Craft-repair',' Prof-specialty',' England', ' White', ' Japan' , ' Mexico', ' United-States']
    # df = _dfFeatureSelection(df, lst_feature)
    # print(type(df.columns.values))
    # print(df.columns.values)
    
    # print(df)
    return df


if __name__ == "__main__":
    

    model = load_model('./krs_model_vld_0.1_rmsprop_1000_128_mimmax.h5')


    df_tst_x = prepX(xTstPath)
    pred = model.predict(df_tst_x.values)

    prediction_filename = outputPath
    ans = []
    for i in range(len(pred)):
        ans.append([str(i+1)])
        if(pred[i].item(0) > pred[i].item(1)):
            a = 1
        else:
            a = 0
        ans[i].append(a)

    # print(ans)

    text = open(prediction_filename, "w+")
    s = csv.writer(text,delimiter=',',lineterminator='\n')
    s.writerow(["id","label"])
    for i in range(len(ans)):
        s.writerow(ans[i]) 
    text.close()

