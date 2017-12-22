import csv
import numpy as np
import pandas as pd
from keras.models import load_model
import keras.backend as K
import sys



RATING_MEAN = 3.58171208604
RATING_STD = 1.11689766115


MODEL_FILE = './0.84679_5.h5'
MODEL_FILE_1 = './0.84828_7.h5'

TEST_FILE = sys.argv[1]
PREDICTION_FILE = sys.argv[2]

def rmse(y_true, y_predict):
    return K.sqrt(K.mean((y_predict - y_true) ** 2))

def predict(testFile, modelFile, predictionFile):
    pd_tst = pd.read_csv(testFile)
    np_test = (pd_tst.values)[:, 1:3]
    np_test = np_test.astype('float64')

    model = load_model(modelFile, custom_objects={'rmse':rmse})
    ans = model.predict(np.hsplit(np_test, 2))

    # ans *= RATING_STD
    ans += RATING_MEAN

    ans[np.isnan(ans)] = RATING_MEAN

    with open(predictionFile, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['TestDataID', 'Rating'])
        for i in range(len(ans)):
            writer.writerow([i+1, ans[i][0]])

def predictEnsemble(testFile, modelFile0, modelFile1, predictionFile):
    pd_tst = pd.read_csv(testFile)
    np_test = (pd_tst.values)[:, 1:3]
    np_test = np_test.astype('float64')

    model = load_model(modelFile0, custom_objects={'rmse':rmse})
    model_1 = load_model(modelFile1, custom_objects={'rmse':rmse})
    
    ans = model.predict(np.hsplit(np_test, 2))
    ans_1 = model_1.predict(np.hsplit(np_test, 2))
    
    ans += RATING_MEAN
    ans_1 += RATING_MEAN

    ans[np.isnan(ans)] = RATING_MEAN
    ans_1[np.isnan(ans_1)] = RATING_MEAN

    ans = (ans + ans_1) / 2

    with open(predictionFile, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['TestDataID', 'Rating'])
        for i in range(len(ans)):
            writer.writerow([i+1, ans[i][0]])

if __name__ == '__main__':
    # predict(TEST_FILE, MODEL_FILE, PREDICTION_FILE)
    predictEnsemble(TEST_FILE, MODEL_FILE, MODEL_FILE_1, PREDICTION_FILE)