import csv
import pickle
import numpy as np
import pandas as pd
import keras.backend as K
from keras.optimizers import Adam
from keras.losses import mean_squared_error
from keras.utils import np_utils
from keras.models import Model, load_model
from keras.layers import Input, Embedding, Flatten, Dot, Add, Dense, Dropout, BatchNormalization
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
import sys

# DIM = 240
LATENT_DIM = 120
NUM_OF_USER = 6041
NUM_OF_MOVIE = 3953
RATING_MEAN = 3.58171208604
RATING_STD = 1.11689766115

X_PKL = './prep/wc_x.pkl'
Y_PKL = './prep/wc_y.pkl'
LOG_FILE = './log.csv'
MODEL_FILE = './model.h5'

TEST_FILE = sys.argv[1]
PREDICTION_FILE = sys.argv[2]

def rmse(y_true, y_predict):
    return K.sqrt(K.mean((y_predict - y_true) ** 2))

def prepTrain(trainFile, prepX, prepY):
    pd_trn = pd.read_csv(trainFile)
    np_x = (pd_trn.values)[:, 1:3]
    # np_x = np_x.astype('float64')
    np_y = (pd_trn.values)[:, 3]
    np_y = np_y.astype('float64')
    # y_mean = np.mean(np_y)
    # y_std = np.std(np_y)
    # print(y_mean, y_std)
    np_y -= RATING_MEAN
    # np_y /= RATING_STD

    with open(prepX, 'wb') as handle:
        pickle.dump(np_x, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(prepY, 'wb') as handle:
        pickle.dump(np_y, handle, protocol=pickle.HIGHEST_PROTOCOL)


def build():
    user_input = Input(shape = (1,))
    movie_input = Input(shape = (1,))

    usr_vctr = Embedding(input_dim=NUM_OF_USER, output_dim=LATENT_DIM, embeddings_regularizer=l2(0.000005), input_length=1)(user_input)
    # usr_vctr = BatchNormalization()(usr_vctr)
    usr_vctr = Flatten()(usr_vctr)    
    
    mve_vctr = Embedding(input_dim=NUM_OF_MOVIE, output_dim=LATENT_DIM, embeddings_regularizer=l2(0.000005), input_length=1)(movie_input)
    # mve_vctr = BatchNormalization()(mve_vctr)
    mve_vctr = Flatten()(mve_vctr)

    usr_bias = Embedding(input_dim=NUM_OF_USER, output_dim=1, embeddings_initializer='zero')(user_input)
    usr_bias = Flatten()(usr_bias)    

    mve_bias = Embedding(input_dim=NUM_OF_MOVIE, output_dim=1, embeddings_initializer='zero')(movie_input)
    mve_bias = Flatten()(mve_bias)
    
    dot = Dot(axes = 1)([usr_vctr, mve_vctr])
    add = Add()([dot, usr_bias, mve_bias])
    model = Model([user_input, movie_input], add)
    optzr = Adam(lr = 0.0005)
    model.compile(optimizer=optzr, loss=rmse, metrics=[rmse])
    model.summary()

    return model


def train(model):
    with open(X_PKL, 'rb') as handle:
       x = pickle.load(handle)
    with open(Y_PKL, 'rb') as handle:
       y = pickle.load(handle)

    randomize = np.arange(len(x))
    np.random.shuffle(randomize)
    x = x[randomize]
    y = y[randomize]

    csvlogger = CSVLogger(LOG_FILE, append=True, separator=';')
    checkpoint = ModelCheckpoint(MODEL_FILE, monitor = 'val_rmse', save_best_only=True, verbose=1)
    earlystopping = EarlyStopping(monitor='val_rmse', patience=3, verbose=1)
    cllbks = [csvlogger, checkpoint, earlystopping]

    hist = model.fit(np.hsplit(x, 2), y,
            batch_size=512, 
            epochs=30, 
            validation_split=0.1,
            callbacks=cllbks)
    
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

if __name__ == '__main__':
    model = build()
    train(model)
    predict(TEST_FILE, MODEL_FILE, PREDICTION_FILE)