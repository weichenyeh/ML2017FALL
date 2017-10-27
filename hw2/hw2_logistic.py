import os, sys
import numpy as np
from random import shuffle
import argparse
from math import log, floor
import pandas as pd

import sys
xTrnPath = sys.argv[3]
yTrnPath = sys.argv[4]
xTstPath = sys.argv[5]
outputPath = sys.argv[6]

lamb = 0
l_rate = 0.1
batch_size = 32
epoch_num = 1000

def load_data(train_data_path, train_label_path, test_data_path):
    # X_train = pd.read_csv(train_data_path, sep=',', header=0)
    X_train = pd.read_csv(train_data_path)
    # print(X_train.columns.values)
    X_train['age_sqr'] = X_train['age'] ** 2
    X_train['capital_gain_sqr'] = X_train['capital_gain'] ** 2
    X_train['hours_per_week_sqr'] = X_train['hours_per_week'] ** 2
    
    # X_train.to_csv('trn_yoyo.csv')

    X_train = np.array(X_train.values)
    

    # print(X_train.shape)
    Y_train = pd.read_csv(train_label_path, sep=',', header=0)
    Y_train = np.array(Y_train.values)

    # X_test = pd.read_csv(test_data_path, sep=',', header=0)
    X_test = pd.read_csv(test_data_path)    
    X_test['age_sqr'] = X_test['age'] ** 2
    X_test['capital_gain_sqr'] = X_test['capital_gain'] ** 2
    X_test['hours_per_week_sqr'] = X_test['hours_per_week'] ** 2   
    # X_test.to_csv('tst_yoyo.csv')
     
    X_test = np.array(X_test.values)
    # print(X_test.shape)


    return (X_train, Y_train, X_test)

def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

def normalize(X_all, X_test):
    # Feature normalization with train and test X
    X_train_test = np.concatenate((X_all, X_test))
    mu = (sum(X_train_test) / X_train_test.shape[0])
    sigma = np.std(X_train_test, axis=0)
    mu = np.tile(mu, (X_train_test.shape[0], 1))
    sigma = np.tile(sigma, (X_train_test.shape[0], 1))
    X_train_test_normed = (X_train_test - mu) / sigma

    # Split to train, test again
    X_all = X_train_test_normed[0:X_all.shape[0]]
    X_test = X_train_test_normed[X_all.shape[0]:]
    return X_all, X_test

def split_valid_set(X_all, Y_all, percentage):
    all_data_size = len(X_all)
    valid_data_size = int(floor(all_data_size * percentage))

    X_all, Y_all = _shuffle(X_all, Y_all)

    X_train, Y_train = X_all[0:valid_data_size], Y_all[0:valid_data_size]
    X_valid, Y_valid = X_all[valid_data_size:], Y_all[valid_data_size:]

    return X_train, Y_train, X_valid, Y_valid

def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-8, 1-(1e-8))
def valid(w, b, X_valid, Y_valid):
    valid_data_size = len(X_valid)

    z = (np.dot(X_valid, np.transpose(w)) + b)
    y = sigmoid(z)
    y_ = np.around(y)
    result = (np.squeeze(Y_valid) == y_)
    # print('Validation acc = %f' % (float(result.sum()) / valid_data_size))
    return

def train(X_all, Y_all):
    # Split a 10%-validation set from the training set
    valid_set_percentage = 0.1
    X_train, Y_train, X_valid, Y_valid = split_valid_set(X_all, Y_all, valid_set_percentage)

    # Initiallize parameter, hyperparameter
    w = np.zeros((109,))
    b = np.zeros((1,))
    # l_rate = 0.1
    # batch_size = 32
    train_data_size = len(X_train)
    step_num = int(floor(train_data_size / batch_size))
    # epoch_num = 2000
    save_param_iter = 50

    # Start training
    total_loss = 0.0
    for epoch in range(1, epoch_num):
        # Do validation and parameter saving
        if (epoch) % save_param_iter == 0:
            # print('=====Saving Param at epoch %d=====' % epoch)
            # if not os.path.exists(save_dir):
            #     os.mkdir(save_dir)
            # np.savetxt(os.path.join(save_dir, 'sqr_w_lr'+str(l_rate)+'_bth'+str(batch_size)+'_epch'+str(epoch_num)+'_lamb'+str(lamb)), w)
            # np.savetxt(os.path.join(save_dir, 'sqr_b_lr'+str(l_rate)+'_bth'+str(batch_size)+'_epch'+str(epoch_num)+'_lamb'+str(lamb)), [b,])
            # print('epoch avg loss = %f' % (total_loss / (float(save_param_iter) * train_data_size)))
            total_loss = 0.0
            valid(w, b, X_valid, Y_valid)

        # Random shuffle
        X_train, Y_train = _shuffle(X_train, Y_train)

        # Train with batch
        for idx in range(step_num):
            X = X_train[idx*batch_size:(idx+1)*batch_size]
            Y = Y_train[idx*batch_size:(idx+1)*batch_size]

            z = np.dot(X, np.transpose(w)) + b
            y = sigmoid(z)

            cross_entropy = -1 * (np.dot(np.squeeze(Y), np.log(y)) + np.dot((1 - np.squeeze(Y)), np.log(1 - y)))
            total_loss += cross_entropy

            w_grad = np.mean(-1 * X * (np.squeeze(Y) - y).reshape((batch_size,1)), axis=0) + 2 * lamb * w
            b_grad = np.mean(-1 * (np.squeeze(Y) - y))

            # SGD updating parameters
            w = w - l_rate * w_grad
            b = b - l_rate * b_grad

    return (w, b)

def infer(X_test, output_dir,w, b):
    test_data_size = len(X_test)

    # Load parameters
    # print('=====Loading Param from %s=====' % save_dir)
    # w = np.loadtxt(os.path.join(save_dir, 'sqr_w_lr'+str(l_rate)+'_bth'+str(batch_size)+'_epch'+str(epoch_num)+'_lamb'+str(lamb)))
    # b = np.loadtxt(os.path.join(save_dir, 'sqr_b_lr'+str(l_rate)+'_bth'+str(batch_size)+'_epch'+str(epoch_num)+'_lamb'+str(lamb)))

    # predict
    z = (np.dot(X_test, np.transpose(w)) + b)
    y = sigmoid(z)
    y_ = np.around(y)

    # print('=====Write output to %s =====' % output_dir)
    # if not os.path.exists(output_dir):
    #     os.mkdir(output_dir)
    # output_path = os.path.join(output_dir, 'sqr_log_prediction_'+str(l_rate)+'_bth'+str(batch_size)+'_epch'+str(epoch_num)+'_lamb'+str(lamb)+'.csv')
    # output_path = os.path.join(output_dir, 'sqr_log_prediction_'+str(l_rate)+'_bth'+str(batch_size)+'_epch'+str(epoch_num)+'_lamb'+str(lamb)+'.csv')
    
    with open(output_dir, 'w') as f:
        f.write('id,label\n')
        for i, v in  enumerate(y_):
            f.write('%d,%d\n' %(i+1, v))

    return


if __name__ == '__main__':
    # Load feature and label
    X_all, Y_all, X_test = load_data(xTrnPath, yTrnPath, xTstPath)
    # Normalization
    X_all, X_test = normalize(X_all, X_test)
    # To train and infer
    w, b = train(X_all, Y_all)
    infer(X_test, outputPath, w, b)