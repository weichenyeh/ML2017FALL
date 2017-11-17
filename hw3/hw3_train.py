import numpy as np
import pandas as pd
import keras
import os
import csv
# import scipy.misc
# import scipy
# from scipy import ndimage
# from matplotlib import pyplot as plt

import tensorflow as tf
from keras.layers import Dense, Convolution2D, UpSampling2D, MaxPooling2D, ZeroPadding2D, Flatten, Dropout, Reshape, Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.models import Sequential, load_model
from keras.utils import np_utils
from keras.optimizers import SGD, Adam, Adadelta
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.callbacks import CSVLogger, EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau


from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D

import sys


def load_prep_data(trainPath):
    x = pd.read_csv(trainPath)
    data = x.values

    y = data[:, 0]
    pixels = data[:, 1]
    X = np.zeros((pixels.shape[0], 48*48))

    for ix in range(X.shape[0]):
        p = pixels[ix].split(' ')
        for iy in range(X.shape[1]):
            X[ix, iy] = int(p[iy])
    # x = np.load('./facial_data_X.npy')
    # y = np.load('./facial_labels.npy')
    # np.save('facial_data_X', X)
    # np.save('facial_labels', y)
    # x -= np.mean(x, axis=0)
    # x /= np.std(x, axis=0)
    X /= 255
    X_train = X[0:28000,:]
    X_crossval = X[28000:28710,:]
    X_train = X_train.reshape((X_train.shape[0], 48, 48, 1))
    X_crossval = X_crossval.reshape((X_crossval.shape[0], 48, 48, 1))
    
    y_ = np_utils.to_categorical(y, num_classes=7)
    Y_train = y_[:28000]
    Y_crossval = y_[28000:28710]
    return X_train, X_crossval, Y_train, Y_crossval


def build_model_bn():
    img_rows, img_cols = 48, 48
    model = Sequential()
    model.add(Convolution2D(64, (5, 5), input_shape=(img_rows,img_cols, 1), padding='valid'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(2, 2), data_format='channels_last'))
    model.add(MaxPooling2D(pool_size=(5, 5),strides=(2, 2)))
    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), data_format='channels_last'))


    model.add(Convolution2D(64, (3, 3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), data_format='channels_last'))

    model.add(Convolution2D(64, (3, 3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))
    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), data_format='channels_last'))

    model.add(Convolution2D(128, (3, 3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), data_format='channels_last'))

    model.add(Convolution2D(128, (3, 3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
    model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))

    model.add(Flatten())

    model.add(Dense(1024))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))    
    model.add(Dropout(0.2))
    model.add(Dense(1024))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))    
    model.add(Dropout(0.2))
    model.add(Dense(7))
    model.add(Activation('softmax'))
    ada = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    model.summary()
    return model



def train(trainpth):
    model = build_model_bn()
    X_train, X_crossval, Y_train, Y_crossval = load_prep_data(trainpth)

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=40,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images



    datagen.fit(X_train)

    ModelCheckPoint_filename='b_model.{epoch:02d}-{val_acc:.4f}.h5'
    log_filename='b_log.csv'
    cllbks = [
            CSVLogger(log_filename, append=True, separator=';'),
            # EarlyStopping(monitor='val_loss', patience=100, verbose=0),
            # TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True),
            TensorBoard(log_dir='./Graph'),
            ModelCheckpoint(ModelCheckPoint_filename, monitor='val_loss', verbose=1, save_best_only=False, mode='auto', period=20),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=100, verbose=1)
                ]

    model.fit_generator(datagen.flow(X_train, Y_train,batch_size=128),
                        epochs=200,
                        validation_data=(X_crossval, Y_crossval),
                        samples_per_epoch=X_train.shape[0],
                        verbose=1,
                        callbacks=cllbks)

    model.save("b_model_last.h5")
    return model


if __name__=='__main__':
    trnPath = sys.argv[1]
    model = train(trnPath)
