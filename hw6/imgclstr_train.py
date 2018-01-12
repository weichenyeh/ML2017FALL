from __future__ import print_function

from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam

from sklearn.cluster import KMeans
from keras.callbacks import CSVLogger, EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau

import pandas as pd
import numpy as np

CHECKPOINT = './model.{epoch:02d}-{val_loss:.4f}.h5'
LOG ='./log.csv'
TFBOARD = './Graph'

# build model
input_img = Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

# build encoder
encoder = Model(input=input_img, output=encoded)
encoder.summary()
# # build autoencoder
# adam = Adam(lr=5e-4)
# autoencoder = Model(input=input_img, output=decoded)
# autoencoder.compile(optimizer=adam, loss='mse')
# autoencoder.summary()



# # load images
# train_num = 130000
# X = np.load('./data/image.npy')
# X = X.astype('float32') / 255.
# X = np.reshape(X, (len(X), -1))
# x_train = X[:train_num]
# x_val = X[train_num:]
# x_train.shape, x_val.shape


# cllbks = [
#     CSVLogger(LOG, append=True, separator=';'),
#     # EarlyStopping(monitor='val_loss', patience=100, verbose=1),
#     TensorBoard(log_dir=TFBOARD),
#     ModelCheckpoint(CHECKPOINT, verbose=1, save_best_only=True, mode='auto', period=1),
#     # ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=100, verbose=1)
#     ]

# # train autoencoder
# autoencoder.fit(x_train, x_train,
#                 epochs=10,
#                 batch_size=256,
#                 shuffle=True,
#                 validation_data=(x_val, x_val),
#                 callbacks=cllbks,
#                 )

# # after training, use encoder to encode image, and feed it into Kmeans
# encoded_imgs = encoder.predict(X)
# encoded_imgs = encoded_imgs.reshape(encoded_imgs.shape[0], -1)
# kmeans = KMeans(n_clusters=2, random_state=0).fit(encoded_imgs)

# # get test cases
# f = pd.read_csv('./data/test_case.csv')
# IDs, idx1, idx2 = np.array(f['ID']), np.array(f['image1_index']), np.array(f['image2_index'])



# # predict
# o = open('ta_prediction.csv', 'w')
# o.write("ID,Ans\n")
# for idx, i1, i2 in zip(IDs, idx1, idx2):
#     p1 = kmeans.labels_[i1]
#     p2 = kmeans.labels_[i2]
#     if p1 == p2:
#         pred = 1  # two images in same cluster
#     else: 
#         pred = 0  # two images not in same cluster
#     o.write("{},{}\n".format(idx, pred))
# o.close()

