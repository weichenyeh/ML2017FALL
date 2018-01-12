from __future__ import print_function

from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras.optimizers import Adam

from sklearn.cluster import KMeans
from keras.callbacks import CSVLogger, EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau

import pandas as pd
import numpy as np
import sys

MODEL = './model.14900-0.0088.h5'
DATASET = sys.argv[1]
TESTCASE = sys.argv[2]
PREDICTION = sys.argv[3]
# print(DATASET)
# print(TESTCASE)
# print(PREDICTION)


autoencoder = load_model(MODEL)

input_img = Input(shape=(784,))

encoded = autoencoder.layers[1](input_img)
encoded = autoencoder.layers[2](encoded)
encoded = autoencoder.layers[3](encoded)
# build encoder
encoder = Model(input=input_img, output=encoded)
# encoder.summary()



# load images
X = np.load(DATASET)
X = X.astype('float32') / 255.
X = np.reshape(X, (len(X), -1))


# after training, use encoder to encode image, and feed it into Kmeans
encoded_imgs = encoder.predict(X)
encoded_imgs = encoded_imgs.reshape(encoded_imgs.shape[0], -1)
kmeans = KMeans(n_clusters=2, random_state=0).fit(encoded_imgs)

# get test cases
f = pd.read_csv(TESTCASE)
IDs, idx1, idx2 = np.array(f['ID']), np.array(f['image1_index']), np.array(f['image2_index'])



# predict
o = open(PREDICTION, 'w')
o.write("ID,Ans\n")
for idx, i1, i2 in zip(IDs, idx1, idx2):
    p1 = kmeans.labels_[i1]
    p2 = kmeans.labels_[i2]
    if p1 == p2:
        pred = 1  # two images in same cluster
    else: 
        pred = 0  # two images not in same cluster
    o.write("{},{}\n".format(idx, pred))
o.close()

