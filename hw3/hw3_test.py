import numpy as np
import pandas as pd
import csv

import sys
xTstPath = sys.argv[1]
outputPath = sys.argv[2]

tst_x = pd.read_csv(xTstPath)
tst_x = tst_x['feature']
pixels = tst_x.values
X = np.zeros((pixels.shape[0], 48*48))
for ix in range(X.shape[0]):
    p = pixels[ix].split(' ')
    for iy in range(X.shape[1]):
        X[ix, iy] = int(p[iy])

# X -= np.mean(X, axis=0)
# X /= np.std(X, axis=0)
X /= 255

X = X.reshape((X.shape[0], 48, 48, 1))


# print(X[0:10].shape)
from keras.models import load_model
model = load_model('./b_model.179-0.6742.h5')
pred = model.predict(X)

# print(type(pred))
# print(len(pred))
# print(pred[0])
# print(pred)

ans = []
for i in range(len(pred)):
    ans.append([str(i)])
    a = np.argmax(pred[i])
    ans[i].append(a)

# print(ans)

text = open(outputPath, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()