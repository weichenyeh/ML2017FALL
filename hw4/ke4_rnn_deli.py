from keras.models import Sequential
from keras.layers import Dense, Dropout, Bidirectional
from keras.layers.recurrent import GRU, LSTM
from keras.preprocessing import text
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers.embeddings import Embedding
from keras.callbacks import CSVLogger, EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing import text
import pickle
from gensim.models import Word2Vec as w2v
from keras.optimizers import Adam
from keras.layers import Activation
from keras import regularizers
from keras.layers.normalization import BatchNormalization

import csv
import sys
import numpy as np

# TRAIN_DATA_FILE = './input_data/training_label.txt'
# TRAIN_DATA_NOLABEL_FILE = './input_data/training_nolabel.txt'

TRAIN_DATA_FILE = sys.argv[1]
TRAIN_DATA_NOLABEL_FILE = sys.argv[2]

TEST_DATA_FILE = './input_data/testing_data.txt'
MAX_WORD_NUM = 30
# WORD_VEC_DIR = './word_vec_all_null' # word vector model
# WORD_VEC_DIR = './w2v_200d' # word vector model
WORD_VEC_DIR = './w2v_100d_nonull_deli'
# WORD_VEC_DIR = 'word_vec'
ModelCheckPoint_filename='ke4_delimodel.{epoch:02d}-{val_acc:.4f}.h5'
log_filename='ke4_delilog.csv'
PREDICT_FILE = './ke4_delipredict.csv'

def load_data_set():
    X = []
    Y = []

    with open(TRAIN_DATA_FILE) as input_file:
        input_data = input_file.readlines()

        for i in range(len(input_data)):
            input_data[i] = input_data[i].split('+++$+++')
            Y.append(input_data[i][0])
            X.append(input_data[i][1].rstrip())

    return X,Y

# def load_test_data_set():
#     X = []
#     with open(TEST_DATA_FILE) as input_file:
#         # i=0 is header thus test size+1
#         # input_data = input_file.readlines()[0:(tst_size+1)]
#         # input_data = input_file.readlines()[0:(50+1)]
#         input_data = input_file.readlines()
#         for i in range(len(input_data)):

#             # i=0 is header thus skipped
#             if(i > 0):
#                 X.append((input_data[i].split(',', 1))[1].rstrip())
#     return X


def load_nolabel_data_set():
    X = []
    with open(TRAIN_DATA_NOLABEL_FILE) as input_file:
        input_data = input_file.readlines()
        for i in range(len(input_data)):
            X.append(input_data[i].rstrip())
            # print(input_data[i].rstrip())
            # X.append((input_data[i].split(',', 1))[1].rstrip())
    return X

# def load_test_data_set():
#     X = []
#     with open(TEST_DATA_FILE) as input_file:
#         # i=0 is header thus test size+1
#         # input_data = input_file.readlines()[0:(tst_size+1)]
#         # input_data = input_file.readlines()[0:(50+1)]
#         input_data = input_file.readlines()
#         for i in range(len(input_data)):

#             # i=0 is header thus skipped
#             if(i > 0):
#                 X.append((input_data[i].split(',', 1))[1].rstrip())
#     return X


def outputAns(pred):
    # print(len(pred), pred[0])
    print(pred)
    ans = []
    for i in range(len(pred)):
        ans.append([str(i)])
        a = np.argmax(pred[i])
        ans[i].append(a)

    # print(ans)

    text = open(PREDICT_FILE, "w+")
    s = csv.writer(text,delimiter=',',lineterminator='\n')
    s.writerow(["id","label"])
    for i in range(len(ans)):
        s.writerow(ans[i]) 
    text.close()

    
w2v_model = w2v.load(WORD_VEC_DIR)
# print(dir(w2v_model.wv))
# print(w2v_model.wv.index2word)
# reverse = {word:idx for idx,word in enumerate(w2v_model.wv.index2word, 1)}
weight = w2v_model.wv.syn0
num_word = weight.shape[0]
emb_dim = weight.shape[1]
# print(num_word, emb_dim)
# print(w2v_model.wv['0'])
# print(weight[55290])
# print(weight[0]==w2v_model.wv['i'])



# # load train data

x_raw, y_raw = load_data_set()
# x_nolabel = load_nolabel_data_set()
# x_test = load_test_data_set()


# tokenizer = Tokenizer(filters='')
# all_corpus = x_raw + x_nolabel + x_test
# tokenizer.fit_on_texts(all_corpus)
# with open('ke_tokenizer_deli.pickle', 'wb') as handle:
#    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
# print('tokenizer built')

with open('ke_tokenizer_deli.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
print('tokenizer loaded')


word2index = tokenizer.word_index
# print(word2index['i'])            
index2word = {v: k for k, v in word2index.items()}
# print('index of i: ', word2index['i'])
# print(index2word[1])
# print(len(word2index))
# print(len(index2word))
# for i in range(len(word2index)):
#     (word2index[index2word[i+1]])
    # print(word2index[index2word[i]])

embedding_matrix = np.ones((num_word,emb_dim))
for word, i in word2index.items():
    if i < num_word:
        try:
            embedding_matrix[i] = w2v_model.wv[word]
        except:
            embedding_matrix[i] = w2v_model.wv['i']


# build keras model
model = Sequential()
model.add(Embedding(num_word, emb_dim, weights=[embedding_matrix]))
model.add(Bidirectional(LSTM(256, return_sequences=True)))
# model.add(Bidirectional(LSTM(256, return_sequences=True)))
model.add(Bidirectional(LSTM(256)))
model.add(Dropout(0.5))

#model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
#model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

#model.add(BatchNormalization())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
#model.add(BatchNormalization())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

#model.add(BatchNormalization())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
#model.add(BatchNormalization())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))

#model.add(BatchNormalization())
model.add(Dense(2, activation='softmax'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()
print ('model built')








train_sequences = tokenizer.texts_to_sequences(x_raw)
for b in range(len(train_sequences)):
    for i in range(len(train_sequences[b])):
        try:
            w2v_model.wv[index2word[train_sequences[b][i]]]
        except KeyError:
            train_sequences[b][i] = 1

train_sequences = pad_sequences(train_sequences, maxlen=MAX_WORD_NUM, value=1)
# print(train_sequences)

y = to_categorical(y_raw)



cllbks = [
    CSVLogger(log_filename, append=True, separator=';'),
    # EarlyStopping(monitor='val_loss', patience=100, verbose=0),
    TensorBoard(log_dir='./Graph'),
    # ModelCheckpoint(ModelCheckPoint_filename, monitor='val_acc', verbose=1, save_best_only=False, mode='auto', period=2),
    # ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=100, verbose=1)
        ]
model.fit(train_sequences, y, batch_size=128, epochs=2, callbacks=cllbks, validation_split=0.05)
model.save('ke4_model.02-0.8162.h5')
# test_sequences = tokenizer.texts_to_sequences(x_test)

# for b in range(len(test_sequences)):
#     for i in range(len(test_sequences[b])):
#         try:
#             w2v_model.wv[index2word[test_sequences[b][i]]]
#         except KeyError:
#             test_sequences[b][i] = 1

# test_sequences = pad_sequences(test_sequences, maxlen=MAX_WORD_NUM, value=1)
# tst_predicted_out = model.predict(test_sequences)
# outputAns(tst_predicted_out)

