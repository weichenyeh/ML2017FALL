# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.layers.recurrent import GRU, LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# from keras.utils import to_categorical
# from keras.layers.embeddings import Embedding
# from keras.callbacks import CSVLogger, EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
# from keras.preprocessing import text
from gensim.models import Word2Vec as w2v
from keras.models import load_model
import csv
import numpy as np
import pickle
import sys

TEST_DATA_FILE = sys.argv[1]
PREDICT_FILE = sys.argv[2]

# TRAIN_DATA_FILE = './input_data/training_label.txt'
# TRAIN_DATA_NOLABEL_FILE = './input_data/training_nolabel.txt'
MODEL_FILE = './ke4_model.02-0.8162.h5'
WORD_VEC_DIR ='./w2v_100d_nonull_deli'
MAX_WORD_NUM = 30

# def load_data_set():
#     X = []
#     Y = []
#     with open(TRAIN_DATA_FILE) as input_file:
#         input_data = input_file.readlines()

#         for i in range(len(input_data)):
#             input_data[i] = input_data[i].split('+++$+++')
#             Y.append(input_data[i][0])
#             X.append(input_data[i][1].rstrip())

#     return X,Y

def load_test_data_set():
    X = []
    with open(TEST_DATA_FILE) as input_file:
        # i=0 is header thus test size+1
        # input_data = input_file.readlines()[0:(tst_size+1)]
        # input_data = input_file.readlines()[0:(50+1)]
        input_data = input_file.readlines()
        for i in range(len(input_data)):

            # i=0 is header thus skipped
            if(i > 0):
                X.append((input_data[i].split(',', 1))[1].rstrip())
    return X


# def load_nolabel_data_set():
#     X = []
#     with open(TRAIN_DATA_NOLABEL_FILE) as input_file:
#         input_data = input_file.readlines()
#         for i in range(len(input_data)):
#             X.append(input_data[i].rstrip())
#             # print(input_data[i].rstrip())
#             # X.append((input_data[i].split(',', 1))[1].rstrip())
#     return X



def outputAns(pred):
    # print(len(pred), pred[0])
    # print(pred)
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





# x_raw, y_raw = load_data_set()
# x_nolabel = load_nolabel_data_set()
x_test = load_test_data_set()
# tokenizer = Tokenizer(filters='!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
# all_corpus = x_raw + x_nolabel + x_test
# tokenizer.fit_on_texts(all_corpus)
with open('ke_tokenizer_deli.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
print('tokenizer loaded')


test_sequences = tokenizer.texts_to_sequences(x_test)


w2v_model = w2v.load(WORD_VEC_DIR)
word2index = tokenizer.word_index
# print(word2index['i'])            
index2word = {v: k for k, v in word2index.items()}

for b in range(len(test_sequences)):
    for i in range(len(test_sequences[b])):
        try:
            w2v_model.wv[index2word[test_sequences[b][i]]]
        except KeyError:
            test_sequences[b][i] = 1




test_sequences = pad_sequences(test_sequences, maxlen=MAX_WORD_NUM, value=1)
print('text to padded seq')
model = load_model(MODEL_FILE)
tst_predicted_out = model.predict(test_sequences)

print('prediction done')
outputAns(tst_predicted_out)


