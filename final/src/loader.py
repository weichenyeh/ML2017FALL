import os
import jieba
import _pickle as pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from wordvec import *

def get_reverse_word_index(word_index):
	reverse = {}
	for word, index in word_index.items():
		reverse[index] = word
	return reverse

# 60
def count_sentence_lengths():
	jieba.set_dictionary('jieba/dict.txt.big')

	maxlen = 0
	qlen_dist, olen_dist = {}, {}
	with open("data/testing_data.csv", "r", encoding="utf-8") as inf:
		inf.readline()
		for line in inf:
			_, q, options = line.strip().split(',')
			q = q.replace("A:", "").replace("B:", "").replace("\t", "")
			q = q.split(" ")
			
			qlen = 0
			for tmp_line in q:
				tmp_line = jieba.cut(tmp_line, cut_all=False)
				tmp_line = list(tmp_line)
				qlen += len(tmp_line)

			if qlen > maxlen:
				maxlen = qlen
				sent = q
	
			options = options.replace("A:", "").replace("B:", "")
			options = options.split("\t")
			options = [o.split(' ') for o in options]
			for o in options:
				olen = 0
				for tmp_line in o:
					tmp_line = jieba.cut(tmp_line, cut_all=False)
					tmp_line = list(tmp_line)
					olen += len(tmp_line)
				if olen > maxlen:
					maxlen = olen
					sent = o

	return maxlen


def get_train_sents(folder="data/training_data", wordvec_path="wordvecs/wordvec_dim150_mincount3", num_sent=2, offset=100):
	jieba.set_dictionary('jieba/dict.txt.big')
	wordvec = load_wordvec(path=wordvec_path)

	wordvec_name = os.path.basename(wordvec_path)
	pickle_path = os.path.join("data/pickles", "%s_numsent%d" % (wordvec_name, num_sent))
	if os.path.isfile(pickle_path):
		with open(pickle_path, 'rb') as inf:
			return pickle.load(inf)

	sents, files = [], {}
	for f in os.listdir(folder):
		f = os.path.join(folder, f)

		files[f] = []
		with open(f, 'r', encoding="utf-8") as inf:
			for line in inf:
				line = line.strip()
				line = jieba.cut(line, cut_all=False)
				sent = " ".join([word if word in wordvec else 'unk' for word in line])
				sents.append(sent)
				files[f].append(sent)

	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(sents)

	reverse_word_index = get_reverse_word_index(tokenizer.word_index)

	first_sents, second_sents, labels = [], [], []
	for f, sents in files.items():	
		for idx in range(len(sents)):
			if idx > len(sents) - num_sent * 2:
				break

			first, second, neg = "", "", ""
			for time in range(num_sent):
				first += sents[idx+time] + " "
				second += sents[idx+num_sent+time] + " "

				if idx + offset + num_sent > len(sents):
					neg += sents[idx-offset+time] + " "
				else:
					neg += sents[idx+offset+time] + " "

			first_sents.append(first)
			first_sents.append(first)
			second_sents.append(second)
			second_sents.append(neg)
			labels.append(1)
			labels.append(0)

	first_sents = tokenizer.texts_to_sequences(first_sents)
	second_sents = tokenizer.texts_to_sequences(second_sents)

	maxlen = max([len(sent) for sent in first_sents])

	first_sents = sequence.pad_sequences(first_sents, maxlen=maxlen, padding="post")
	second_sents = sequence.pad_sequences(second_sents, maxlen=maxlen, padding="post")
	labels = np.asarray(labels)

	with open(pickle_path, 'wb') as outf:
		pickle.dump([first_sents, second_sents, labels, tokenizer.word_index], outf)
	return first_sents, second_sents, labels, tokenizer.word_index

def reverse_sent(indices, reverse_word_index):
	reverse_sent = ""
	for index in indices:
		word = reverse_word_index[index]
		reverse_sent += "%s " % word
	return reverse_sent

def main():
	count_sentence_lengths()
