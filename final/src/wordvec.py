import os
import jieba
import numpy as np
from gensim.models import word2vec as w2v
from keras.utils.np_utils import to_categorical

wordvec_dir = "wordvecs"


def wordvec_to_embedding(word_index, wordvec):
	num_words = len(word_index) + 1

	notin_list = []
	embeddings = None
	for word, idx in word_index.items():
		if word not in wordvec:
			notin_list.append(word)
			vec = wordvec['unk']
		else:
			vec = wordvec[word]
		if embeddings is None:
			embeddings = np.zeros((num_words, vec.shape[0]))
		embeddings[idx] = vec

	return embeddings


def load_wordvec(path=os.path.join(wordvec_dir, "wordvec_100")):
	return w2v.Word2Vec.load(path)


def train_wordvec(sentences, dim=150, min_count=3):
	print("Training word vectors")
	word_vec = w2v.Word2Vec(sentences, size=dim, min_count=min_count, window=7, sg=1, iter=30)
	word_vec.save(os.path.join(wordvec_dir, "wordvec_dim%d_mincount%d" % (dim, min_count)))


def get_all_sentences(min_count=3):
	jieba.set_dictionary('jieba/dict.txt.big')

	words, count = [], {}
	with open("data/training_data/all_train.txt", "r", encoding="utf-8") as inf:
		for line in inf:
			line = line.strip()
			#line = "".join([c for c in line if u'\u4e00' <= c <= u'\u9fff'])
			line = jieba.cut(line, cut_all=False)

			sent = []
			for c in line:
				count[c] = 1 if c not in count else count[c] + 1
				sent.append(c)
			words.append(sent)

	# Replace words appearing less than min_count as unk
	sentences = []
	for sent in words:
		converted = []
		for word in sent:
			word = "unk" if count[word] < min_count else word
			converted.append(word)
		sentences.append(converted)
	return sentences


def main():
	sentences = get_all_sentences()
	word_vec = train_wordvec(sentences, min_count=3)


if __name__ == '__main__':
	main()
