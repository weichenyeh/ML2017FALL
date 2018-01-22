import sys
import jieba
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from loader import *


def write_answers_to_file(answers, out_path="results/test.csv"):
	with open(out_path, 'w') as outf:
		outf.write("id,ans\n")
		for idx, ans in enumerate(answers):
			outf.write("%d," % (idx + 1))
			for prob in ans:
				outf.write("%.4f " % prob)
			outf.write("\n")


def get_test_sents(tokenizer, maxlen, path, wordvec):
	jieba.set_dictionary("jieba/dict.txt.big")

	questions, options = [], [[] for _ in range(6)]
	with open(path, "r", encoding="utf-8") as inf:
		header = inf.readline()
		for line in inf:
			_, tmp_q, tmp_options = line.strip().split(',')
			tmp_q = tmp_q.replace("A:", "").replace("B:", "").replace("\t", " ").split(' ')
			q = ""
			for sent in tmp_q:
				sent = jieba.cut(sent, cut_all=False)
				for word in sent:
					if word not in wordvec:
						q += "unk "
					else:
						q += "%s " % word

			questions.append(q)
	
			tmp_options = tmp_options.replace("A:", "").replace("B:", "").split('\t')

			for idx, tmp_o in enumerate(tmp_options):
				o = ""
				for sent in tmp_o.split(" "):
					sent = jieba.cut(sent, cut_all=False)
					for word in sent:
						if word not in wordvec:
							o += "unk "
						else:
							o += "%s " % word
				options[idx].append(o)

		'''
		print(questions[:100])
		for idx in range(len(options)):
			print(options[idx][:100])
		input("")
		'''

		questions = tokenizer.texts_to_sequences(questions)
		questions = sequence.pad_sequences(questions, maxlen=maxlen, padding='post')

		for idx in range(len(options)):
			options[idx] = tokenizer.texts_to_sequences(options[idx])
			for oid, option in enumerate(options[idx]):
				if len(option) > maxlen:
					options[idx][oid] = option[:maxlen]
			options[idx] = sequence.pad_sequences(options[idx], maxlen=maxlen, padding='post')

		return questions, options


def main():
	model_path = sys.argv[1]
	test_path = sys.argv[2]
	model = load_model(model_path)
	
	wordvec_path = os.path.basename(os.path.dirname(model_path)).rsplit("_", 1)[0]
	wordvec = load_wordvec(path=os.path.join("wordvecs", "wordvec_dim150_mincount3"))
	#num_sent = int(os.path.dirname(model_path).split("numsent")[1][0])
	num_sent = 2
	
	#first_sents, _, _, word_index = get_train_sents(num_sent=num_sent)
	#maxlen = first_sents.shape[1]
	maxlen = 21

	#tokenizer = Tokenizer()
	#tokenizer.word_index = word_index
	import _pickle as pk
	with open("tokenizer.pickle", "rb") as handle:
		tokenizer = pickle.load(handle)
	word_index = tokenizer.word_index

	questions, options = get_test_sents(tokenizer, maxlen, test_path, wordvec)
	
	losses = []
	for option in options:
		loss = model.predict([questions, option], verbose=1)
		losses.append(loss[:, 0])
	losses = np.asarray(losses)
	ans = losses.T
	#ans = np.argmax(losses, axis=0)

	name, ext = os.path.splitext(os.path.basename(model_path))
	result_path = os.path.join("results", name + '.csv')
	write_answers_to_file(ans, out_path=result_path)


if __name__ == '__main__':
	main()
