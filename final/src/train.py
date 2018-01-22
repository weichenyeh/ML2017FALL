import argparse
import numpy as np
from keras.callbacks import ModelCheckpoint
from loader import *
from wordvec import *
from model import *
from sklearn.model_selection import StratifiedKFold
modeldir = "models"


def parse_inputs():
	parser = argparse.ArgumentParser()
	parser.add_argument("--wordvec", help="Pre-trained word vector path", default="wordvecs/wordvec_dim150_mincount3")
	parser.add_argument("--batch_size", help="Batch size", default=128)
	parser.add_argument("--n_epoch", help="Number of epoch", default=5)
	parser.add_argument("--kfold", help="Number of fold", default=10)
	parser.add_argument("--num_sent", help="Number of sentences", default=2)
	return parser.parse_args()


def main():
	args = parse_inputs()

	first_sents, second_sents, labels, word_index = get_train_sents(num_sent=args.num_sent)
	wordvec = load_wordvec(path=args.wordvec)
	embedding = wordvec_to_embedding(word_index, wordvec)

	perm = np.random.permutation(len(first_sents))
	first_sents = first_sents[perm]
	second_sents = second_sents[perm]
	labels = labels[perm]

	skf = StratifiedKFold(n_splits=args.kfold)
	for cv_idx, (train_index, val_index) in enumerate(skf.split(first_sents, labels)):

		train_x1 = first_sents[train_index, :]
		train_x2 = second_sents[train_index, :]
		train_y = labels[train_index]

		val_x1 = first_sents[val_index, :]
		val_x2 = second_sents[val_index, :]
		val_y = labels[val_index]

		wordvec_name = os.path.basename(args.wordvec)
		subdir = os.path.join(modeldir, "%s_numsent%s" % (wordvec_name, args.num_sent), "cv%d" % cv_idx)
		os.makedirs(subdir, exist_ok=True)

		filepath = os.path.join(subdir, 'Model.{epoch:02d}-{acc:.4f}-{val_acc:.4f}-{val_loss:.4f}.hdf5')
		ckpointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)

		model = build_model(first_sents.shape[1], embedding)
		model.fit(
			[train_x1, train_x2],
			train_y,
			validation_data=([val_x1, val_x2], val_y),
			epochs=args.n_epoch,
			batch_size=args.batch_size,
			callbacks=[ckpointer])


if __name__ == '__main__':
	main()
