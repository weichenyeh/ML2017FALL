import os
import sys
import numpy as np
from collections import Counter

def read_labels(in_csv):
	with open(in_csv, "r") as inf:
		header = inf.readline()
		labels = [line.strip().split(',')[1].split(' ') for line in inf]

	return header, labels


def ensemble(labels):
	'''
	labels = np.asarray(labels, dtype=np.float32)
	answers = np.sum(labels, axis=0)
	results = np.argmax(answers, axis=1)
	print(results)
	'''

	labels = np.asarray(labels, dtype=np.float32)
	answers = np.sum(labels, axis=0)

	top3 = (-answers).argsort()[:, :3]

	all_ans = None
	for model in labels:
		top3_ans = []
		for q, index in zip(model, top3):
			q = q[index]
			q = np.exp(q) / np.sum(np.exp(q))
			top3_ans.append(q)
		top3_ans = np.asarray(top3_ans)

		if all_ans is None:
			all_ans = top3_ans
		else:
		 	all_ans += top3_ans

	all_ans = np.asarray(all_ans)
	first_ans = np.argmax(all_ans, axis=1)

	results = []
	for ans, idx in zip(first_ans, top3):
		results.append(idx[ans])

	results = np.asarray(results)
	return results


def main():
	out_csv = sys.argv[-1]

	with open(out_csv, "w") as outf:
		labels, losses = [], []
		for name in sys.argv[1:-1]:
			csv = name
			loss = float(os.path.splitext(csv)[0].split('-')[-1])
			losses.append(loss)

			header, lbl = read_labels(csv)
			lbl = np.asarray(lbl, dtype=np.float32)
			labels.append(lbl)

		labels = ensemble(labels)

		outf.write(header)
		for idx, label in enumerate(labels):
			outf.write(str(idx+1) + ',' + str(label) + "\n")


if __name__ == '__main__':
	main()
