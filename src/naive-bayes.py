from data import Dataset, Labels
from utils import evaluate
import math
import os, sys
import operator
import string


class NaiveBayes:
	def __init__(self):
		# total number of documents in the training set.
		self.n_doc_total = 0
		# total number of documents for each label/class in the trainin set.
		self.n_doc = {l: 0 for l in Labels}
		# frequency of words for each label in the trainng set.
		self.vocab = {l: {} for l in Labels}

		#total word count in bag of words for label
		self.totWords = {l: 0 for l in Labels}
		#number of total unique words
		self.difWords = 0

	def train(self, ds):
		"""
		ds: list of (id, x, y) where id corresponds to document file name,
		x is a string for the email document and y is the label.

		TODO: Loop over the dataset (ds) and update self.n_doc_total,
		self.n_doc and self.vocab.
		"""

		totDict = {}
		
		for d in ds:
			self.n_doc_total = self.n_doc_total + 1
			self.n_doc[d[2]] += 1
			for word in d[1].split():
				wordl = word.lower()
				if wordl in self.vocab[d[2]]:
					self.vocab[d[2]][wordl] += 1
				else:
					self.vocab[d[2]][wordl] = 1
		for l in Labels:
			totDict.update(self.vocab[l])
			for w in self.vocab[l]:
				self.totWords[l] += self.vocab[l][w]
		self.difWords = len(totDict)

	def predict(self, x):
		"""
		x: string of words in the document.
		
		TODO: Use self.n_doc_total, self.n_doc and self.vocab to calculate the
		prior and likelihood probabilities.
		Add the log of prior and likelihood probabilities.
		Use MAP estimation to return the Label with hight score as
		the predicted label.
		"""

		nbProb = {l: 0.0 for l in Labels}

		prior = {l: 0.0 for l in Labels}
		for l in Labels:
			prior[l] = self.n_doc[l] / self.n_doc_total
		
		xWords = x.split()
		xWordsL = []
		for xW in xWords:
			xWordsL.append(xW.lower())
		likelihood = {l: 0 for l in Labels}
		for l in Labels:
			for word in xWordsL:
				countWC = 1
				if word in self.vocab[l]:
					countWC += self.vocab[l][word]
				likelihood[l] += math.log(countWC / (self.totWords[l] + self.difWords + 1))
			likelihood[l] += math.log(prior[l])

		return max(likelihood.items(), key=operator.itemgetter(1))[0]


def main(train_split):
	nb = NaiveBayes()
	ds = Dataset(train_split).fetch()
	val_ds = Dataset('val').fetch()
	nb.train(ds)
	
	# Evaluate the trained model on training data set.
	print('-'*20 + ' TRAIN ' + '-'*20)
	evaluate(nb, ds)
	# Evaluate the trained model on validation data set.
	print('-'*20 + ' VAL ' + '-'*20)
	evaluate(nb, val_ds)

	# students should ignore this part.
	# test dataset is not public.
	# only used by the grader.
	if 'GRADING' in os.environ:
		print('\n' + '-'*20 + ' TEST ' + '-'*20)
		test_ds = Dataset('test').fetch()
		evaluate(nb, test_ds)


if __name__ == "__main__":
	train_split = 'train'
	if len(sys.argv) > 1 and sys.argv[1] == 'train_half':
		train_split = 'train_half'
	main(train_split)
