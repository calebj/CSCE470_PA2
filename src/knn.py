from data import Dataset, Labels
from utils import evaluate
import os, sys

import collections
import math

K = 5

class KNN:
	def __init__(self):
		# bag of words document vectors
		self.bow = []

	def train(self, ds):
		"""
		ds: list of (id, x, y) where id corresponds to document file name,
		x is a string for the email document and y is the label.

		TODO: Save all the documents in the train dataset (ds) in self.bow.
		You need to transform the documents into vector space before saving
		in self.bow.
		"""
		for d in ds:
			docVocab = {}
			for word in d[1].split():
				wordl = word.lower()
				if wordl in docVocab:
					docVocab[wordl] += 1
				else:
					docVocab[wordl] = 1
			docLen = 0
			for dWord in docVocab:
				docLen += (docVocab[dWord] ** 2)
			self.bow.append((d[2], math.sqrt(docLen), docVocab))

	def predict(self, x):
		"""
		x: string of words in the document.

		TODO: Predict class for x.
		1. Transform x to vector space.
		2. Find k nearest neighbors.
		3. Return the class which is most common in the neighbors.
		"""

		xVocab = {}
		similarity = []
		xLen = 0

		#create x vector space
		for word in x.split():
			wordl = word.lower()
			if wordl in xVocab:
				xVocab[wordl] += 1
			else:
				xVocab[wordl] = 1
		
		#get length of x vector space
		for xWord in xVocab:
			xLen += (xVocab[xWord] ** 2)
		xLen = math.sqrt(xLen)

		#calculate similarity
		for doc in self.bow:
			ab = 0
			for xWord in xVocab:
				if xWord in doc[2]:
					ab += xVocab[xWord] * doc[2][xWord]
			similarity.append((doc[0], (ab / (doc[1] * xLen))))

		#sort the similarity
		similarity.sort(key=lambda tup: tup[1])
		similarity.reverse()

		#get the K+ most similar values
		index = 0
		found = False
		counts = {l: 0 for l in Labels}
		maxCat = 0

		while not found:
			counts[similarity[index][0]] += 1
			if index >= K - 1:
				found = True
				max = 0
				for l in Labels:
					if counts[l] > max:
						max = counts[l]
						maxCat = l
						found = True
					elif counts[l] == max:
						found = False
			index += 1

		return maxCat

def main(train_split):
	knn = KNN()
	ds = Dataset(train_split).fetch()
	val_ds = Dataset('val').fetch()
	knn.train(ds)

	# Evaluate the trained model on training data set.
	print('-'*20 + ' TRAIN ' + '-'*20)
	evaluate(knn, ds)
	# Evaluate the trained model on validation data set.
	print('-'*20 + ' VAL ' + '-'*20)
	evaluate(knn, val_ds)

	# students should ignore this part.
	# test dataset is not public.
	# only used by the grader.
	if 'GRADING' in os.environ:
		print('\n' + '-'*20 + ' TEST ' + '-'*20)
		test_ds = Dataset('test').fetch()
		evaluate(knn, test_ds)


if __name__ == "__main__":
	train_split = 'train'
	if len(sys.argv) > 1 and sys.argv[1] == 'train_half':
		train_split = 'train_half'
	main(train_split)
