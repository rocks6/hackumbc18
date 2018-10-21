#import nltk
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import np_utils
import pandas as pd
from copy import deepcopy

# text = nltk.word_tokenize("")

# nltk.pos_tag(text)

# data = pd.read_csv('foobar.text', delimiter = " ")

# k = 3

# C_rx = np.random.randint(0, np.max(X)-20, size = k)

# C_ry = np.random.randint(0, np.max(X)-20, size = k)

# C = np.arrray(list(zip(C_rx, C_ry)), dtype = np.float32)

# C_prev = np.zeros(C.shape)

# clusters = np.zeroes(len(X))

# error = euclideanDistance(C, C_prev, None)

windowSize = 2
corpus = ["I really hope that this corpus is sufficient to demonstrate my bit of code okay"]
tokenizedCorpus, vocabSize = tokenize(corpus)

# for i, (x, y) in enumerate(corpusToIO(tokenizedCorpus, vocabSize, windowSize)):
	#print(i, "\n center word = ", y, "\n context words =\n", x)

#while error != 0:
#	for x in range(len(X)):
#		distances = euclideanDistance(X[x], C)
#		cluster = np.argmin(distances)
#		clusters[x] = cluster
#	C_prev = deepcopy(C)
#	for x in range(k):
#		points = [X[j] for y in range(len(X) if clusters[y] == x)]
#		C[x] = np.mean(points, axis = 0)
#	error = euclideanDistance(C, C_prev, None)

#for x in range(k):
#	points = np.array([X[y] for y in range(len(X)) if clusters[y] == k)
	
	
# def euclideanDistance(a, b, ax=1):
	# return np.linalg.norm(a - b, axis = ax)
	
def tokenize(corpus):

	tokenizer = Tokenizer()
	tokenize.fit_on_texts(corpus)
	tokenizedCorpus = tokenizer.texts_to_sequences(corpus)
	vocabSize = len(tokenizer.word_index)
	
	# vocabulary in order of appearance
	return (tokenizedCorpus, vocabSize)
	
def toCategorical(classVector, numClasses = None):
	classVector = np.array(classVector, dtype = 'int')
	# ehhh
	inputShape = classVector.shape
	if(inputShape and inputShape[-1] == 1 and len(inputShape) >= 1) :
		inputShape = tuple(inputShape[:-1])
	classVector = classVector.ravel()
	if not numClasses:
		numClasses = np.max(classVector) + 1
	n = classVector.shape[0]
	
	categorical = np.zeros((n, numClasses))
	categorical[np.arrange(n), y] = 1
	# second argument? 
	outputShape = inputShape + (numClasses,)
	categorical = np.reshape(categorical, outputShape)
	return categorical
	
def corpusToIO(tokenizedCorpus, vocabSize, windowSize):
	for words in tokenizedCorpus:
		lengthOfWords = len(words)
		for index, word in enumerate(words):
			contexts = []
			labels = []
			start = index - windowSize
			end = index + windowSize + 1
			contexts.append([words[i] - 1 for i in range(start, end) if 0 <= i < lengthOfWords and i != index])
			labels.append(word - 1)
			x = np_utils.toCategorical(contexts, vocabSize)
			y = np_utils.toCategorical(labels, vocabSize)
			
			yield (x, y.ravel())