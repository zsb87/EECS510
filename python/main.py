#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
=======================================================================================
Twitter United Airlines Multivariate Sentiment Analysis Based on Support Vector Machine
=======================================================================================
@author: Yuanhui Yang
@email: yuanhui.yang@u.northwestern.edu
=======================================================================================
"""
print(__doc__)

from types import *
import numpy as np
from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from math import sqrt
import matplotlib.pyplot as plt
# import gensim, logging
import pandas as pd


class Data:
	def __init__(self, csvfilepath):
		self.feature()
		self.dataType()
		self.raw_table = pd.read_csv(csvfilepath, usecols=self.feature, skip_blank_lines=True, dtype=self.data_type, keep_default_na=True)
		self.dropNaN()
		self.setOutput()
		self.setInput()
		self.preprocess()
		self.svr_linear()
		self.svr_rbf()
		self.svr_sigmoid()

	def printRawTable(self):
		print self.raw_table
		return self.raw_table

	def feature(self):
		self.feature = ['airline_sentiment', 'airline_sentiment_confidence', 'airline', 'text', 'tweet_created', 'tweet_location', 'user_timezone']
		return self.feature
	def dataType(self):
		self.data_type = {'airline_sentiment': str, 'airline_sentiment_confidence': np.float64, 'airline': str, 'text': str, 'tweet_created': str, 'tweet_location': str, 'user_timezone': str}
	def dropNaN(self):
		self.raw_table = self.raw_table.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
		self.raw_table = self.raw_table.dropna(axis=1, how='any', thresh=None, subset=None, inplace=False)
		return self.raw_table
	def setOutput(self):
		airline_sentiment = np.array(self.raw_table['airline_sentiment'])
		airline_sentiment_confidence = np.array(self.raw_table['airline_sentiment_confidence'])
		self.length_of_sequence = len(airline_sentiment)
		self.output = np.zeros(self.length_of_sequence)
		for idx in range(self.length_of_sequence):
			if airline_sentiment[idx] == 'neutral':
				self.output[idx] = np.float64(0.5 * airline_sentiment_confidence[idx])
			elif airline_sentiment[idx] == 'positive':
				self.output[idx] = np.float64(1.0 * airline_sentiment_confidence[idx])
			else:
				self.output[idx] = np.float64(0.0)
		pd.DataFrame(self.output).to_csv('../data/Output.csv', sep=',')
		return self.output

	def setInput(self):
		self.text2Vector()
		self.airline2Vector()
		self.tweetCreated2Vector()
		self.tweetlocation2Vector()
		self.userTimezone2Vector()
		self.input = np.column_stack((self.airline2Vector, self.text2Vector, self.tweetCreated2Vector, self.tweetlocation2Vector, self.userTimezone2Vector))
		pd.DataFrame(self.input).to_csv('../data/Input.csv', sep=',')
		return self.input

	def preprocess(self):
		self.length_of_prediction_sequence = int(0.2 * len(self.input))
		self.input_scaler = preprocessing.StandardScaler()
		self.input_transform = self.input_scaler.fit_transform(self.input)
		self.output_scaler = preprocessing.StandardScaler()
		self.output_transform = self.output_scaler.fit_transform(self.output)
		self.input_transform_train = self.input_transform[:(self.length_of_sequence - self.length_of_prediction_sequence)]
		self.output_transform_train =  self.output_transform[:(self.length_of_sequence - self.length_of_prediction_sequence)]
		self.input_transform_test = self.input_transform[(self.length_of_sequence - self.length_of_prediction_sequence):]
		self.output_test = self.output[(self.length_of_sequence - self.length_of_prediction_sequence):]
		# print len(self.input_transform_test)
		# print len(self.input_transform_train)

	def svr_linear(self):
		self.svr_linear = GridSearchCV(SVR(kernel='linear', gamma=0.1), cv=10, param_grid={'C': np.logspace(-10.0, 10.0, num=5, base=2.0), 'gamma': np.logspace(-10.0, 10.0, num=5, base=2.0)})
		# self.svr_linear = SVR(kernel='linear', gamma=0.1)
		self.svr_linear.fit(self.input_transform_train, self.output_transform_train)
		self.output_transform_predict_linear = self.svr_linear.predict(self.input_transform_test)
		self.output_predict_linear = self.output_scaler.inverse_transform(self.output_transform_predict_linear.reshape((self.length_of_prediction_sequence, 1)))
		self.square_root_of_mean_squared_error_linear = sqrt(mean_squared_error(self.output_test, self.output_predict_linear))
		plt.figure()
		x = np.arange(0, self.length_of_prediction_sequence)
		plt.plot(x, self.output_test, 'ro-', label='Actual')
		plt.plot(x, self.output_predict_linear, 'bo-', label='Predicted')
		plt.title('linear-SVR: RMSE = %.3f' %self.square_root_of_mean_squared_error_linear)
		plt.legend()
		plt.grid(True)
		plt.savefig('../figure/linear-SVR.eps', format='eps', dpi=1000)
		plt.savefig('../figure/linear-SVR.png', format='png', dpi=1000)
		# plt.show()

	def svr_rbf(self):
		self.svr_rbf = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=10, param_grid={'C': np.logspace(-10.0, 10.0, num=5, base=2.0), 'gamma': np.logspace(-10.0, 10.0, num=5, base=2.0)})
		# self.svr_rbf = SVR(kernel='rbf', gamma=0.1)
		self.svr_rbf.fit(self.input_transform_train, self.output_transform_train)
		self.output_transform_predict_rbf = self.svr_rbf.predict(self.input_transform_test)
		self.output_predict_rbf = self.output_scaler.inverse_transform(self.output_transform_predict_rbf.reshape((self.length_of_prediction_sequence, 1)))
		self.square_root_of_mean_squared_error_rbf = sqrt(mean_squared_error(self.output_test, self.output_predict_rbf))
		plt.figure()
		x = np.arange(0, self.length_of_prediction_sequence)
		plt.plot(x, self.output_test, 'ro-', label='Actual')
		plt.plot(x, self.output_predict_rbf, 'bo-', label='Predicted')
		plt.title('rbf-SVR: RMSE = %.3f' %self.square_root_of_mean_squared_error_rbf)
		plt.legend()
		plt.grid(True)
		plt.savefig('../figure/rbf-SVR.eps', format='eps', dpi=1000)
		plt.savefig('../figure/rbf-SVR.png', format='png', dpi=1000)
		# plt.show()

	def svr_sigmoid(self):
		self.svr_sigmoid = GridSearchCV(SVR(kernel='sigmoid', gamma=0.1), cv=10, param_grid={'C': np.logspace(-10.0, 10.0, num=5, base=2.0), 'gamma': np.logspace(-10.0, 10.0, num=5, base=2.0)})
		# self.svr_sigmoid = SVR(kernel='sigmoid', gamma=0.1)
		self.svr_sigmoid.fit(self.input_transform_train, self.output_transform_train)
		self.output_transform_predict_sigmoid = self.svr_sigmoid.predict(self.input_transform_test)
		self.output_predict_sigmoid = self.output_scaler.inverse_transform(self.output_transform_predict_sigmoid.reshape((self.length_of_prediction_sequence, 1)))
		self.square_root_of_mean_squared_error_sigmoid = sqrt(mean_squared_error(self.output_test, self.output_predict_sigmoid))
		plt.figure()
		x = np.arange(0, self.length_of_prediction_sequence)
		plt.plot(x, self.output_test, 'ro-', label='Actual')
		plt.plot(x, self.output_predict_sigmoid, 'bo-', label='Predicted')
		plt.title('sigmoid-SVR: RMSE = %.3f' %self.square_root_of_mean_squared_error_sigmoid)
		plt.legend()
		plt.grid(True)
		plt.savefig('../figure/sigmoid-SVR.eps', format='eps', dpi=1000)
		plt.savefig('../figure/sigmoid-SVR.png', format='png', dpi=1000)
		# plt.show()

	def word2Vector(self, item, num_feature):
		corpus = np.array(self.raw_table[item])
		hashingVectorizer = HashingVectorizer(decode_error='ignore', n_features=num_feature, non_negative=False)
		word2Vector = hashingVectorizer.fit_transform(corpus).toarray()
		return word2Vector
		# pd.DataFrame(word2Vector).to_csv('../data/result.csv', sep=',', encoding='utf-8',)
		# print self.vector.ravel()
		# print type(corpus[0])
		# print corpus
		# self.vectorizer = TfidfVectorizer(min_df=1, max_df=1.0,  stop_words='english', max_features=10, norm='l2', sublinear_tf=True)
		# print self.vectorizer
		# vec = self.vectorizer.fit_transform(corpus).toarray()
		# print vec.toarray()
	# def setInput(self):
	# def airline2Vector(self):
	# 	corpus = np.array(self.raw_table['airline'])
	# 	self.dictVectorizer = DictVectorizer()
	# 	self.airline2Vector = self.dictVectorizer.fit_transform(corpus).toarray()
	# 	print self.airline2Vector
	def airline2Vector(self):
		corpus = np.array(self.raw_table['airline'])
		set_corpus = set(corpus)
		list_corpus = list(set_corpus)
		# print array_corpus
		length_row = len(corpus)
		length_column = len(set_corpus)
		self.airline2Vector = np.zeros((length_row, length_column))
		# print corpus
		for i in range(length_row):
			for j in range(length_column):
				if corpus[i] == list_corpus[j]:
					self.airline2Vector[i][j] = 1.0
		pd.DataFrame(self.airline2Vector).to_csv('../data/Airline2Vector.csv', sep=',')
		return self.airline2Vector

	def text2Vector(self):
		self.text2Vector = self.word2Vector('text', 30)
		pd.DataFrame(self.text2Vector).to_csv('../data/Text2Vector.csv', sep=',')
		return self.text2Vector

	def tweetCreated2Vector(self):
		self.tweetCreated2Vector = self.word2Vector('tweet_created', 10)
		pd.DataFrame(self.tweetCreated2Vector).to_csv('../data/TweetCreated2Vector.csv', sep=',')
		return self.tweetCreated2Vector

	def tweetlocation2Vector(self):
		self.tweetlocation2Vector = self.word2Vector('tweet_location', 10)
		pd.DataFrame(self.tweetlocation2Vector).to_csv('../data/TweetLocation2Vector.csv', sep=',')
		return self.tweetlocation2Vector

	def userTimezone2Vector(self):
		self.userTimezone2Vector = self.word2Vector('user_timezone', 10)
		pd.DataFrame(self.userTimezone2Vector).to_csv('../data/UserTimezone2Vector.csv', sep=',')
		return self.userTimezone2Vector

def main():
	csvfilepath = '../data/Tweets.csv'
	data = Data(csvfilepath)

if __name__ == "__main__":
	main()
