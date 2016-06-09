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
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from math import sqrt
import matplotlib.pyplot as plt
import gensim, logging
import pandas as pd


class Data:
	def __init__(self, csvfilepath):
		self.feature()
		self.dataType()
		self.raw_table = pd.read_csv(csvfilepath, usecols=self.feature, skip_blank_lines=True, dtype=self.data_type, keep_default_na=True)
		self.dropNaN()
		self.setOutput()
		self.doText2Vec()
	def printRawTable(self):
		print self.raw_table
	def feature(self):
		self.feature = ['airline_sentiment', 'airline_sentiment_confidence', 'airline', 'text', 'tweet_created', 'tweet_location', 'user_timezone']
	def dataType(self):
		self.data_type = {'airline_sentiment': str, 'airline_sentiment_confidence': np.float64, 'airline': str, 'text': str, 'tweet_created': str, 'tweet_location': str, 'user_timezone': str}
	def dropNaN(self):
		self.raw_table = self.raw_table.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
		self.raw_table = self.raw_table.dropna(axis=1, how='any', thresh=None, subset=None, inplace=False)
	def setOutput(self):
		airline_sentiment = np.array(self.raw_table['airline_sentiment'])
		airline_sentiment_confidence = np.array(self.raw_table['airline_sentiment_confidence'])
		length = len(airline_sentiment)
		self.output = np.zeros(length)
		for idx in range(length):
			if airline_sentiment[idx] == 'neutral':
				self.output[idx] = np.float64(0.5 * airline_sentiment_confidence[idx])
			elif airline_sentiment[idx] == 'positive':
				self.output[idx] = np.float64(1.0 * airline_sentiment_confidence[idx])
			else:
				self.output[idx] = np.float64(0.0)
	def doText2Vec(self):
		corpus = np.array(self.raw_table['text'])
		self.vectorizer = TfidfVectorizer(min_df=1, stop_words='english', encoding='utf-8')
		self.vectorizer.fit_transform(corpus)
		# print self.vectorizer.toarray()
	# def setInput(self):

def main():
	csvfilepath = '../data/Tweets.csv'
	data = Data(csvfilepath)

if __name__ == "__main__":
    main()
# data.printRawTable()
# print type(data.raw_table)
# print data.raw_table.dtypes
# for key in data.raw_table:
# 	for value in data.raw_table[key]:
# 		print type(value), value
# 	# print type(data.raw_table[key][2]), data.raw_table[key][2]
# length_of_sequence = 100
# data = Data(length_of_sequence)
# data.length_of_unit = 5
# data.input = np.zeros((data.length_of_sequence - data.length_of_unit, data.length_of_unit))
# data.output = np.zeros((data.length_of_sequence - data.length_of_unit, 1))
# for i in range(0, data.length_of_sequence - data.length_of_unit):
# 	data.output[i] = data.sequence[data.length_of_unit + i]
# 	for j in range(0, data.length_of_unit):
# 		data.input[i][j] = data.sequence[i + j]

# data.length_of_prediction_sequence = 20
# data.input_scaler = preprocessing.StandardScaler()
# data.input_transform = data.input_scaler.fit_transform(data.input)
# data.output_scaler = preprocessing.StandardScaler()
# data.output_transform = data.output_scaler.fit_transform(data.output)
# data.input_transform_train = data.input_transform[:(data.length_of_sequence - data.length_of_unit - data.length_of_prediction_sequence)]
# data.output_transform_train =  data.output_transform[:(data.length_of_sequence - data.length_of_unit - data.length_of_prediction_sequence)].ravel()
# data.input_transform_test = data.input_transform[(data.length_of_sequence - data.length_of_unit - data.length_of_prediction_sequence):]
# data.output_test = data.output[(data.length_of_sequence - data.length_of_unit - data.length_of_prediction_sequence):].ravel()

# svr_linear = GridSearchCV(SVR(kernel='linear', gamma=0.1), cv=10, param_grid={'C': np.logspace(-10.0, 10.0, num=40, base=2.0), 'gamma': np.logspace(-10.0, 10.0, num=40, base=2.0)})
# svr_linear.fit(data.input_transform_train, data.output_transform_train)
# data.output_transform_predict_linear = svr_linear.predict(data.input_transform_test)
# data.output_predict_linear = data.output_scaler.inverse_transform(data.output_transform_predict_linear.reshape((data.length_of_prediction_sequence, 1)))
# data.square_root_of_mean_squared_error_linear = sqrt(mean_squared_error(data.output_test, data.output_predict_linear))

# svr_rbf = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=10, param_grid={'C': np.logspace(-10.0, 10.0, num=40, base=2.0), 'gamma': np.logspace(-10.0, 10.0, num=40, base=2.0)})
# svr_rbf.fit(data.input_transform_train, data.output_transform_train)
# data.output_transform_predict_rbf = svr_rbf.predict(data.input_transform_test)
# data.output_predict_rbf = data.output_scaler.inverse_transform(data.output_transform_predict_rbf.reshape((data.length_of_prediction_sequence, 1)))
# data.square_root_of_mean_squared_error_rbf = sqrt(mean_squared_error(data.output_test, data.output_predict_rbf))

# svr_sigmoid = GridSearchCV(SVR(kernel='sigmoid', gamma=0.1), cv=10, param_grid={'C': np.logspace(-10.0, 10.0, num=40, base=2.0), 'gamma': np.logspace(-10.0, 10.0, num=40, base=2.0)})
# svr_sigmoid.fit(data.input_transform_train, data.output_transform_train)
# data.output_transform_predict_sigmoid = svr_sigmoid.predict(data.input_transform_test)
# data.output_predict_sigmoid = data.output_scaler.inverse_transform(data.output_transform_predict_sigmoid.reshape((data.length_of_prediction_sequence, 1)))
# data.square_root_of_mean_squared_error_sigmoid = sqrt(mean_squared_error(data.output_test, data.output_predict_sigmoid))

# plt.figure(1)
# x = np.arange(0, data.length_of_prediction_sequence)
# plt.plot(x, data.output_test, 'ro-', linewidth=2.0, label='Actual')
# plt.plot(x, data.output_predict_linear, 'bo-', linewidth=2.0, label='Predicted')
# plt.title('linear-SVR: RMSE = %.3f' %data.square_root_of_mean_squared_error_linear)
# plt.legend()
# plt.grid(True)
# plt.show()

# plt.figure(2)
# x = np.arange(0, data.length_of_prediction_sequence)
# plt.plot(x, data.output_test, 'ro-', linewidth=2.0, label='Actual')
# plt.plot(x, data.output_predict_rbf, 'bo-', linewidth=2.0, label='Predicted')
# plt.title('rbf-SVR: RMSE = %.3f' %data.square_root_of_mean_squared_error_rbf)
# plt.legend()
# plt.grid(True)
# plt.show()

# plt.figure(3)
# x = np.arange(0, data.length_of_prediction_sequence)
# plt.plot(x, data.output_test, 'ro-', linewidth=2.0, label='Actual')
# plt.plot(x, data.output_predict_sigmoid, 'bo-', linewidth=2.0, label='Predicted')
# plt.title('sigmoid-SVR: RMSE = %.3f' %data.square_root_of_mean_squared_error_sigmoid)
# plt.legend()
# plt.grid(True)
# plt.show()
