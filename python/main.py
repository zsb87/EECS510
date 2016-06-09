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

import numpy as np
from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer
from math import sqrt
import matplotlib.pyplot as plt
import gensim, logging
import pandas as pd

class Data(object):
	def __init__(self, csvfilepath):
		self.raw_table = pd.read_csv(csvfilepath)
	def printRawTable(self):
		print self.raw_table
	# def feature(self):
	# 	self.raw_feature = 

csvfilepath = '../data/Tweets.csv'
data = Data(csvfilepath)
data.printRawTable()
print data.raw_table['airline_sentiment']
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
