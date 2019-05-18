# -*- coding: utf-8 -*-
"""
Created on Sat May 18 11:20:22 2019

@author: ashis
"""
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import pandas as pd


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data) 
	cols, names = list(), list() 
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):   
		cols.append(df.shift(i)) 
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out): 
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1) 
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg 


numbers = pd.read_csv("hcltech_input.csv")
close = pd.DataFrame(numbers['close'])
close = close.dropna()
n_hours = 7
numbersX = series_to_supervised(close, n_hours,0) 

x = numbers[[ 'close','change',
 'stock_volatility',
 'stock_momentum',
 'index_volatility',
 'index_momentum',
 'sector_momentum',]]
 

final_dataset = pd.concat([numbersX,x],axis = 1, sort = False) 
final_dataset = final_dataset.dropna()
final_dataset = final_dataset.rename(columns={ final_dataset.columns[0]: "close" })
values = final_dataset.values 
values = values.astype('float32')
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled = scaler.fit_transform(values) 


n_train_hours = 900
train = values[:n_train_hours, :] 
test = values[n_train_hours:, :]

train_X, train_Y = train[:, 1:], train[:,0:1]
test_X, test_Y = test[:, 1:], test[:, 0:1]

train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))


model = Sequential() 
model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
model.fit(train_X, train_Y, epochs=50, batch_size=72, validation_data=(test_X, test_Y), verbose=2, shuffle=False)

trainPredict = model.predict(train_X)
testPredict = model.predict(test_X)

"""
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
train_Y = scaler.inverse_transform([train_Y])
testPredict = scaler.inverse_transform(testPredict)
test_Y = scaler.inverse_transform([test_Y])
"""



