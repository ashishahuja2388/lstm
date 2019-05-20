# -*- coding: utf-8 -*-
"""
Created on Sun May 19 21:03:10 2019

@author: ashis
"""

from numpy import array
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score
 
 
# split the input data into continuous samples 
def split_sequence(sequence, window_size):
	X, y = list(), list()
	for i in range(len(sequence)):
		end_ix = i + window_size
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
 
# get the input data 
data = pd.read_csv("mindtree_input.csv")
close = pd.DataFrame(data["close"])
close = close.dropna()
values = close.values
values = values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1)) 
scaled = scaler.fit_transform(values) 

# choose a window size
window_size = 7
# frame as supervised learning 
X, y = split_sequence(scaled, window_size) 

# Split the data in training and testing
n_samples = 900
X_train = X[:n_samples, :] 
X_test = X[n_samples: , :]
y_train = y[:n_samples, :]
y_test = y[n_samples : , :]

# Reshape the input to 3D
n_features = 1
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features)) 
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))

# design network
model = Sequential() 
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
model.fit(X_train, y_train, epochs=50, batch_size=72, validation_data=(X_test, y_test), verbose=2, shuffle=False)
#make predictions
yhat = model.predict(X_test) 
prediction = scaler.inverse_transform(yhat)
y_test = scaler.inverse_transform(y_test)

#plot on graph
plt.plot(prediction, color = 'orange', label = 'TATA Stock Price')
plt.plot(y_test, color = 'green', label = 'Predicted TATA Stock Price')
plt.xlabel('Time')
plt.ylabel('HCL Stock Price')
plt.legend()
plt.show()

