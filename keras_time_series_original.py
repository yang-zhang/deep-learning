
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import keras
import sklearn
import sklearn.preprocessing
import sklearn.metrics


# In[3]:

url = 'https://raw.githubusercontent.com/blue-yonder/pydse/master/pydse/data/international-airline-passengers.csv'
dataframe = pd.read_csv(url, sep=';')


# In[10]:

np.random.seed(7)
dataset = dataframe.Passengers.values
dataset = dataset.astype('float32').reshape(-1, 1)


# In[11]:

scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)


# In[12]:

train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))


# In[13]:

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


# In[14]:

# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)


# In[15]:

trainX.shape


# In[16]:

trainY.shape


# In[17]:

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


# In[18]:

trainX.shape


# In[19]:

# create and fit the LSTM network
model = keras.models.Sequential()
model.add(keras.layers.LSTM(4, input_shape=(1, look_back)))
model.add(keras.layers.Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)


# In[ ]:

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


# In[ ]:

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])


# In[20]:

import math
# calculate root mean squared error
trainScore = math.sqrt(sklearn.metrics.mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(sklearn.metrics.mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


# In[29]:

# calculate root mean squared error
trainScore = np.sqrt(sklearn.metrics.mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = np.sqrt(sklearn.metrics.mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


# http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
