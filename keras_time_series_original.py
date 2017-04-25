
# coding: utf-8

# In[1]:

from ds_utils.imports import *


# In[3]:

dataframe = pd.read_csv('data/international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)


# In[ ]:

np.random.seed(7)
dataset = dataframe.values
dataset = dataset.astype('float32')


# In[9]:

scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)


# In[32]:

train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))


# In[33]:

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


# In[34]:

# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)


# In[35]:

trainX.shape


# In[36]:

trainY.shape


# In[37]:

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


# In[38]:

trainX.shape


# In[39]:

# create and fit the LSTM network
model = keras.models.Sequential()
model.add(keras.layers.LSTM(4, input_shape=(1, look_back)))
model.add(keras.layers.Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)


# In[25]:

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


# In[26]:

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])


# In[28]:

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
