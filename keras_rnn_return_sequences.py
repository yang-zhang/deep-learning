
# coding: utf-8

# In[1]:

import keras


# In[3]:

n_timestamps = 10
n_features = 5


# In[5]:

model = keras.models.Sequential()
model.add(keras.layers.LSTM(units=64,
                           input_shape=(n_timestamps, n_features)))
model.summary()


# In[6]:

model = keras.models.Sequential()
model.add(keras.layers.LSTM(units=64,
                           input_shape=(n_timestamps, n_features),
                           return_sequences=True))
model.summary()


# In[ ]:



