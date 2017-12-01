
# coding: utf-8

# In[1]:

import keras
import numpy as np


# In[58]:

num_samples = 10
num_timestamps = 7 
num_features = 3

inputs = keras.layers.Input(shape=(num_timestamps, num_features))


# In[59]:

inputs_data = np.random.random(size=(num_samples, num_timestamps, num_features))


# In[60]:

layer_dense = keras.layers.Dense(5)


# In[61]:

inputs.shape


# In[62]:

x = inputs
layer_timedistributed = keras.layers.TimeDistributed(layer_dense)
outputs = layer_timedistributed(x)
model = keras.models.Model(inputs, outputs)
model.summary()


# In[63]:

assert 5 * (num_features + 1) == 20


# In[64]:

layer_dense.get_weights()


# In[65]:

layer_timedistributed.get_weights()


# In[66]:

weights = model.get_weights()


# In[67]:

A, b = tuple(weights)


# In[68]:

A, b


# In[69]:

outputs_data = model.predict(inputs_data)


# In[70]:

outputs_data.shape


# In[71]:

outputs_data[4]


# In[72]:

np.dot(inputs_data[4], A) + b


# In[73]:

for i in range(num_samples):
    assert np.allclose(outputs_data[i], np.dot(inputs_data[i], A) + b)


# In[ ]:




# In[ ]:



