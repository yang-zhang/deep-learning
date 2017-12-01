
# coding: utf-8

# In[1]:

import keras
import numpy as np


# In[2]:

keras.layers.TimeDistributed


# In[3]:

get_ipython().magic('pinfo2 keras.layers.TimeDistributed')


# In[4]:

num_samples = 10
num_timestamps = 7 
num_features = 3

inputs = np.random.random(size=(num_samples, num_timestamps, num_features))


# In[5]:

model = keras.models.Sequential()


# In[6]:

layer_dense = keras.layers.Dense(5)


# In[7]:

layer_timedistributed = keras.layers.TimeDistributed(layer_dense, input_shape=(num_timestamps, num_features))


# In[8]:

model.add(layer_timedistributed)


# In[9]:

model.output_shape


# In[10]:

model.summary()


# In[23]:

assert 5 * (num_features + 1) == 20


# In[11]:

layer_dense.get_weights()


# In[12]:

layer_timedistributed.get_weights()


# In[13]:

weights = model.get_weights()


# In[14]:

A, b = tuple(weights)


# In[15]:

A, b


# In[16]:

model_outputs = model.predict(inputs)


# In[17]:

model_outputs.shape


# In[18]:

model_outputs[4]


# In[19]:

np.dot(inputs[4], A) + b


# In[20]:

for i in range(num_samples):
    assert np.allclose(model_outputs[i], np.dot(inputs[i], A) + b)


# In[ ]:



