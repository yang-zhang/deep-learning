
# coding: utf-8

# In[1]:

from ds_utils.imports import *


# In[2]:

keras.layers.TimeDistributed


# In[3]:

get_ipython().magic('pinfo2 keras.layers.TimeDistributed')


# In[4]:

num_samples = 5
num_timestamps = 10 
num_features = 3

inputs = np.random.random(size=(num_samples, num_timestamps, num_features))


# In[5]:

model = keras.models.Sequential()


# In[8]:

layer_dense = keras.layers.Dense(5)


# In[10]:

layer_timedistributed = keras.layers.TimeDistributed(layer_dense, input_shape=(num_timestamps, num_features))


# In[11]:

model.add(layer_timedistributed)


# In[13]:

model.output_shape


# In[17]:

layer_dense.get_weights()


# In[18]:

layer_timedistributed.get_weights()


# In[14]:

weights = layer.get_weights()


# In[15]:

A, b = tuple(weights)


# In[16]:

A, b


# In[43]:

model_outputs = model.predict(inputs)


# In[44]:

model_outputs.shape


# In[47]:

model_outputs[0]


# In[48]:

np.dot(inputs[0], A) + b


# In[49]:

for i in range(num_samples):
    assert np.allclose(model_outputs[0], np.dot(inputs[0], A) + b)

