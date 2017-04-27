
# coding: utf-8

# In[2]:

from ds_utils.imports import *


# In[3]:

keras.layers.TimeDistributed


# In[4]:

get_ipython().magic('pinfo2 keras.layers.TimeDistributed')


# In[26]:

num_samples = 5
num_timestamps = 10 
num_features = 3

inputs = np.random.random(size=(num_samples, num_timestamps, num_features))


# In[27]:

model = keras.models.Sequential()


# In[28]:

layer = keras.layers.TimeDistributed(
    keras.layers.Dense(5), input_shape=(num_timestamps, num_features))


# In[29]:

model.add(layer)


# In[38]:

model.output_shape


# In[39]:

weights = layer.get_weights()


# In[40]:

A, b = tuple(weights)


# In[41]:

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

