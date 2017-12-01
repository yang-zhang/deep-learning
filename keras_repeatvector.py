
# coding: utf-8

# In[1]:

import keras
import numpy as np


# In[15]:

n_repeat = 7


# In[16]:

layer_dense = keras.layers.Dense(16, )


# In[17]:

repeat_vector_layer = keras.layers.RepeatVector(n=n_repeat)


# In[19]:

n_features = 3
inputs = keras.layers.Input(shape=(n_features,))
x = inputs
x = layer_dense(x)
outputs = repeat_vector_layer(x)
model = keras.models.Model(inputs, outputs)
model.summary()


# In[20]:

n_sample = 10
x_data = np.random.randn(n_sample, n_features)


# In[21]:

y_data = model.predict(x_data)


# In[22]:

y_data.shape


# In[25]:

y[0][0]


# In[26]:

y[0][1]


# In[27]:

y[0][6]


# In[ ]:



