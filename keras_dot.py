
# coding: utf-8

# In[1]:

import keras
import numpy as np


# In[17]:

max_num = 10
n_samples = 10


# ### 1-D case

# In[93]:

a = keras.Input(shape=(3,))
b = keras.Input(shape=(3,))
dot_layer = keras.layers.dot([a, b], axes=[1,1])
model = keras.models.Model([a, b], dot_layer)

a_input = np.random.choice(max_num, (n_samples, 3))
b_input = np.random.choice(max_num, (n_samples, 3))
dot_product = model.predict([a_input, b_input])

dot_product.shape


# In[94]:

np.dot(a_input[0], b_input[0])


# In[95]:

dot_product[0]


# In[96]:

for i in range(n_samples):
    assert np.dot(a_input[i], b_input[i]) == dot_product[i][0]


# ### 2-D case

# #### 2-D case - 1

# In[99]:

a = keras.Input(shape=(3,4))
b = keras.Input(shape=(4,5))
dot_layer = keras.layers.dot([a, b], axes=(2, 1))
model = keras.models.Model([a, b], dot_layer)

a_input = np.random.choice(max_num, (n_samples, 3, 4))
b_input = np.random.choice(max_num, (n_samples, 4, 5))
dot_product = model.predict([a_input, b_input])

dot_product.shape


# In[100]:

np.dot(a_input[0], b_input[0])


# In[101]:

dot_product[0]


# In[102]:

for i in range(n_samples):
    assert np.alltrue(np.dot(a_input[i], b_input[i]) == dot_product[i])

