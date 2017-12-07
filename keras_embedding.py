
# coding: utf-8

# In[7]:

import numpy as np

from keras.layers import Embedding
from keras.models import Sequential

import utilities


# In[101]:

# https://keras.io/layers/embeddings/#embedding 
model = Sequential()
model.add(Embedding(1000, 64, input_length=10))
# the model will take as input an integer matrix of size (batch, input_length).
# the largest integer (i.e. word index) in the input should be no larger than 999 (vocabulary size).
# now model.output_shape == (None, 10, 64), where None is the batch dimension.

input_array = np.random.randint(1000, size=(32, 10))

model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)
assert output_array.shape == (32, 10, 64)


# In[111]:

model = Sequential([Embedding(input_dim=13, output_dim=17)])
model.compile('rmsprop', 'mse')


# In[112]:

model.summary()


# In[113]:

utilities.plot_keras_model(model)


# In[114]:

utilities.print_weights_shape(model)


# In[115]:

weights = model.get_weights()[0]


# In[116]:

weights.shape


# In[117]:

x = np.random.choice(a=13, size=(20, 30))


# In[118]:

x.shape


# In[119]:

output = model.predict(x)


# In[120]:

output.shape


# In[121]:

x[0].shape


# In[123]:

output[0].shape


# In[124]:

x[0]


# In[128]:

output[0][0]


# In[126]:

weights[x[0][0]]


# In[129]:

np.alltrue(output[0][0]==weights[x[0][0]])


# In[130]:

np.alltrue(output[2][3]==weights[x[2][3]])


# In[133]:

for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        assert np.alltrue(output[i][j]==weights[x[i][j]])

