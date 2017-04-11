
# coding: utf-8

# Demo to verify the number of paramters of layers in Keras models.

# In[47]:

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D


# ### Multilayer Perceptron (MLP) for multi-class softmax classification

# In[48]:

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(7, activation='relu', input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(13, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(23, activation='softmax'))


# In[49]:

print(model.summary())

assert(7 * (20+1) == 147)
assert(13 * (7+1) == 104)
assert(23 * (13+1) == 322)


# ### MLP for binary classification

# In[51]:

model = Sequential()
model.add(Dense(7, input_dim=20, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(13, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


# In[52]:

print(model.summary())

assert(7 * (20+1) == 147)
assert(13 * (7+1) == 104)
assert(1 * (13+1) == 14)


# ### VGG-like convnet

# In[94]:

model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 23 convolution filters of size 3x3 each.
model.add(Conv2D(7, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(Conv2D(13, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(17, (3, 3), activation='relu'))
model.add(Conv2D(19, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(23, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(29, activation='softmax'))


# In[93]:

print(model.summary())

assert(7 * (3 * (3*3) + 1) == 196)
assert(13 * (7 * (3*3) + 1) == 832)
assert(17 * (13 * (3*3) + 1) == 2006)
assert(19 * (17 * (3*3) + 1) == 2926)
assert(22 * 22 * 19 == 9196)
assert(23 * (9196+1) == 211531)
assert(29 * (23+1) == 696)


# ### Sequence classification with LSTM

# In[69]:

max_features = 7
model = Sequential()
model.add(Embedding(max_features, output_dim=13))
model.add(LSTM(17))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


# In[70]:

print(model.summary())

assert(13 * 7 == 91)
print(2108/17, 'then?')
assert(1 * (17+1) == 18)


# ### Sequence classification with 1D convolutions

# In[121]:

seq_length = 70
model = Sequential()
model.add(Conv1D(13, 5, activation='relu', input_shape=(seq_length, 103)))
model.add(Conv1D(17, 4, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(19, 7, activation='relu'))
model.add(Conv1D(23, 6, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


# In[125]:

6708/13


# In[126]:

(516-1)/103


# In[136]:

print(model.summary())
assert(13 * (5*103 + 1) == 6708)
assert(17 * (4*13 + 1) == 901)
assert(63/3 == 21)
assert(19 * (7*17 + 1) == 2280)
assert(23 * (6*19 + 1) == 2645)
assert(1 * (23 + 1) == 24)


# In[ ]:




# Reference: 
# - https://keras.io/getting-started/sequential-model-guide/

# In[ ]:



