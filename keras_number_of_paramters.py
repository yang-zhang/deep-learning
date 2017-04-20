
# coding: utf-8

# Demo to verify the number of paramters of layers in Keras models.

# In[12]:

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.layers import SimpleRNN, LSTM


# ### Multilayer Perceptron (MLP) for multi-class softmax classification

# In[2]:

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(7, activation='relu', input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(13, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(23, activation='softmax'))


# In[3]:

print(model.summary())

assert(7 * (20+1) == 147)
assert(13 * (7+1) == 104)
assert(23 * (13+1) == 322)


# ### MLP for binary classification

# In[4]:

model = Sequential()
model.add(Dense(7, input_dim=20, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(13, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


# In[5]:

print(model.summary())

assert(7 * (20+1) == 147)
assert(13 * (7+1) == 104)
assert(1 * (13+1) == 14)


# ### VGG-like convnet

# In[6]:

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


# In[7]:

print(model.summary())

assert(7 * (3 * (3*3) + 1) == 196)
assert(13 * (7 * (3*3) + 1) == 832)
assert(17 * (13 * (3*3) + 1) == 2006)
assert(19 * (17 * (3*3) + 1) == 2926)
assert(22 * 22 * 19 == 9196)
assert(23 * (9196+1) == 211531)
assert(29 * (23+1) == 696)


# ### Simple RNN
# https://github.com/yang-zhang/courses/blob/master/deeplearning1/nbs/lesson6.ipynb

# In[81]:

n_hidden, n_fac, cs, vocab_size = (256, 42, 8, 86)

model=Sequential([
        Embedding(input_dim=vocab_size, output_dim=n_fac, input_length=cs),
        SimpleRNN(n_hidden, activation='relu', inner_init='identity'),
        Dense(vocab_size, activation='softmax')
    ])

print(model.summary())


# In[91]:

assert 86 * 42 == 3612

assert 86 * (256 + 1) == 22102

assert 256 * (42 + (256 + 1)) == 76544


# ### Sequence classification with LSTM

# In[8]:

max_features = 7
model = Sequential()
model.add(Embedding(max_features, output_dim=13))
model.add(LSTM(17))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


# In[9]:

print(model.summary())
# TODO

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


# In[136]:

print(model.summary())
assert(13 * (5*103 + 1) == 6708)
assert(17 * (4*13 + 1) == 901)
assert(63/3 == 21)
assert(19 * (7*17 + 1) == 2280)
assert(23 * (6*19 + 1) == 2645)
assert(1 * (23 + 1) == 24)


# ### Stacked LSTM for sequence classification

# In[140]:

data_dim = 11
timesteps = 7
num_classes = 13

# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(31, return_sequences=True,
               input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 31
model.add(LSTM(37, return_sequences=True))  # returns a sequence of vectors of dimension 37
model.add(LSTM(41))  # return a single vector of dimension 41
model.add(Dense(17, activation='softmax'))


# In[150]:

print(model.summary())
# TODO
(5332/31)
172/4


# ### Same stacked LSTM model, rendered "stateful"

# In[151]:

data_dim = 11
timesteps = 7
num_classes = 13
batch_size = 47

# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(31, return_sequences=True, stateful=True,
               batch_input_shape=(batch_size, timesteps, data_dim)))  # returns a sequence of vectors of dimension 31
model.add(LSTM(37, return_sequences=True, stateful=True))  # returns a sequence of vectors of dimension 37
model.add(LSTM(41, stateful=True))  # return a single vector of dimension 41
model.add(Dense(17, activation='softmax'))


# In[ ]:

data_dim = 16
timesteps = 8
num_classes = 10
batch_size = 32

# Expected input batch shape: (batch_size, timesteps, data_dim)
# Note that we have to provide the full batch_input_shape since the network is stateful.
# the sample of index i in batch k is the follow-up for the sample i in batch k-1.
model = Sequential()
model.add(LSTM(32, return_sequences=True, stateful=True,
               batch_input_shape=(batch_size, timesteps, data_dim)))
model.add(LSTM(32, return_sequences=True, stateful=True))
model.add(LSTM(32, stateful=True))
model.add(Dense(10, activation='softmax'))


# In[153]:

print(model.summary())
# TODO


# Reference: 
# - https://keras.io/getting-started/sequential-model-guide/

# In[ ]:



