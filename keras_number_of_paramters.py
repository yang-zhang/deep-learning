
# coding: utf-8

# Demo to verify the number of paramters of layers in Keras models.

# In[1]:

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Embedding
from keras.layers import LSTM


# ### Multilayer Perceptron (MLP) for multi-class softmax classification

# In[40]:

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

print(model.summary())
assert(64 * (20+1) == 1344)
assert(64 * (64+1) == 4160)
assert(10 * (64+1) == 650)


# ### MLP for binary classification

# In[41]:

model = Sequential()
model.add(Dense(64, input_dim=20, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())

assert(64 * (20+1) == 1344)
assert(64 * (64+1) == 4160)
assert(1 * (64+1) == 65)


# In[48]:

model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 23 convolution filters of size 3x3 each.
model.add(Conv2D(23, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

print(model.summary())
assert(32 * (3 * (3*3) + 1) == 896)
assert(32 * (32 * (3*3) + 1) == 9248)
assert(64 * (32 * (3*3) + 1) == 18496)
assert(64 * (64 * (3*3) + 1) == 36928)
assert(22 * 22 * 64 == 30976)
assert(256 * (30976+1) == 7930112)
assert(10 * (256+1) == 2570)


# ### Sequence classification with LSTM

# In[2]:

max_features = 3
model = Sequential()
model.add(Embedding(max_features, output_dim=256))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


# Reference: 
# - https://keras.io/getting-started/sequential-model-guide/
