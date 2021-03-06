
# coding: utf-8

# https://medium.com/p/3eb35ab7f3c/edit

# In[41]:

import utilities


# In[1]:

img_rows, img_cols = 224, 224
colors = 3
input_size = img_rows * img_cols * colors
input_shape = (img_rows, img_cols, colors)

num_classes = 10


# In[2]:

from keras.models import Sequential
from keras.layers import Dense

model = Sequential([
    Dense(32, activation='relu', input_shape=(input_size,)),
    Dense(64, activation='relu'),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.summary()


# In[42]:

utilities.print_weights_shape(model)

output_size * (input_size + 1) == number_parameters
# In[4]:

assert 32 * (input_size + 1) == 4816928
assert 64 * (32 + 1) == 2112
assert 128 * (64 + 1) == 8320
assert num_classes * (128 + 1) == 1290


# In[44]:

from keras.models import Sequential
from keras.layers import Conv2D


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    Conv2D(64, (3, 3), activation='relu'),
    Conv2D(128, (3, 3), activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.summary()


# In[45]:

utilities.print_weights_shape(model)

output_size * (input_size + 1) == number_parameters

i.e.,

output_channels * (input_channels * window_size + 1) == number_parameters

here window_size=3*3
# In[6]:

assert 32 * (3 * (3*3) + 1) == 896
assert 64 * (32 * (3*3) + 1) == 18496
assert 128 * (64 * (3*3) + 1) == 73856
assert num_classes * (128 + 1) == 1290


# In[46]:

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.summary()


# In[47]:

utilities.print_weights_shape(model)


# In[48]:

assert 32 * (3 * (3*3) + 1) == 896
assert 64 * (32 * (3*3) + 1) == 18496
assert 110 * 110 * 64 == 774400
assert 128 * (774400 + 1) == 99123328
assert num_classes * (128 + 1) == 1290


# In[ ]:



