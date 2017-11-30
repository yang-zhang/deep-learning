
# coding: utf-8

# https://medium.com/p/3eb35ab7f3c/edit

# In[1]:

img_rows, img_cols = 224, 224
colors = 3
input_size = img_rows * img_cols * colors
input_shape = (img_rows, img_cols, colors)

num_classes = 10


# In[3]:

from keras.models import Sequential
from keras.layers import Dense

model = Sequential([
    Dense(32, activation='relu', input_shape=(input_size,)),
    Dense(64, activation='relu'),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.summary()


# In[4]:

assert 32 * (input_size + 1) == 4816928
assert 64 * (32 + 1) == 2112
assert 128 * (64 + 1) == 8320
assert num_classes * (128 + 1) == 1290


# In[5]:

from keras.models import Sequential
from keras.layers import Conv2D


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    Conv2D(64, (3, 3), activation='relu'),
    Conv2D(128, (3, 3), activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.summary()


# In[6]:

assert 32 * (3 * (3*3) + 1) == 896
assert 64 * (32 * (3*3) + 1) == 18496
assert 128 * (64 * (3*3) + 1) == 73856
assert num_classes * (128 + 1) == 1290


# In[7]:

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


# In[8]:

assert 32 * (3 * (3*3) + 1) == 896
assert 64 * (32 * (3*3) + 1) == 18496
assert 110 * 110 * 64 == 774400
assert 128 * (774400 + 1) == 99123328
assert num_classes * (128 + 1) == 1290

