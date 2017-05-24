
# coding: utf-8

# In[1]:

from ds_utils.imports import *


# In[2]:

iris = sklearn.datasets.load_iris()
X = iris.data
y = iris.target
y_cat = keras.utils.to_categorical(y)

X.shape, y.shape, y_cat.shape


# In[3]:

model = keras.models.Sequential()
model.add(keras.layers.Dense(3, activation='relu', input_dim=X.shape[1]))
model.add(keras.layers.Dense(5, activation='relu'))
model.add(keras.layers.Dense(y_cat.shape[1], activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Option-1

# In[4]:

from keras.utils import plot_model

plot_model(model, to_file='test_keras_plot_model.png', show_shapes=True)

from IPython.display import Image 
Image("test_keras_plot_model.png")


# Option-2

# In[5]:

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
SVG(model_to_dot(model).create(prog='dot', format='svg'))


# In[6]:

def plot_keras_model(model):
    from IPython.display import SVG
    from keras.utils.vis_utils import model_to_dot
    return SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))


# In[7]:

plot_keras_model(model)


# In[8]:

import ds_utils.misc


# In[10]:

ds_utils.misc.plot_keras_model(model)


# In[11]:

ds_utils.misc.plot_keras_model(model, show_shapes=False)


# https://keras.io/visualization/
