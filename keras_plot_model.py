
# coding: utf-8

# In[6]:

import keras
import IPython

model = keras.models.Sequential()
model.add(keras.layers.Dense(3, activation='relu', input_dim=3))
model.add(keras.layers.Dense(5, activation='relu'))
model.add(keras.layers.Dense(2, activation='softmax'))


# Option-1

# In[7]:

keras.utils.plot_model(model, to_file='test_keras_plot_model.png', show_shapes=True)
IPython.display.Image("test_keras_plot_model.png")


# Option-2

# In[11]:

IPython.display.SVG(keras.utils.vis_utils.model_to_dot(model).create(prog='dot', format='svg'))


# In[9]:

def plot_keras_model(model, show_shapes=True):
    return IPython.display.SVG(keras.utils.vis_utils.model_to_dot(model, show_shapes=show_shapes).create(prog='dot', format='svg'))


# In[10]:

plot_keras_model(model, show_shapes=True)


# https://keras.io/visualization/
