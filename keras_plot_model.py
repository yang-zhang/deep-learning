
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


# In[12]:

def plot_keras_model(model, show_shapes=True, show_layer_names=True):
    from IPython.display import SVG
    from keras.utils.vis_utils import model_to_dot
    return SVG(model_to_dot(model, show_shapes=show_shapes, show_layer_names=show_layer_names).create(prog='dot', format='svg'))


# In[13]:

plot_keras_model(model, show_shapes=True, show_layer_names=False)


# https://keras.io/visualization/
