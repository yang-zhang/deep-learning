
# coding: utf-8

# In[1]:

from ds_utils.imports import *


# In[34]:

main_input = keras.layers.Input(shape=(1,))


# In[35]:

embedding_layer = keras.layers.Embedding(input_dim=10, output_dim=1)


# In[36]:

lstm_layer = keras.layers.LSTM(units=5)


# In[37]:

aux_input = keras.layers.Input(shape=(1,))


# In[38]:

aux_output = keras.layers.Dense(units=1)


# In[39]:

merge_layer = keras.layers.Merge(layers=[aux_input, lstm_layer], mode='sum')


# In[1]:

model = keras.models.Model()


# In[ ]:




# Ref: 
# - https://keras.io/getting-started/functional-api-guide/#multi-input-and-multi-output-models
