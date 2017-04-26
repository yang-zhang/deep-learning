
# coding: utf-8

# In[1]:

from ds_utils.imports import *


# In[45]:

keras.models.Input


# In[46]:

keras.layers.Input


# How are these two pointing to the same funcction?

# In[29]:

get_ipython().magic('pinfo2 keras.models')


# In[33]:

cat ../keras/keras/models.py | grep 'import Input'


# In[34]:

get_ipython().magic('pinfo2 keras.models.Input')


# In[35]:

cat ../keras/keras/engine/topology.py | grep 'def Input'


# In[40]:

get_ipython().magic('pinfo2 keras.layers')


# In[42]:

cat ../keras/keras/layers/__init__.py | grep 'import Input$'


# In[44]:

cat ../keras/keras/engine/__init__.py | grep 'import Input$'


# In[38]:

get_ipython().magic('pinfo2 keras.layers.Input')


# In[39]:

cat ../keras/keras/engine/topology.py | grep 'def Input'


# Ref: 
# - https://keras.io/getting-started/functional-api-guide/#multi-input-and-multi-output-models
