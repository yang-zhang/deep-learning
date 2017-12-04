
# coding: utf-8

# - https://datascience.stackexchange.com/questions/10615/number-of-parameters-in-an-lstm-model
# - https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py

# In[1]:

import utilities


# In[2]:

maxlen_sentence = 43
num_chars = 57


# In[3]:

from keras.models import Sequential
from keras.layers import SimpleRNN, LSTM

units = 128
model = Sequential([
   SimpleRNN(units, input_shape=(maxlen_sentence, num_chars)),
])

model.summary()


# In[4]:

utilities.print_weights_shape(model)


# In[5]:

assert units*units + units*(num_chars+1) == 23808


# In[6]:

from keras.models import Sequential
from keras.layers import LSTM

units = 128
model = Sequential([
   LSTM(units, input_shape=(maxlen_sentence, num_chars)),
])

model.summary()

assert 4 * units * (units + (num_chars+1)) == 95232


# In[7]:

utilities.print_weights_shape(model)


# In[8]:

assert 4*units*units + 4*units*(num_chars + 1) == 95232

