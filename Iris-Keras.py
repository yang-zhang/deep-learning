
# coding: utf-8

# In[4]:

get_ipython().magic('matplotlib inline')
import os
import matplotlib.pyplot as plt

import numpy as np
import sklearn.datasets
import sklearn.dummy
import sklearn.model_selection
import sklearn.ensemble
import sklearn.metrics

os.environ["KERAS_BACKEND"] = "theano"
from keras.models import Sequential
from keras.layers import Dense, Activation
import keras.utils


# In[5]:

iris = sklearn.datasets.load_iris()
X = iris.data
y = iris.target


# In[16]:

sklearn.model_selection.cross_val_score(sklearn.dummy.DummyClassifier(), X, y, scoring='neg_log_loss')


# In[19]:

sklearn.model_selection.cross_val_score(sklearn.ensemble.RandomForestClassifier(), X, y, scoring='neg_log_loss')


# In[22]:

y_cat = keras.utils.to_categorical(y)


# In[32]:

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=X.shape[1]))
model.add(Dense(16, activation='relu'))
model.add(Dense(y_cat.shape[1], activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X, y_cat, epochs=100, batch_size=1)


# In[33]:

y_pred = model.predict(X)


# In[34]:

sklearn.metrics.log_loss(y_cat, y_pred)

