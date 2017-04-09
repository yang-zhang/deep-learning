
# coding: utf-8

# Keras Regression example using
# - Sequential model
# - Functional API

# In[11]:

from ds_utils.imports import *


# In[12]:

boston = sklearn.datasets.load_boston()
X = boston.data
y = boston.target


# ## Benchmark

# In[13]:

sklearn.model_selection.cross_val_score(
    sklearn.dummy.DummyRegressor(), X, y, scoring='neg_mean_squared_error')


# In[14]:

sklearn.model_selection.cross_val_score(
    xgb.XGBRegressor(), X, y, scoring='neg_mean_squared_error')


# ## Sequential model

# In[15]:

model = keras.models.Sequential()
model.add(keras.layers.Dense(32, activation='relu', input_dim=X.shape[1]))
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1))


# In[7]:

model.compile(optimizer='adam', loss = 'mean_squared_error')
model.fit(X, y, epochs=100, batch_size=32)


# In[18]:

y_pred = model.predict(X)


# In[17]:

sklearn.metrics.mean_squared_error(y, y_pred)


# ## Functional KPI

# In[23]:

inputs = keras.layers.Input(shape=(X.shape[1],))

x = keras.layers.Dense(32, activation='relu')(inputs)
x = keras.layers.Dense(16, activation='relu')(x)
predictions = keras.layers.Dense(1)(x)

model = keras.models.Model(inputs=inputs, outputs=predictions)

model.compile(optimizer='adam',
              loss='mean_squared_error')
model.fit(X, y, epochs=100, batch_size=1)


# Reference: http://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
