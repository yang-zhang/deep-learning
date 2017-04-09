
# coding: utf-8

# Keras Classification example using
# - Sequential model
# - Functional API

# In[2]:

from ds_utils.imports import *


# In[3]:

iris = sklearn.datasets.load_iris()
X = iris.data
y = iris.target


# ## Benchmark

# In[4]:

sklearn.model_selection.cross_val_score(sklearn.dummy.DummyClassifier(), X, y, scoring='neg_log_loss')


# In[5]:

sklearn.model_selection.cross_val_score(sklearn.ensemble.RandomForestClassifier(), X, y, scoring='neg_log_loss')


# ## Sequential model

# In[6]:

y_cat = keras.utils.to_categorical(y)


# In[16]:

model = keras.models.Sequential()
model.add(keras.layers.Dense(32, activation='relu', input_dim=X.shape[1]))
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(y_cat.shape[1], activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X, y_cat, epochs=100, batch_size=1)


# In[33]:

y_pred = model.predict(X)


# In[34]:

sklearn.metrics.log_loss(y_cat, y_pred)


# ## Functional API

# In[11]:

inputs = keras.layers.Input(shape=(X.shape[1],))

x = keras.layers.Dense(32, activation='relu')(inputs)
x = keras.layers.Dense(16, activation='relu')(x)
predictions = keras.layers.Dense(y_cat.shape[1], activation='softmax')(x)

model = keras.models.Model(inputs=inputs, outputs=predictions)

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X, y_cat, epochs=10, batch_size=1)


# ## References
# - https://keras.io/getting-started/functional-api-guide/
