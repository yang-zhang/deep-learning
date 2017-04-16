
# coding: utf-8

# In[1]:

from ds_utils.imports import *


# In[33]:

iris = sklearn.datasets.load_iris()
X = iris.data
y = iris.target
y_cat = keras.utils.to_categorical(y)

X.shape, y.shape, y_cat.shape


# In[39]:

model = keras.models.Sequential()
model.add(keras.layers.Dense(3, activation='relu', input_dim=X.shape[1]))
model.add(keras.layers.Dense(5, activation='relu'))
model.add(keras.layers.Dense(y_cat.shape[1], activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[47]:

model.fit(X, y_cat, epochs=10, batch_size=1, verbose=0)


# ### Summary

# In[40]:

model.summary()


# ### Layers

# In[137]:

layers = model.layers


# In[138]:

layer = layers[2]


# In[139]:

layer.get_config()


# In[140]:

layer.get_weights()


# In[ ]:

layer.


# ### Model Config

# In[46]:

config = model.get_config()
config


# In[90]:

pd.DataFrame([layer['config']['kernel_initializer']['config'] for layer in config])


# In[89]:

pd.DataFrame([layer['config']['bias_initializer']['config'] for layer in config])


# In[23]:

pd.DataFrame([layer['config'] for layer in config])


# ### Model Weights

# In[123]:

[w.shape for w in weights]


# In[126]:

for layer in model.layers:
    print(layer.get_config())
    print(layer.get_weights())


# In[127]:

weights[0]


# In[128]:

np.dot(X[0], weights[0][:,0])


# In[129]:

np.dot(X[0], weights[0])


# In[93]:

assert np.dot(X[0], weights[0][:,0]) == (np.dot(X[0], weights[0]))[0]


# In[ ]:



