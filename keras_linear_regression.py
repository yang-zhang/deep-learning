
# coding: utf-8

# - Linear regression in sklearn v.s. Keras. 
# - Validation data in Keras
# - Getting coefficients (weights) in Keras.

# In[43]:

import ds_utils.imports; import imp; imp.reload(ds_utils.imports)
from ds_utils.imports import *


# ## Data

# In[100]:

X = np.random.uniform(size=1000).reshape(-1, 1)
bias = 1
w = 2
noise = np.random.normal(scale=0.1, size=y.shape)
y = np.dot(X, w) + bias + noise
y = y.reshape(-1, 1)


# In[101]:

plt.scatter(X, y)


# In[102]:

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, test_size=0.2)


# In[103]:

X_train.shape, y_train.shape, X_test.shape, y_test.shape


# ## Linear regression

# In[106]:

mdl = sklearn.linear_model.LinearRegression().fit(X_train, y_train)


# In[107]:

y_pred = mdl.predict(X_test)


# In[108]:

sklearn.metrics.mean_squared_error(y_test, y_pred)


# In[111]:

mdl.coef_


# In[110]:

plt.scatter(X_test, y_test)
plt.scatter(X_test, y_pred)


# ## Keras linear regression

# In[112]:

mdl = keras.models.Sequential(
    [keras.layers.Dense(
        units=y_train.shape[1], input_dim=X_train.shape[1])])
mdl.compile(optimizer=keras.optimizers.SGD(lr=0.1), loss='mse', metrics=[])
mdl.fit(x=X_train,
        y=y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_test, y_test))


# In[113]:

y_pred = mdl.predict(X_test)


# In[114]:

sklearn.metrics.mean_squared_error(y_test, y_pred)


# In[116]:

plt.scatter(X_test, y_test)
plt.scatter(X_test, y_pred)


# In[123]:

mdl.evaluate(X_test, y_test)


# In[117]:

mdl.summary()


# In[121]:

mdl.get_layer('dense_6').get_weights()


# In[122]:

mdl.get_weights()

