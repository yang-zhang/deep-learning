
# coding: utf-8

# In[1]:

from sklearn import datasets
from sklearn.metrics import log_loss

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical


# ## Data

# In[2]:

data = datasets.load_iris()


# In[3]:

X_train = data.data


# In[4]:

y_train = data.target


# In[5]:

X_train.shape


# In[6]:

y_train.shape


# In[7]:

y_train_binary = to_categorical(y_train)


# ## Model

# In[8]:

mdl = Sequential()

mdl.add(Dense(10, input_dim=4, activation='relu'))
mdl.add(Dense(output_dim=3, activation='softmax'))


# In[9]:

mdl.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.01))


# In[10]:

mdl.fit(X_train, y_train_binary, nb_epoch=50, batch_size=32)


# In[11]:

pred_classes = mdl.predict_classes(X_train, batch_size=32)
pred_prob = mdl.predict_proba(X_train, batch_size=32)


# In[12]:

act = y_train_binary
pred = pred_prob


# In[13]:

log_loss(y_train_binary.reshape(-1,1), pred_prob.reshape(-1, 1))


# - https://keras.io/
