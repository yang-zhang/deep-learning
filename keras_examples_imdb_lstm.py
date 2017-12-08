
# coding: utf-8

# https://github.com/fchollet/keras/blob/master/examples/imdb_lstm.py

# In[9]:

import utilities


# In[151]:

'''Trains an LSTM model on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
# Notes
- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.
- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
import numpy as np


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import LabelEncoder

max_features = 20000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')


# In[152]:

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)


# In[153]:

word_index = imdb.get_word_index()
index_word = dict((v, k) for k, v in word_index.items())


# In[154]:

' '.join('0' if i==0 else index_word[i] for i in x_train[np.random.choice(len(x_train))])


# ### logistic regression

# In[155]:

logreg = LogisticRegression()
logreg.fit(x_train, np.array(y_train).reshape(-1, 1))


# In[157]:

valid_probs = logreg.predict_proba(x_test)
valid_preds = logreg.predict(x_test)


# In[165]:

log_loss(y_test, valid_probs)


# In[160]:

accuracy_score(y_test, valid_preds)


# ### rnn

# In[87]:

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[85]:

model.summary()


# In[86]:

utilities.print_weights_shape(model)


# In[87]:

utilities.plot_keras_model(model)


# In[88]:

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=15,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)


# In[ ]:



