
# coding: utf-8

# In[1]:

import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn.datasets
import sklearn.model_selection
import sklearn.pipeline
import sklearn.metrics

import sklearn.dummy
import sklearn.linear_model
import sklearn.ensemble


os.environ["KERAS_BACKEND"] = "theano"
import keras
from keras import backend as K
K.set_image_dim_ordering('th')


# # Prework

# Following this fast.ai [lesson](https://github.com/fastai/courses/blob/master/deeplearning1/nbs/lesson4.ipynb) and [video](https://www.youtube.com/watch?v=V2h3IOBDvrA&feature=youtu.be&t=5761).

# In[2]:

# get data
# !wget -O data/ml-latest-small.zip http://files.grouplens.org/datasets/movielens/ml-latest-small.zip
# !unzip data/ml-latest-small.zip


# # Preprocessing

# In[67]:

ratings = pd.read_csv('data/ml-latest-small/ratings.csv')


# In[68]:

ratings.shape


# In[69]:

ratings.head(5)


# In[70]:

users = ratings.userId.unique()
movies = ratings.movieId.unique()


# In[72]:

userid2idx = {o:i for i,o in enumerate(users)}
movieid2idx = {o:i for i,o in enumerate(movies)}


# In[73]:

ratings.userId = ratings.userId.apply(lambda x: userid2idx[x])
ratings.movieId = ratings.movieId.apply(lambda x: movieid2idx[x])


# In[74]:

n_users = ratings.userId.nunique()
n_movies = ratings.movieId.nunique()


# In[75]:

ratings.userId.min(), ratings.userId.max(), ratings.movieId.min(), ratings.movieId.max()


# In[76]:

n_factors = 50


# In[83]:

np.random.seed = 42


# In[84]:

msk = np.random.rand(len(ratings)) < 0.8
trn = ratings[msk]
val = ratings[~msk]


# In[89]:

msk.shape


# In[86]:

val.shape


# # Dot product

# In[160]:

user_in = keras.layers.Input(shape=(1, ), dtype='int64', name='user_in')

u = keras.layers.Embedding(
    n_users,
    n_factors,
    input_length=1,    
    W_regularizer=keras.regularizers.l2(1e-4)
)(user_in)

movie_in = keras.layers.Input(shape=(1,), dtype='int64', name='movie_in')

m = keras.layers.Embedding(
    n_movies,
    n_factors,
    input_length=1,
    W_regularizer=keras.regularizers.l2(1e-4)
)(movie_in)


# In[151]:

user_in = keras.layers.Input(shape=(1, ), dtype='int64', name='user_in')
u = keras.layers.Embedding(
    n_users,
    n_factors,
    input_length=1,
    W_regularizer=keras.regularizers.l2(1e-4))(user_in)
movie_in = keras.layers.Input(shape=(1, ), dtype='int64', name='movie_in')
m = keras.layers.Embedding(
    n_movies,
    n_factors,
    input_length=1,
    W_regularizer=keras.regularizers.l2(1e-4))(movie_in)


# In[161]:

x = keras.layers.merge([u, m], mode='dot')
x = keras.layers.Flatten()(x)
model = keras.models.Model([user_in, movie_in], x)
model.compile(keras.optimizers.Adam(0.001), loss='mse')


# In[162]:

model.fit([trn.userId, trn.movieId], trn.rating, batch_size=64, nb_epoch=1, 
          validation_data=([val.userId, val.movieId], val.rating))


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[165]:

user_in = keras.layers.Input(shape=(1, ), dtype='int64', name='user_in')

u = keras.layers.Embedding( 
    input_dim=n_users, 
    output_dim=n_factors,
    input_length=1,    
    W_regularizer=keras.regularizers.l2(l=1e-4)
)(user_in)

movie_in = keras.layers.Input(shape=(1,), dtype='int64', name='movie_in')

m = keras.layers.Embedding(
    input_dim=n_movies,
    output_dim=n_factors,
    input_length=1,
    W_regularizer=keras.regularizers.l2(l=1e-4)
)(movie_in)


# In[166]:

x = keras.layers.merge([u, m], mode='dot')

x = keras.layers.Flatten()(x)

model = keras.models.Model(input=[user_in, movie_in], output=x)

model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='mse')


# In[167]:

model.fit(x=[trn.userId, trn.movieId],
          y=trn.rating,
          batch_size=64,
          nb_epoch=1,
          validation_data=([val.userId, val.movieId], val.rating))


# In[168]:

model.optimizer.lr = 0.01


# In[172]:

model.fit(x=[trn.userId, trn.movieId],
          y=trn.rating,
          batch_size=64,
          nb_epoch=3,
          validation_data=([val.userId, val.movieId], val.rating)
         )


# In[173]:

model.optimizer.lr = 0.001


# In[174]:

model.fit(x=[trn.userId, trn.movieId],
          y=trn.rating,
          batch_size=64,
          nb_epoch=6,
          validation_data=([val.userId, val.movieId], val.rating)
         )


# # Bias

# In[ ]:




# # scratch

# In[ ]:




# In[ ]:




# In[ ]:



