
# coding: utf-8

# In[71]:

from ds_utils.imports import *


# # Prework

# Following this fast.ai [lesson](https://github.com/fastai/courses/blob/master/deeplearning1/nbs/lesson4.ipynb) and [video](https://www.youtube.com/watch?v=V2h3IOBDvrA&feature=youtu.be&t=5761).

# In[84]:

# get data
# !wget -O data/ml-latest-small.zip http://files.grouplens.org/datasets/movielens/ml-latest-small.zip
# !unzip data/ml-latest-small.zip


# # Preprocessing

# In[85]:

ratings = pd.read_csv('data/ml-latest-small/ratings.csv')


# In[86]:

ratings.shape


# In[87]:

ratings.sample(5)


# In[88]:

users = ratings.userId.unique()
movies = ratings.movieId.unique()


# In[89]:

userid2idx = {o: i for i, o in enumerate(users)}
movieid2idx = {o: i for i, o in enumerate(movies)}


# In[90]:

ratings.userId = ratings.userId.apply(lambda x: userid2idx[x])
ratings.movieId = ratings.movieId.apply(lambda x: movieid2idx[x])


# In[91]:

n_users = ratings.userId.nunique()
n_movies = ratings.movieId.nunique()


# In[92]:

n_factors = 50


# In[93]:

np.random.seed = 42


# In[94]:

msk = np.random.rand(len(ratings)) < 0.8
trn = ratings[msk]
val = ratings[~msk]


# # Dot product

# In[108]:

user_in = keras.layers.Input(shape=(1, ), dtype='int64', name='user_in')


# In[109]:

u = keras.layers.Embedding(
    input_dim=n_users,
    output_dim=n_factors,
    input_length=1,
    embeddings_regularizer=keras.regularizers.l2(l=1e-4))(user_in)


# In[111]:

movie_in = keras.layers.Input(shape=(1, ), dtype='int64', name='movie_in')


# In[112]:

m = keras.layers.Embedding(
    input_dim=n_movies,
    output_dim=n_factors,
    input_length=1,
    embeddings_regularizer=keras.regularizers.l2(l=1e-4))(movie_in)


# In[113]:

x = keras.layers.merge([u, m], mode='dot')


# In[114]:

x = keras.layers.Flatten()(x)


# In[115]:

model = keras.models.Model(inputs=[user_in, movie_in], outputs=x)


# In[116]:

model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='mse')


# In[117]:

model.fit(x=[trn.userId, trn.movieId],
          y=trn.rating,
          batch_size=64,
          epochs=1,
          validation_data=([val.userId, val.movieId], val.rating))


# In[67]:

model.optimizer.lr = 0.01


# In[118]:

model.fit(
    x=[trn.userId, trn.movieId],
    y=trn.rating,
    batch_size=64,
    epochs=3,
    validation_data=([val.userId, val.movieId], val.rating), )


# In[119]:

model.optimizer.lr = 0.001


# In[120]:

model.fit(x=[trn.userId, trn.movieId],
          y=trn.rating,
          batch_size=64,
          epochs=6,
          validation_data=([val.userId, val.movieId], val.rating),
          shuffle=False)


# # Bias

# In[130]:

user_bias = keras.layers.Embedding(input_dim=n_users, output_dim=1, input_length=1)(user_in)
user_bias = keras.layers.Flatten()(user_bias)


# In[131]:

movie_bias = keras.layers.Embedding(input_dim=n_movies, output_dim=1, input_length=1)(movie_in)
movie_bias = keras.layers.Flatten()(movie_bias)


# In[132]:

x = keras.layers.merge([u, m], mode='dot')
x = keras.layers.Flatten()(x)


# In[133]:

x = keras.layers.merge([x, user_bias], mode='sum')
x = keras.layers.merge([x, movie_bias], mode='sum')


# In[151]:

model = keras.models.Model(inputs=[user_in, movie_in], outputs=x)


# In[152]:

model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='mse')


# In[158]:

model.summary()


# In[153]:

model.fit(x=[trn.userId, trn.movieId], y=trn.rating, batch_size=64, validation_data=([val.userId, val.movieId], val.rating))


# In[154]:

model.optimizer.lr=0.01


# In[155]:

model.fit(x=[trn.userId, trn.movieId], y=trn.rating, batch_size=64, epochs=10, validation_data=([val.userId, val.movieId], val.rating))


# In[156]:

model.optimizer.lr=0.001


# In[157]:

model.fit(x=[trn.userId, trn.movieId], y=trn.rating, batch_size=64, epochs=5, validation_data=([val.userId, val.movieId], val.rating))


# # scratch

# In[ ]:




# In[ ]:




# In[ ]:



