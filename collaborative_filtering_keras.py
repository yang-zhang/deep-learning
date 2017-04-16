
# coding: utf-8

# In[52]:

from ds_utils.imports import *


# In[57]:

assert(keras.backend.backend()=='theano')


# # Prework

# Following this fast.ai [lesson](https://github.com/fastai/courses/blob/master/deeplearning1/nbs/lesson4.ipynb) and [video](https://www.youtube.com/watch?v=V2h3IOBDvrA&feature=youtu.be&t=5761).

# In[58]:

# get data
# !wget -O data/ml-latest-small.zip http://files.grouplens.org/datasets/movielens/ml-latest-small.zip
# !unzip data/ml-latest-small.zip


# # Preprocessing

# In[59]:

ratings = pd.read_csv('data/ml-latest-small/ratings.csv')


# In[60]:

ratings.shape


# In[61]:

ratings.sample(5)


# In[62]:

users = ratings.userId.unique()
movies = ratings.movieId.unique()


# In[63]:

userid2idx = {o: i for i, o in enumerate(users)}
movieid2idx = {o: i for i, o in enumerate(movies)}


# In[64]:

ratings.userId = ratings.userId.apply(lambda x: userid2idx[x])
ratings.movieId = ratings.movieId.apply(lambda x: movieid2idx[x])


# In[65]:

n_users = ratings.userId.nunique()
n_movies = ratings.movieId.nunique()


# In[66]:

n_factors = 50


# In[67]:

np.random.seed = 42


# In[68]:

msk = np.random.rand(len(ratings)) < 0.8
trn = ratings[msk]
val = ratings[~msk]


# # Dot product

# In[22]:

user_in = keras.layers.Input(shape=(1, ), dtype='int64', name='user_in')


# In[23]:

u = keras.layers.Embedding(
    input_dim=n_users,
    output_dim=n_factors,
    input_length=1,
    embeddings_regularizer=keras.regularizers.l2(l=1e-4))(user_in)


# In[24]:

movie_in = keras.layers.Input(shape=(1, ), dtype='int64', name='movie_in')


# In[25]:

m = keras.layers.Embedding(
    input_dim=n_movies,
    output_dim=n_factors,
    input_length=1,
    embeddings_regularizer=keras.regularizers.l2(l=1e-4))(movie_in)


# In[26]:

x = keras.layers.merge([u, m], mode='dot')


# In[27]:

x = keras.layers.Flatten()(x)


# In[28]:

model = keras.models.Model(inputs=[user_in, movie_in], outputs=x)


# In[29]:

model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='mse')


# In[30]:

model.fit(x=[trn.userId, trn.movieId],
          y=trn.rating,
          batch_size=64,
          epochs=1,
          validation_data=([val.userId, val.movieId], val.rating))


# In[31]:

model.optimizer.lr = 0.01


# In[32]:

model.fit(
    x=[trn.userId, trn.movieId],
    y=trn.rating,
    batch_size=64,
    epochs=3,
    validation_data=([val.userId, val.movieId], val.rating), )


# In[33]:

model.optimizer.lr = 0.001


# In[34]:

model.fit(x=[trn.userId, trn.movieId],
          y=trn.rating,
          batch_size=64,
          epochs=6,
          validation_data=([val.userId, val.movieId], val.rating),
          shuffle=False)


# # Bias

# In[69]:

user_bias = keras.layers.Embedding(input_dim=n_users, output_dim=1, input_length=1)(user_in)
user_bias = keras.layers.Flatten()(user_bias)


# In[37]:

movie_bias = keras.layers.Embedding(input_dim=n_movies, output_dim=1, input_length=1)(movie_in)
movie_bias = keras.layers.Flatten()(movie_bias)


# In[38]:

x = keras.layers.merge([u, m], mode='dot')
x = keras.layers.Flatten()(x)


# In[39]:

x = keras.layers.merge([x, user_bias], mode='sum')
x = keras.layers.merge([x, movie_bias], mode='sum')


# In[79]:

type(x)


# In[40]:

model = keras.models.Model(inputs=[user_in, movie_in], outputs=x)


# In[41]:

model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='mse')


# In[42]:

model.summary()


# In[43]:

model.fit(x=[trn.userId, trn.movieId], y=trn.rating, batch_size=64, validation_data=([val.userId, val.movieId], val.rating))


# In[44]:

model.optimizer.lr=0.01


# In[45]:

model.fit(x=[trn.userId, trn.movieId], y=trn.rating, batch_size=64, epochs=10, validation_data=([val.userId, val.movieId], val.rating))


# In[46]:

model.optimizer.lr=0.001


# In[47]:

model.fit(x=[trn.userId, trn.movieId], y=trn.rating, batch_size=64, epochs=5, validation_data=([val.userId, val.movieId], val.rating))


# # Inspect

# In[92]:

type(user_in)


# In[93]:

user_in_layer = model.get_layer(name='user_in')


# In[94]:

user_in_layer.input_shape


# In[95]:

user_in_layer.output_shape


# In[96]:

model.get_layer(index=2).output_shape


# # NN

# In[133]:

x = keras.layers.merge([u, m], mode='concat')

x = keras.layers.Flatten()(x)
x = keras.layers.Dropout(0.3)(x)
x = keras.layers.Dense(70, activation='relu')(x)
x = keras.layers.Dropout(0.75)(x)
x = keras.layers.Dense(1)(x)


# In[138]:

nn = keras.models.Model(inputs=[user_in, movie_in], outputs=x)


# In[140]:

nn.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='mse')


# In[ ]:

nn.fit(x=[trn.userId, trn.movieId],
          y=trn.rating,
          batch_size=64,
          epochs=10,
          validation_data=([val.userId, val.movieId], val.rating))


# # Get Parts of Model

# ## Get bias

# In[157]:

mdl_movie_bias = keras.models.Model(inputs=movie_in, outputs=movie_bias)


# In[158]:

mdl_movie_bias.summary()


# In[159]:

movies


# In[160]:

mdl_movie_bias.predict(np.random.choice(ratings.movieId, 5))


# ## Get embedding

# In[161]:

mdl_movie_embedding = keras.models.Model(inputs=movie_in, outputs=m)


# In[162]:

mdl_movie_embedding.summary()


# In[165]:

mdl_movie_embedding.predict(np.random.choice(ratings.movieId, 2))

