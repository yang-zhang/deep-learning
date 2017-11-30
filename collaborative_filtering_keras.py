
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import keras


# # Prework

# Following this fast.ai [lesson](https://github.com/fastai/courses/blob/master/deeplearning1/nbs/lesson4.ipynb) and [video](https://www.youtube.com/watch?v=V2h3IOBDvrA&feature=youtu.be&t=5761).

# In[2]:

# get data
# !wget -O data/ml-latest-small.zip http://files.grouplens.org/datasets/movielens/ml-latest-small.zip
# !unzip data/ml-latest-small.zip


# # Preprocessing

# In[3]:

path = '/opt/notebooks/data/movielens/ml-latest-small/'


# In[54]:

ratings = pd.read_csv(path+'ratings.csv')


# In[55]:

ratings.shape


# In[56]:

ratings.head(5)


# In[57]:

users = ratings.userId.unique()
movies = ratings.movieId.unique()


# In[58]:

userid2idx = {o: i for i, o in enumerate(users)}
movieid2idx = {o: i for i, o in enumerate(movies)}


# In[59]:

ratings.userId = ratings.userId.apply(lambda x: userid2idx[x])
ratings.movieId = ratings.movieId.apply(lambda x: movieid2idx[x])


# In[60]:

ratings.head(5)


# In[61]:

n_users = ratings.userId.nunique()
n_movies = ratings.movieId.nunique()


# In[62]:

n_factors = 50


# In[63]:

np.random.seed = 42


# In[64]:

msk = np.random.rand(len(ratings)) < 0.8
trn = ratings[msk]
val = ratings[~msk]


# # Dot product

# In[65]:

user_in = keras.layers.Input(shape=(1, ), dtype='int64', name='user_in')


# In[66]:

u = keras.layers.Embedding(
    input_dim=n_users,
    output_dim=n_factors,
    input_length=1,
    embeddings_regularizer=keras.regularizers.l2(l=1e-4))(user_in)


# In[67]:

movie_in = keras.layers.Input(shape=(1, ), dtype='int64', name='movie_in')


# In[68]:

m = keras.layers.Embedding(
    input_dim=n_movies,
    output_dim=n_factors,
    input_length=1,
    embeddings_regularizer=keras.regularizers.l2(l=1e-4))(movie_in)


# In[69]:

x = keras.layers.dot([u, m], axes=[2,2])


# In[70]:

x.shape


# In[71]:

x = keras.layers.Flatten()(x)


# In[72]:

x.shape


# In[73]:

model = keras.models.Model(inputs=[user_in, movie_in], outputs=x)


# In[74]:

model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='mse')


# In[26]:

model.fit(x=[trn.userId, trn.movieId],
          y=trn.rating,
          batch_size=64,
          epochs=1,
          validation_data=([val.userId, val.movieId], val.rating))


# In[27]:

model.optimizer.lr = 0.01


# In[28]:

model.fit(
    x=[trn.userId, trn.movieId],
    y=trn.rating,
    batch_size=64,
    epochs=3,
    validation_data=([val.userId, val.movieId], val.rating), )


# In[29]:

model.optimizer.lr = 0.001


# In[30]:

model.fit(x=[trn.userId, trn.movieId],
          y=trn.rating,
          batch_size=64,
          epochs=6,
          validation_data=([val.userId, val.movieId], val.rating),
          shuffle=False)


# # Bias

# In[31]:

user_bias = keras.layers.Embedding(input_dim=n_users, output_dim=1, input_length=1)(user_in)
user_bias = keras.layers.Flatten()(user_bias)


# In[34]:

movie_bias = keras.layers.Embedding(input_dim=n_movies, output_dim=1, input_length=1)(movie_in)
movie_bias = keras.layers.Flatten()(movie_bias)


# In[35]:

# x = keras.layers.merge([u, m], mode='dot')
x = keras.layers.dot([u, m], axes=[2, 2])
x = keras.layers.Flatten()(x)


# In[37]:

x = keras.layers.add([x, user_bias])
x = keras.layers.add([x, movie_bias])
# x = keras.layers.merge([x, user_bias], mode='sum')
# x = keras.layers.merge([x, movie_bias], mode='sum')


# In[39]:

model = keras.models.Model(inputs=[user_in, movie_in], outputs=x)


# In[40]:

model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='mse')


# In[41]:

model.summary()


# In[35]:

model.fit(x=[trn.userId, trn.movieId], y=trn.rating, batch_size=64, validation_data=([val.userId, val.movieId], val.rating))


# In[36]:

model.optimizer.lr=0.01


# In[42]:

model.fit(x=[trn.userId, trn.movieId], y=trn.rating, batch_size=64, epochs=10, validation_data=([val.userId, val.movieId], val.rating))


# In[43]:

model.optimizer.lr=0.001


# In[44]:

model.fit(x=[trn.userId, trn.movieId], y=trn.rating, batch_size=64, epochs=5, validation_data=([val.userId, val.movieId], val.rating))


# # Inspect

# In[45]:

type(user_in)


# In[46]:

user_in_layer = model.get_layer(name='user_in')


# In[47]:

user_in_layer.input_shape


# In[48]:

user_in_layer.output_shape


# In[49]:

model.get_layer(index=2).output_shape


# # NN

# In[50]:

# x = keras.layers.merge([u, m], mode='concat')
x = keras.layers.concatenate([u, m])

x = keras.layers.Flatten()(x)
x = keras.layers.Dropout(0.3)(x)
x = keras.layers.Dense(70, activation='relu')(x)
x = keras.layers.Dropout(0.75)(x)
x = keras.layers.Dense(1)(x)


# In[51]:

nn = keras.models.Model(inputs=[user_in, movie_in], outputs=x)


# In[52]:

nn.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='mse')


# In[53]:

nn.fit(x=[trn.userId, trn.movieId],
          y=trn.rating,
          batch_size=64,
          epochs=10,
          validation_data=([val.userId, val.movieId], val.rating))


# # Get parts of model

# ## Get bias

# In[49]:

mdl_movie_bias = keras.models.Model(inputs=movie_in, outputs=movie_bias)


# In[50]:

mdl_movie_bias.summary()


# In[51]:

movies


# In[52]:

mdl_movie_bias.predict(np.random.choice(ratings.movieId, 5))


# In[53]:

predicted_movies_bias = mdl_movie_bias.predict(ratings.movieId)


# In[54]:

predicted_movies_bias.shape


# In[55]:

predicted_movies_bias[:10]


# In[56]:

model.summary()


# In[57]:

model.layers[8].get_weights()[:10]


# ## Get embedding

# In[58]:

mdl_movie_embedding = keras.models.Model(inputs=movie_in, outputs=m)


# In[59]:

mdl_movie_embedding.summary()


# In[60]:

mdl_movie_embedding.predict(ratings.movieId)[0]


# In[61]:

model.layers[3].get_weights()[0][0]


# In[ ]:



