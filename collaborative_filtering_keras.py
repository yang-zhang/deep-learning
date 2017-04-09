
# coding: utf-8

# In[4]:

from ds_utils.imports import *


# In[5]:

import keras.layers


# # Prework

# Following this fast.ai [lesson](https://github.com/fastai/courses/blob/master/deeplearning1/nbs/lesson4.ipynb) and [video](https://www.youtube.com/watch?v=V2h3IOBDvrA&feature=youtu.be&t=5761).

# In[6]:

# get data
# !wget -O data/ml-latest-small.zip http://files.grouplens.org/datasets/movielens/ml-latest-small.zip
# !unzip data/ml-latest-small.zip


# # Preprocessing

# In[7]:

ratings = pd.read_csv('data/ml-latest-small/ratings.csv')


# In[8]:

ratings.sample(5)


# In[42]:

users = ratings.userId.unique()
movies = ratings.movieId.unique()


# In[43]:

userid2idx = {o:i for i,o in enumerate(users)}
movieid2idx = {o:i for i,o in enumerate(movies)}


# In[44]:

ratings.userId = ratings.userId.apply(lambda x: userid2idx[x])
ratings.movieId = ratings.movieId.apply(lambda x: movieid2idx[x])


# In[46]:

n_users = ratings.userId.nunique()
n_movies = ratings.movieId.nunique()


# In[49]:

n_factors = 50


# In[50]:

np.random.seed = 42


# In[55]:

msk = np.random.rand(len(ratings))<0.8
trn = ratings[msk]
val = ratings[~msk]


# # dot product

# In[ ]:




# In[2]:

keras.layers.Input


# # scratch
