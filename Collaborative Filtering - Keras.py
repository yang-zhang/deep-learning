
# coding: utf-8

# In[20]:

from ds_utils.imports import *


# # Prework

# Following this fast.ai [lesson](https://www.youtube.com/watch?v=V2h3IOBDvrA&feature=youtu.be&t=5761).

# In[ ]:

# get data
# !wget -O data/ml-latest-small.zip http://files.grouplens.org/datasets/movielens/ml-latest-small.zip
# !unzip data/ml-latest-small.zip


# # Preprocessing

# In[38]:

ratings = pd.read_csv('data/ml-latest-small/ratings.csv')


# In[39]:

ratings.sample(5)


# In[45]:

ratings.describe()


# In[41]:

ratings.shape


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


# # excel like table

# In[60]:

g = ratings.groupby('userId')['rating'].count()


# In[62]:

topUsers = g.sort_values(ascending=False)[:15]


# In[65]:

g = ratings.groupby('movieId')['rating'].count()


# In[66]:

topMovies = g.sort_values(ascending=False)[:15]


# In[72]:

top_r = ratings.join(topUsers, how='inner', on='userId', rsuffix='_r')


# In[74]:

top_r = top_r.join(topMovies, how='inner', on='movieId', rsuffix='_r')


# In[76]:

top_r.head()


# In[79]:

pd.crosstab(ratings.userId, ratings.movieId, ratings.rating, aggfunc=sum)


# In[77]:

pd.crosstab(top_r.userId, top_r.movieId, top_r.rating, aggfunc=np.sum)


# # dot product

# In[ ]:



