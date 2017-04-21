
# coding: utf-8

# In[1]:

from ds_utils.imports import *


# In[2]:

seasons = np.arange(1, 20)

data_seasons = []
for season in seasons:
    data_season = pd.read_csv('https://github.com/BobAdamsEE/SouthParkData/raw/master/Season-%d.csv' % season)
    data_seasons.append(data_season)

data = pd.concat(data_seasons)


# In[3]:

text = ' '.join([line[:-1] for line in data.Line])
print(text[:1000])


# In[4]:

chars = list(set(text))
vocab_size = len(chars)+1

char_ind = dict((i, c) for i, c in enumerate(chars))
ind_char = dict((c, i) for i, c in enumerate(chars))

idx = [ind_char[c] for c in text]


# In[5]:

cs = 8


# In[6]:

c_in_dat = [[idx[i+n] for i in range(0, len(idx)-1-cs, cs)]
            for n in range(cs)]


# In[7]:

c_out_dat = [idx[i+cs] for i in range(0, len(idx)-1-cs, cs)]


# In[8]:

xs = [np.stack(c[:-2]) for c in c_in_dat]


# In[9]:

y = np.stack(c_out_dat[:-2])


# In[10]:

n_fac = 42


# In[11]:

def embedding_input(name, n_in, n_out):
    inp = keras.layers.Input(shape=(1,), dtype='int64', name=name+'_in')
    emb = keras.layers.Embedding(n_in, n_out, input_length=1, name=name+'_emb')(inp)
    return inp, keras.layers.Flatten()(emb)


# In[12]:

c_ins = [embedding_input('c'+str(n), vocab_size, n_fac) for n in range(cs)]


# In[13]:

n_hidden = 256


# In[14]:

dense_in = keras.layers.Dense(n_hidden, activation='relu')
dense_hidden = keras.layers.Dense(n_hidden, activation='relu', init='identity')
dense_out = keras.layers.Dense(vocab_size, activation='softmax')


# In[15]:

hidden = dense_in(c_ins[0][1])


# In[16]:

for i in range(1,cs):
    c_dense = dense_in(c_ins[i][1])
    hidden = dense_hidden(hidden)
    hidden = keras.layers.merge([c_dense, hidden])


# In[17]:

c_out = dense_out(hidden)


# In[29]:

model = keras.models.Model([c[0] for c in c_ins], c_out)
model.compile(
    loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam())


# In[19]:

model.fit(xs, y, batch_size=64, epochs=12)


# In[72]:

model_weight_path = 'models/southpark_rnn_weights.h5'


# In[73]:

model.save_weights(model_weight_path)



# In[75]:

model.load_weights(model_weight_path)


# In[76]:

def get_next(inp):
    idxs = [np.array(ind_char[c])[np.newaxis] for c in inp]
    p = model.predict(idxs)
    return chars[np.argmax(p)]


# In[77]:

for i in range(100):
    n = np.random.choice(len(text))
    str_piece = text[n:n + 8]
    str_piece_context = text[n - 10:n + 8 + 10]
    print(str_piece_context)
    print(10 * '.' + str_piece + str(get_next(str_piece)))
    print(50 * '-')


# References:
# - https://github.com/yang-zhang/courses/blob/master/deeplearning1/nbs/lesson6.ipynb
# - https://github.com/yang-zhang/courses/blob/master/deeplearning1/nbs/char-rnn.ipynb
