
# coding: utf-8

# In[109]:

from ds_utils.imports import *


# In[110]:

from imp import reload
import ds_utils.misc; reload(ds_utils.misc)


# ### Regular model using all training data

# In[111]:

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()


# In[112]:

# see https://github.com/yang-zhang/code-data-science/blob/master/numpy_newaxis.ipynb
X_train = X_train[:, np.newaxis]
X_test = X_test[:, np.newaxis]


# In[113]:

y_train = keras.utils.np_utils.to_categorical(y_train, 10)
y_test = keras.utils.np_utils.to_categorical(y_test, 10)


# In[114]:

X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[115]:

ds_utils.misc.imshow_gray(X_train[0][0])


# In[116]:

y_train[0]


# In[117]:

def make_compile_model():
    model = keras.models.Sequential([
        keras.layers.Convolution2D(
            filters=32,
            kernel_size=(3, 3),
            activation='relu',
            input_shape=(1, 28, 28)),
        keras.layers.Convolution2D(
            filters=32, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.25),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(lr=0.001),
        loss=keras.losses.categorical_crossentropy,
        metrics=['accuracy'])
    return model


# In[94]:

model = make_compile_model()
model.fit(X_train, y_train, validation_data=[X_test, y_test], epochs=1)


# ### Suppose we only have a smaller training set.

# In[124]:

train_small = np.random.choice(range(X_train.shape[0]),100)


# In[125]:

X_train_small, y_train_small = X_train[train_small], y_train[train_small]


# In[126]:

X_train_small.shape, y_train_small.shape, X_test.shape, y_test.shape


# Performance is worse on smaller data as expected.

# In[127]:

model = make_compile_model()
model.fit(X_train_small, y_train_small, validation_data=[X_test, y_test], epochs=50)


# ### Psudo labeling

# In[128]:

X_pseudo = X_test
y_pseudo = model.predict(X_test)


# In[ ]:

X_comb_pseudo = np.concatenate([X_train, X_pseudo])
y_comb_pseudo = np.concatenate([y_train, y_pseudo])


# In[ ]:

model.fit(X_comb_pseudo, y_comb_pseudo, validation_data=[X_test, y_test], epochs=5)


# Ref: 
# - https://github.com/yang-zhang/deep-learning/blob/master/MNIST_keras.ipynb
# - https://github.com/yang-zhang/courses/blob/master/deeplearning1/nbs/statefarm.ipynb
