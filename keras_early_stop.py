
# coding: utf-8

# In[13]:

from ds_utils.imports import *


# In[14]:

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# see https://github.com/yang-zhang/code-data-science/blob/master/numpy_newaxis.ipynb
X_train = X_train[:, np.newaxis]
X_test = X_test[:, np.newaxis]

y_train = keras.utils.np_utils.to_categorical(y_train, 10)
y_test = keras.utils.np_utils.to_categorical(y_test, 10)


# In[15]:

def make_compile_model():
    model = keras.models.Sequential([
        keras.layers.Convolution2D(
            filters=32,
            kernel_size=(3, 3),
            activation='relu',
            input_shape=(1, 28, 28)), keras.layers.Convolution2D(
                filters=32, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.25), keras.layers.Flatten(), keras.layers.Dense(
            128, activation='relu'), keras.layers.Dropout(0.5),
        keras.layers.Dense(
            10, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(lr=0.001),
        loss=keras.losses.categorical_crossentropy,
        metrics=['accuracy'])
    return model


# In[16]:

def sample_xy(x, y, sample_size):
    sample = np.random.choice(range(x.shape[0]), sample_size)
    return x[sample], y[sample]


# In[17]:

X_train_sample, y_train_sample = sample_xy(X_train, y_train, 100)
X_test_sample, y_test_sample = sample_xy(X_test, y_test, 500)


# In[18]:

model = make_compile_model()
model.fit(X_train_sample,
          y_train_sample,
          validation_data=[X_test_sample, y_test_sample],
          epochs=1000)


# In[19]:

model = make_compile_model()
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=100)
model.fit(X_train_sample,
          y_train_sample,
          validation_data=[X_test_sample, y_test_sample],
          epochs=1000,
          callbacks=[early_stopping])


# ## References
# - https://keras.io/callbacks/#earlystopping/

# In[ ]:



