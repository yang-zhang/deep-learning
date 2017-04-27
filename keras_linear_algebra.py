
# coding: utf-8

# In[1]:

from ds_utils.imports import *


# ### Dense layer 

# In[2]:

num_inputs = 5
dim_input = 3
dim_output = 2


# In[3]:

x = np.random.random(size=(num_inputs, dim_input))
x


# In[4]:

layer = keras.layers.Dense(
    units=dim_output,
    input_shape=(dim_input, ),  # same as input_dim=dim_input, 
    kernel_initializer='glorot_uniform',
    bias_initializer='glorot_uniform'  # using a non-default bias_initializer for illustration
)


# layer has no weights before added to the model.

# In[5]:

layer.get_weights()


# In[6]:

mdl = keras.models.Sequential()
mdl.add(layer)


# In[7]:

layer.get_weights()


# Notice that I did not compile or fit the model. 

# In[8]:

model_outputs = mdl.predict(x)
model_outputs


# In[9]:

A = layer.get_weights()[0]
b = layer.get_weights()[1]
Ax_plus_b = np.dot(x, A) + b
Ax_plus_b


# In[10]:

assert np.allclose(model_outputs, Ax_plus_b)


# ### Merge

# In[48]:

num_inputs = 5
dim_input = 3


# In[49]:

x1 = np.random.random(size=(num_inputs, dim_input))
x2 = np.random.random(size=(num_inputs, dim_input))
x1, x2


# In[ ]:

inputs1 = np.random.


# In[50]:

in1 = keras.layers.Input(shape=(dim_input,))
in2 = keras.layers.Input(shape=(dim_input,))


# In[51]:

layer_merge = keras.layers.merge([in1, in2], mode='sum')


# In[53]:

model = keras.models.Model(
    inputs=[in1, in2],
    outputs=layer_merge
)


# In[56]:

merged = model.predict([x1, x2])
merged


# In[58]:

x1_plus_x2 = x1 + x2
x1_plus_x2


# In[59]:

np.allclose(merged, x1_plus_x2)


# ### TimeDistributed

# In[2]:

keras.layers.TimeDistributed


# In[3]:

get_ipython().magic('pinfo2 keras.layers.TimeDistributed')


# In[4]:

num_samples = 5
num_timestamps = 10 
num_features = 3


# In[5]:

inputs = np.random.random(size=(num_samples, num_timestamps, num_features))


# In[6]:

model = keras.models.Sequential()


# In[7]:

layer = keras.layers.TimeDistributed(
    keras.layers.Dense(5), input_shape=(num_timestamps, num_features))


# In[8]:

model.add(layer)


# In[9]:

model.output_shape


# In[10]:

weights = layer.get_weights()


# In[11]:

A, b = tuple(weights)


# In[12]:

A, b


# In[13]:

model_outputs = model.predict(inputs)


# In[14]:

model_outputs.shape


# In[15]:

model_outputs[0]


# In[16]:

np.dot(inputs[0], A) + b


# In[17]:

for i in range(num_samples):
    assert np.allclose(model_outputs[0], np.dot(inputs[0], A) + b)

