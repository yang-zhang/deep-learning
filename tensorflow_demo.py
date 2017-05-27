
# coding: utf-8

# In[1]:

import tensorflow as tf


# In[11]:

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node1, node2


# In[10]:

sess = tf.Session()
sess.run([node1, node2])


# In[13]:

node3 = tf.add(node1, node2)
node3


# In[14]:

sess.run(node3)


# In[15]:

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b


# In[18]:

sess.run(adder_node, {a: 3, b: 4.5})


# In[19]:

sess.run(adder_node, {a: [1, 3], b:[2, 4]})


# In[20]:

add_and_triple = adder_node * 3.


# In[21]:

sess.run(add_and_triple, {a: 3, b: 4.5})


# In[22]:

W = tf.Variable([.3], tf.float32)


# In[23]:

b = tf.Variable([-.3], tf.float32)


# In[24]:

x = tf.placeholder(tf.float32)


# In[25]:

linear_model = W * x + b


# In[26]:

init = tf.global_variables_initializer()
sess.run(init)


# In[27]:

sess.run(linear_model, {x:[1, 2, 3, 4]})


# In[28]:

y = tf.placeholder(tf.float32)


# In[29]:

squared_deltas = tf.square(linear_model - y)


# In[30]:

loss = tf.reduce_sum(squared_deltas)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# - https://www.tensorflow.org/get_started/get_started
