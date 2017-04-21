
# coding: utf-8

# In[79]:

import numpy as np
import theano
import theano.tensor as T


# ### [First look](http://deeplearning.net/software/theano/tutorial/adding.html#)

# #### Exercise

# In[50]:

a = T.ivector()


# In[51]:

b = T.vector()


# In[ ]:

out = a**2 + b**2 + 2 * a * b


# In[53]:

f = theano.function([a, b], out)


# In[54]:

f([1, 2], [3., 4.])


# In[62]:

a.type


# ### [pp](http://deeplearning.net/software/theano/tutorial/gradients.html)

# In[218]:

a = T.ivector()
b = T.vector()
out = a + b
theano.pp(out)


# In[216]:

x = T.dscalar('x')
y = x ** 2
gy = T.grad(y, x)


# In[217]:

theano.pp(gy)


# ```
# (
#     (
#         fill(
#             (x ** TensorConstant{2}), TensorConstant{1.0}
#         )
#         * 
#         TensorConstant{2}
#     ) 
#     * 
#     (
#         x ** (TensorConstant{2} - TensorConstant{1})
#     )
# )
# ```

# In[222]:

theano.pp(f.maker.fgraph.outputs[0])


# ### [More examples](http://deeplearning.net/software/theano/tutorial/examples.html)

# #### Random Numbers

# In[112]:

srng = theano.tensor.shared_randomstreams.RandomStreams()


# In[121]:

type(srng)


# In[113]:

rv_u = srng.uniform((2, 2))


# In[115]:

rv_n = srng.normal((2, 2))


# In[116]:

f = theano.function([], rv_u)


# In[128]:

f()


# In[145]:

f()


# In[136]:

g = theano.function([], rv_n, no_default_updates=True)


# In[143]:

g()


# In[144]:

g()


# In[166]:

nearly_zeros = theano.function([], rv_u + rv_u - 2 * rv_u)
nearly_zeros()


# In[148]:

type(rv_n.rng)


# In[203]:

rng_val = rv_u.rng.get_value(borrow=True)


# In[162]:

type(rng_val)


# In[163]:

rng_val.seed(234)


# In[164]:

rv_u.rng.set_value(rng_val, borrow=True)


# In[165]:

srng.seed(213)


# ##### sharing Streams between Functions

# In[168]:

state_after_v0 = rv_u.rng.get_value().get_state()


# In[171]:

type(state_after_v0)


# In[172]:

nearly_zeros()


# In[173]:

v1 = f()


# In[175]:

v1


# In[176]:

rng = rv_u.rng.get_value(borrow=True)


# In[177]:

rng.set_state(state_after_v0)


# In[179]:

rv_u.rng.set_value(rng, borrow=True)


# In[180]:

v2 = f()


# In[181]:

v2


# In[182]:

v3 = f()


# In[186]:

v3 # v3 == v1


# #### [A Real Example: Logistic Regression](http://deeplearning.net/software/theano/tutorial/examples.html#a-real-example-logistic-regression)

# In[ ]:

# TODO


# ### [Loop](http://deeplearning.net/software/theano/tutorial/loop.html#loop)
# - https://github.com/lamblin/ccw_tutorial/blob/master/Scan_W2016/scan_tutorial.ipynb

# In[223]:

v1 = T.vector()
v2 = T.vector()


# In[224]:

v1


# In[ ]:

theano.scan()


# ### [graph](http://deeplearning.net/software/theano/extending/graphstructures.html#tutorial-graphfigure)

# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# ### [Basic Tensor Functionality](http://deeplearning.net/software/theano/library/tensor/basic.html#libdoc-basic-tensor)

# #### TensorType v.s. TensorVariable

# In[102]:

x = T.iscalar('some_var')


# In[103]:

type(x)


# In[93]:

t = T.TensorType(dtype='int32', broadcastable=(False, False))


# In[94]:

type(t)


# In[95]:

x = t('x')


# In[96]:

type(x)


# #### shared

# In[ ]:

x = theano.shared([[1, 2], [3, 4]])


# In[98]:

type(x)


# In[97]:

x.get_value()


# In[82]:

x = theano.shared(np.array([[1, 2], [3, 4]]))
x.get_value()


# #### value of TensorVariable

# In[104]:

x = T.imatrix()


# In[105]:

type(x)


# In[ ]:



