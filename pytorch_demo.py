
# coding: utf-8

# In[ ]:




# ### Basics

# In[1]:

import torch


# In[5]:

x = torch.Tensor(5, 3)


# In[6]:

x


# In[7]:

x.size()


# In[8]:

x, y = torch.rand(5, 3), torch.rand(5, 3)


# In[9]:

x


# In[10]:

y


# In[11]:

x + y


# In[12]:

torch.add(x, y)


# In[13]:

result = torch.Tensor(5, 3)
torch.add(x, y, out=result)


# In[14]:

result


# In[15]:

y.add_(x)


# In[16]:

y


# In[17]:

torch.cuda.is_available()


# In[18]:

#x.cuda()


# In[19]:

from torch.autograd import Variable


# In[20]:

x = Variable(torch.randn(1, 10))


# ### Numpy bridge

# In[21]:

a = torch.ones(5)
b = a.numpy()
a, b


# In[22]:

a.add_(1)
a, b


# In[27]:

import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
a, b


# In[29]:

c = torch.ones(5)
c


# In[30]:

c.add_(10)
c


# In[32]:

torch.add(c, 10, out=c)
c


# ### Autograd

# In[2]:

import torch
from torch.autograd import Variable


# #### Case

# In[3]:

x = Variable(torch.ones(2)*3, requires_grad=True)
x, x.creator


# In[4]:

y = x[0] + x[1]
y, y.creator


# In[5]:

y.backward()


# In[6]:

x.data, x.grad


# In[7]:

y.data, y.grad


# #### Case

# In[8]:

x = Variable(torch.ones(2)*3, requires_grad=True)
x, x.creator


# In[9]:

y = x[0]*2 + x[1]*3
y, y.creator


# In[10]:

y.backward()


# In[11]:

x.data, x.grad


# In[12]:

y.data, y.grad


# #### Case

# In[13]:

x = Variable(torch.ones(2)*3, requires_grad=True)
x, x.creator


# In[14]:

y = x[0]**2 + x[1]**3
y, y.creator


# In[15]:

y.backward()


# In[16]:

x.data, x.grad


# In[17]:

y.data, y.grad


# #### Case - multiple outputs

# In[90]:

x = Variable(torch.FloatTensor([1, 1]), requires_grad=True)
x, x.creator


# In[91]:

y = 5 * x **2
y, y.creator


# ```
# y.backward()
# RuntimeError: backward should be called only on a scalar (i.e. 1-element tensor) or with gradient w.r.t. the variable
# ```

# In[87]:

gradients = torch.FloatTensor([1.0, 2.0])


# In[88]:

y.backward(gradients)


# In[89]:

x.grad


# In[92]:

gradients = torch.FloatTensor([100.0, 200.0])


# In[93]:

y.backward(gradients)


# In[94]:

x.grad


# Not sure why `gradients` is needed.

# #### Case

# In[18]:

x = Variable(torch.ones(2), requires_grad=True)
x, x.creator


# In[19]:

y = 2 * x
y, y.creator


# In[20]:

z = 3 * y**2


# In[21]:

out = z.mean()


# In[22]:

out.backward()


# In[23]:

x.data, x.grad


# In[24]:

y.data, y.grad


# In[25]:

z.data, z.grad


# $$
# o = (z_0 + z_1) / 2 = (3y_0^2 + 3y_1^2) /2 = (3(2x_0)^2 + 3(2x_1)^2) / 2 = 6(x_0^2 + x_1^2)
# $$
# 
# $$
# \left.
# \frac{\partial o}{\partial x_0}
# \right|_{x_0=1} = 
# \left.\frac{\partial 6(x_0^2 + x_1^2)}{\partial x_0}\right|_{x_0=1}=\left.12x_0\right|_{x_0=1} = 12
# $$

# #### Crazy case

# In[27]:

x = Variable(torch.randn(3), requires_grad=True)


# In[37]:

x


# In[29]:

y = x*2


# In[31]:

while y.data.norm() < 1000:
    y = y*2


# In[32]:

y


# In[34]:

gradients = torch.FloatTensor([0.1, 1.0, 0.0001])


# In[35]:

y.backward(gradients)


# In[ ]:

y.backward()


# In[36]:

x.grad


# ### Neural networks

# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# ### References
# - http://pytorch.org/tutorials/index.html

# In[ ]:



