
# coding: utf-8

# ### Basics

# In[4]:

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

# In[56]:

import torch
from torch.autograd import Variable


# #### Case

# In[71]:

x = Variable(torch.ones(2)*3, requires_grad=True)
x, x.creator


# In[62]:

y = x[0] + x[1]
y, y.creator


# In[63]:

y.backward()


# In[64]:

x.data, x.grad


# In[65]:

y.data, y.grad


# #### Case

# In[66]:

x = Variable(torch.ones(2)*3, requires_grad=True)
x, x.creator


# In[67]:

y = x[0]*2 + x[1]*3
y, y.creator


# In[68]:

y.backward()


# In[69]:

x.data, x.grad


# In[70]:

y.data, y.grad


# #### Case

# In[76]:

x = Variable(torch.ones(2)*3, requires_grad=True)
x, x.creator


# In[77]:

y = x[0]**2 + x[1]**3
y, y.creator


# In[78]:

y.backward()


# In[79]:

x.data, x.grad


# In[80]:

y.data, y.grad


# #### Case

# In[106]:

x = Variable(torch.ones(2), requires_grad=True)
x, x.creator


# In[107]:

y = 2 * x
y, y.creator


# ```
# y.backward()
# RuntimeError: backward should be called only on a scalar (i.e. 1-element tensor) or with gradient w.r.t. the variable
# ```

# In[108]:

z = 3 * y**2


# In[109]:

out = z.mean()


# In[110]:

out.backward()


# In[111]:

x.data, x.grad


# In[112]:

y.data, y.grad


# In[113]:

z.data, z.grad


# $$
# o = (z_0 + z_1) / 2 = (3y_0^2 + 3y_1^2) /2 = (3(2x_0)^2 + 3(2x_1)^2) / 2 = 6(x_0^2 + x_1^2)
# $$
# $$
# \left.
# \frac{\partial o}{\partial x_0}
# \right|_{x_0=1} = 
# \left.\frac{\partial 6(x_0^2 + x_1^2)}{\partial x_0}\right|_{x_0=1}=\left.12x_0\right|_{x_0=1} = 12
# $$

# ### References
# - http://pytorch.org/tutorials/index.html

# In[ ]:



