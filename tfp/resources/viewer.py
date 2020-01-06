#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# In[20]:


dat = np.load("01_01.npy")


# In[34]:


fig = plt.figure()
ax = Axes3D(fig)
x=[]
y=[]
z=[]
for counter,i in enumerate(dat[0]):
    x.append(i[0])
    y.append(i[1])
    z.append(i[2])
    ax.text(i[2],i[0],i[1],  '%s' % (str(counter)), size=20, zorder=1,
 color='k')
plt.plot(z,x,y,"b.")
plt.show()


# In[ ]:
