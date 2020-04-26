#!/usr/bin/env python
# coding: utf-8

# In[7]:


from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.font_manager as font_manager
from sklearn.cluster import KMeans

import sys
sys.path.append('..') #('')の中に探索元のpathを書く
import utils.FontUtil


# In[8]:


plt.figure(figsize = (10,10))

X, y = make_blobs(random_state=1)
plt.subplot(3,3,1)
plt.scatter(X[:,0], X[:,1], color='black')
plt.title('random state 1')

X, y = make_blobs(random_state=5)
plt.subplot(3,3,2)
plt.scatter(X[:,0], X[:,1], color='black')
plt.title('random state 5')

X, y = make_blobs(random_state=10)
plt.subplot(3,3,3)
plt.scatter(X[:,0], X[:,1], color='black')
plt.title('random state 10')

X, y = make_blobs(random_state=15)
plt.subplot(3,3,4)
plt.scatter(X[:,0], X[:,1], color='black')
plt.title('random state 15')

X, y = make_blobs(random_state=20)
plt.subplot(3,3,5)
plt.scatter(X[:,0], X[:,1], color='black')
plt.title('random state 20')

X, y = make_blobs(random_state=25)
plt.subplot(3,3,6)
plt.scatter(X[:,0], X[:,1], color='black')
plt.title('random state 25')


# In[11]:


plt.figure(figsize=(20,10))
X, y = make_blobs(random_state=20)
plt.subplot(121)
plt.scatter(X[:,0], X[:,1], color='black')
plt.title('random state 20')

dist_list = []
for i in range(1, 10):
    kmeans =  KMeans(init='random', n_clusters = i)
    kmeans.fit(X)
    dist_list.append(kmeans.inertia_)
plt.subplot(122)
plt.plot(range(1,10), dist_list, marker = '+')
plt.xlabel("number of cluster")
plt.ylabel("Distortion")

# annotate (arrow)
circle_rad = 15
plt.annotate('ELBO', xy=(3, dist_list[2]), xytext=(1,100), textcoords='offset points', color='r', arrowprops=dict(arrowstyle='simple,tail_width=0.3,head_width=0.8,head_length=0.8', shrinkB=circle_rad*1.2))

# circle
bbox_props = dict(boxstyle="circle,pad=0.3", fc="cyan", ec="b", lw=2, alpha = 0.1)
t = plt.text(3, dist_list[2], " ", ha="center", va="center", rotation=45,
            size=15,
            bbox=bbox_props)


# In[10]:


X, y = make_blobs(random_state=25)

dist_list = []
for i in range(1, 10):
    kmeans =  KMeans(init='random', n_clusters = i)
    kmeans.fit(X)
    dist_list.append(kmeans.inertia_)
    
plt.plot(range(1,10), dist_list, marker = '+')
plt.xlabel("number of cluster")
plt.ylabel("Distortion")


# In[2]:


import pandas as pd
kmeans =  KMeans(init='random', n_clusters = 3)
kmeans.fit(X)
clusters = kmeans.predict(X)
df = pd.DataFrame(X, columns = ['X','Y'])
df['Cluster'] = clusters


# In[ ]:


df_0 = df[df['Cluster'] == 0]
df_1 = df[df['Cluster'] == 1]
df_2 = df[df['Cluster'] == 2]

plt.scatter(df_0["X"], df_0["Y"], color = 'blue', label='cluster 0')
plt.scatter(df_1["X"], df_1["Y"], color = 'red', label='cluster 1')
plt.scatter(df_2["X"], df_2["Y"], color = 'green', label='cluster 2')

plt.legend(loc = 'upper left')
plt.show()


# In[ ]:




