#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


Sample = np.array([[ -2.32496308,  -6.6999964 ],
       [  0.51856831,  -4.90086804],
       [  2.44301805,   3.84652646],
       [  5.82662285,  -9.92259335],
       [  2.03300209,   5.28990817],
       [  3.37979515,   4.18880872],
       [  6.04774884, -10.30504657],
       [ -0.42084194,  -4.24889336],
       [  3.78067293,   5.22062163],
       [  5.69248303,  -7.19999368],
       [  5.15909568, -10.13427003],
       [  1.16464321,   5.59667831],
       [  2.94601402,   3.3575069 ],
       [  1.1882891 ,  -5.56058781],
       [ -0.31748917,  -6.86337766],
       [  4.32968132,   5.64396726],
       [  4.28981065,  -9.44982413],
       [  3.49996332,   3.02156553],
       [  5.31414039,  -9.94714146],
       [  2.61105267,   4.22218469],
       [  4.88653379,  -8.87680099],
       [  1.95552599,  -4.05690149],
       [  2.09985134,   3.6556301 ],
       [  1.31468967,  -5.01055177],
       [  5.52556208,  -8.18696464],
       [  0.81677922,   4.75330395],
       [  2.52859794,   4.5759393 ],
       [  3.69548081,   5.14288792],
       [  2.37698085,   5.82428626],
       [  5.69192445,  -9.47641249],
       [  0.91726632,  -6.52637778],
       [  1.44712872,   4.75428451],
       [  2.96590542,   4.5052704 ],
       [  6.68288513, -10.31693051],
       [ -0.43558928,  -4.7222919 ],
       [  0.34789333,  -3.88965912],
       [  0.97700138,  -5.7984931 ],
       [  2.45717481,   5.96515011],
       [  2.60711685,   2.84436554],
       [  2.89022984,   2.98168388]])

cluster_df = pd.DataFrame(Sample)


# In[3]:


from sklearn.cluster import KMeans


# In[4]:


k = KMeans(init='random', n_clusters = 3)


# In[5]:


k.fit(cluster_df)


# In[6]:


pred = k.predict(cluster_df)


# In[7]:


cluster_df['2'] = pred
cluster_df.columns = ['X','Y',"Cluster"]


# # 3

# In[14]:


df_0 = cluster_df[cluster_df['Cluster'] == 0]
df_1 = cluster_df[cluster_df['Cluster'] == 1]
df_2 = cluster_df[cluster_df['Cluster'] == 2]


# In[15]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[20]:


plt.figure(figsize=(10,10))
plt.scatter(df_0['X'],df_0['Y'],color='blue', label="cluster 0")
plt.scatter(df_1['X'],df_1['Y'],color='red', label="cluster 1")
plt.scatter(df_2['X'],df_2['Y'],color='green', label="cluster 2")
plt.legend(loc = 'upper left')
plt.show()


# In[ ]:




