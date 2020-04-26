#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
from gensim.models.keyedvectors import KeyedVectors
import utils.FontUtil as FontUtil
import numpy as np
pd.set_option("display.max_rows", 15)


# In[2]:


title = "stanby-jobs-200d-word2vector.bin"
if not os.path.exists(title):
    print('Downloading ', title)
    url = "https://github.com/bizreach/ai/releases/download/2018-03-13/stanby-jobs-200d-word2vector.bin"
    urllib.request.urlretrieve(url, title)
else:
    print('Exists')
w2v = KeyedVectors.load_word2vec_format(title, binary=True)


# In[3]:


df = pd.read_csv('./words.csv')
vectors = []
zero_vec = np.zeros(200)
for value in df['words'].values:
    try:
        vectors.append(w2v[value])
    except:
        vectors.append(zero_vec)
        pass


# In[4]:


from sklearn.decomposition import PCA


# In[5]:


pca = PCA(n_components=2)


# In[6]:


V = pca.fit_transform(vectors)
type(V)


# In[7]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.font_manager as font_manager
import os


# In[8]:


# fontの設定
if not os.path.exists(FontUtil.get_font_file_name()):
    font_path = FontUtil.retrieveOsaka()
else:
    font_path = FontUtil.get_font_file_name()
prop = font_manager.FontProperties(fname = font_path)


# In[9]:


plt.figure(figsize=(20,20))
plt.scatter(V[:,0], V[:,1])
for w,x,y in zip(df['words'].values, V[:,0], V[:,1]):
    plt.annotate(w, xy=(x,y), xytext=(3,3), textcoords = 'offset points', fontproperties=prop, fontsize=12)


# # (2)

# In[10]:


from sklearn.cluster import KMeans


# In[11]:


k_means = KMeans(n_clusters=10, init='random')


# In[12]:


k_means.fit(V)


# In[13]:


clusters = k_means.predict(V)


# In[14]:


df_V = pd.DataFrame(V, columns = ['X','Y'])
df_V['Cluster'] = clusters
df_V


# In[15]:


colors = df_V['Cluster'].astype(np.float)


# In[16]:


plt.figure(figsize=(20,20))
plt.scatter(V[:,0], V[:,1], c = colors)
for color, w, x, y in zip(colors, df['words'], V[:,0], V[:,1]):
    plt.annotate(w, xy=(x,y), xytext=(3,3), textcoords = 'offset points', fontproperties=prop, fontsize=12)


# # (3)

# In[17]:


from sklearn.manifold import TSNE


# In[18]:


t = TSNE(n_components = 2, random_state=0)


# In[19]:


V = t.fit_transform(vectors)


# In[20]:


plt.figure(figsize=(20, 20))
plt.scatter(V[:,0], V[:,1])
for w, x, y in zip(df['words'], V[:,0], V[:,1]):
    plt.annotate(w, xy=(x,y), xytext=(3,3), textcoords = 'offset points', fontproperties=prop, fontsize=12)


# In[ ]:




