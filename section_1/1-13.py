#!/usr/bin/env python
# coding: utf-8

# In[1]:


import urllib.request
import os
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors


# In[2]:


title = "stanby-jobs-200d-word2vector.bin"
if not os.path.exists(title):
    print('Downloading ', title)
    url = "https://github.com/bizreach/ai/releases/download/2018-03-13/stanby-jobs-200d-word2vector.bin"
    urllib.request.urlretrieve(url, title)
else:
    print('Exists')


# In[3]:


w2v = KeyedVectors.load_word2vec_format(title, binary=True)


# In[4]:


#  w2v.save_word2vec_format("stanby-jobs-200d-word2vector.txt", binary=False)


# In[5]:


words = w2v.most_similar("テクノロジー", topn=5)
for word in words:
    print(word)


# In[6]:


words = w2v.most_similar(positive=["テクノロジー", "金融"], topn=5)
for word in words:
    print(word)


# In[7]:


w2v.most_similar(positive=["テクノロジー", "金融"], negative=["IT"], topn=5)


# In[8]:


vector = w2v["テクノロジー"]
w2v.similar_by_vector(vector, topn=5)


# In[9]:


# コサイン類似度
import numpy as np
def cos_sim(v1, v2):
    return np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
cos_sim(w2v["テクノロジー"], w2v['最新技術'])


# # (2)[Java]と[PHP]のコサイン類似度

# In[10]:


cos_sim(w2v['Java'],w2v['PHP'])


# # (3) 

# In[11]:


import pandas as pd


# In[16]:


df = pd.read_csv('./words.csv')
df_ = df
df.keys()


# In[25]:


print(df['words'][0])
print(df['words'][1])


# In[26]:


results = []
for index, row in df.iterrows():
#     print("index: {}, row: {}".format(index, row['words']))
#     if index > 100:
#         break;
    if index != df.size - 1:
        for i in range(index + 1, df.size):
            try:
                similarity = w2v.similarity(row['words'], df['words'][i])
#                 print("{}, {}, {}, {}".format(i, row['words'], df['words'][i], similarity))
                if similarity > 0.7:
                    results.append ([row['words'],  df['words'][i], similarity])
#                     print("{}, {}, {}".format(row['words'], df['words'][i], similarity))
            except:
                pass
    else:
        break
results


# In[27]:


# resutls = []
# for i in df['words'].values:
#     for j in df_['words'].values:
#         try:
#             similarity = w2v.similarity(i,j)
# #             print('{}, {}, {}'.format(i,j,similarity))
#             if similarity < 0.99 and similarity > 0.7:
#                 results.append([i,j,str(similarity)])
#         except:
#             pass
# results


# In[28]:


df_ = pd.DataFrame(results, columns=['単語A', '単語B','コサイン類似度'])


# In[31]:


df_


# In[32]:


# for i in range(1,3):
#     print (i)


# In[ ]:




