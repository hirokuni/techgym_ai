#!/usr/bin/env python
# coding: utf-8

# In[1]:


import utils.SampleDataRetriever as sdr
from gensim.models import Word2Vec
from janome.tokenizer import Tokenizer
import os


# In[2]:


obj_list = ["名詞","動詞","形容詞"]
def extract_word(sentence):
    tokens = t.tokenize(sentence)
    r = [token.base_form for token in tokens if token.part_of_speech.split(',')[0] in obj_list]
    return r

is_forced_learning = True

if not os.path.exists('words.model') or is_forced_learning:
    text = sdr.retrieve_rojin_to_umi()
    t = Tokenizer()
    sentences = text.split("。")
    word_list = [extract_word(sentence) for sentence in sentences]
    model = Word2Vec(word_list,size=100, min_count=5, window=5, iter=100)
    model.save("./words.model")
    print('created new model "words.model"')
else:
    model = Word2Vec.load('./words.model')
    print('load words.model')


# In[3]:


words = []
words.append('老人')
words.append('海')
words.append('ヘミングウェイ')
words.append('魚')
words.append('彼')


# In[4]:


vectors = []
for w in words:
    vectors.append(model.wv[w])


# In[5]:


vectors


# # 各５つの単語のベクトルをそれぞれ２次元ベクトルへPCAで圧縮する

# In[6]:


from sklearn.decomposition import PCA
import numpy as np


# In[7]:


pca = PCA(n_components = 2)


# In[8]:


pca.fit(vectors)


# In[9]:


vectors_pca = pca.transform(vectors)


# In[10]:


for w in vectors_pca:
    print(w)
    print(np.array2string(w, separator=', ', formatter={'float_kind': lambda x: '{: .4f}'.format(x)}))
    print('--')


# # 平面へプロット

# In[11]:


import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


#フォントの準備
import urllib.request as req
url = "https://github.com/hokuto-HIRANO/Word2Vec/raw/master/font/Osaka.ttc"
req.urlretrieve(url, "./Osaka.ttc")

#フォントの指定
FONTPATH='./Osaka.ttc'
prop = font_manager.FontProperties(fname=FONTPATH)


# In[13]:


print(type(vectors_pca))

for index, vector in enumerate(vectors_pca):
    plt.plot(vector[0], vector[1], ms=5.0, zorder=2,marker='o')
    plt.annotate(words[index], (vector[0], vector[1]), fontproperties=prop, fontsize=15)


# # 3 「老人」と「海」をれぞれに類似している単語の上位１００個を平面にプロット

# In[23]:


def draw_2d_2groups(vectors, target1, target2, topn=100):
    similars1 = [w[0] for w in vectors.wv.most_similar(target1, topn=topn)]
    similars1.insert(0, target1)
    print(similars1)
    print("--")
    similars2 = [w[0] for w in vectors.wv.most_similar(target2, topn=topn)]
    similars2.insert(0, target2)
    print(similars2)
    print("--")
    similars = similars1 + similars2
    colors = ['b']+['g']*(topn)+ ['r']+['orangered']*(topn)
    print(colors)
    X = [vectors.wv[w] for w in similars]
    pca = PCA(n_components=2)
    Y = pca.fit_transform(X)
    plt.figure(figsize=(20,20))
    plt.scatter(Y[:,0], Y[:,1], color=colors)
    for w, x, y, c in zip(similars[:], Y[:,0], Y[:,1], colors):
        plt.annotate(w, xy=(x, y), xytext=(3,3), textcoords='offset points', fontproperties=prop, fontsize=15, color=c)
    plt.show()


# In[24]:


draw_2d_2groups(model, '老人', '海', 100)


# # 4クラスタ数３でクラスタリングして色分けする

# In[34]:


from sklearn.cluster import KMeans
import pandas as pd


# In[53]:


def draw_2d_2groups_k(vectors, target1, target2, topn=100):
    similars1 = [w[0] for w in vectors.wv.most_similar(target1, topn=topn)]
    similars1.insert(0, target1)
    similars2 = [w[0] for w in vectors.wv.most_similar(target2, topn=topn)]
    similars2.insert(0, target2)
    similars = similars1 + similars2
    colors = ['b']+['g']*(topn)+ ['r']+['orangered']*(topn)
    X = [vectors.wv[w] for w in similars]
    pca = PCA(n_components=2)
    Y = pca.fit_transform(X)
    
    
    # クラスター分析
    kmeans = KMeans(init = 'random', n_clusters=3)
    kmeans.fit(Y)
    z_pred = kmeans.predict(Y)
    print(z_pred)
    
    
    # データをmerge
    merge_data = pd.concat([pd.DataFrame(Y[:,0]), pd.DataFrame(Y[:,1]), pd.DataFrame(z_pred)], axis = 1)
    merge_data.columns = ["X","Y","Cluster"]
#     display(merge_data)
    
    df0 = merge_data[merge_data['Cluster']==0] # cluster 0 の座標
    df1 = merge_data[merge_data['Cluster']==1] # cluster 1 の座標
    df2 = merge_data[merge_data['Cluster']==2] # cluster 2 の座標
    
    # plot
    plt.figure(figsize=(20,20))
    plt.scatter(df0['X'], df0['Y'], color='blue', label='cluster 0')
    plt.scatter(df1['X'], df1['Y'], color='red', label='cluster 1')
    plt.scatter(df2['X'], df2['Y'], color='green', label='cluster 2')
    
    for w, x, y, c in zip(similars[:], Y[:,0], Y[:,1], colors):
        plt.annotate(w, xy=(x, y), xytext=(3,3), textcoords='offset points', fontproperties=prop, fontsize=15, color=c)
    plt.legend()
    plt.show()


# In[54]:


draw_2d_2groups_k(model, '老人','海', 100)


# In[ ]:




