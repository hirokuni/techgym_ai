#!/usr/bin/env python
# coding: utf-8

# # (1)

# In[1]:


import pandas as pd


# In[2]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# In[4]:


# Dataset : https://scikit-learn.org/stable/datasets/index.html
from sklearn.datasets import load_breast_cancer


# In[5]:


cancer = load_breast_cancer()
cancer.keys()
# 乳がんデータの説明  https://newtechnologylifestyle.net/scikitlean_logistic/


# In[6]:


cancer.target_names


# In[7]:


print(cancer.DESCR)


# In[8]:


import numpy as np
# DESCRよりMalignant(悪性)が212、 Benign(良性)が357。

print('0: ', np.count_nonzero(cancer.target == 0)) # Malignant
print('1: ', np.count_nonzero(cancer.target == 1)) # Benign


# In[9]:


cancer.feature_names


# In[10]:


print("データ件数", len(cancer.data))
print("次元数: ",cancer.data.ndim)
df = pd.DataFrame(cancer.data)
df.columns = cancer.feature_names
display(df)


# In[11]:


len(df.columns)


# In[12]:


fig = plt.figure()
ax_3d = Axes3D(fig)
ax_3d.set_xlabel(cancer.feature_names[0])
ax_3d.set_ylabel(cancer.feature_names[1])
ax_3d.set_zlabel(cancer.feature_names[2])
ax_3d.view_init(elev=10, azim=15)
ax_3d.plot(cancer.data[:, 0], cancer.data[:, 1], cancer.data[:, 2],marker="o", linestyle='None', color='black' )
plt.show()


# In[13]:


# 標準化 (単位系が違う時は標準化すること！ https://datachemeng.com/autoscaling_before_pca/ )
sc = StandardScaler()
sc.fit(cancer.data)
x_std = sc.transform(cancer.data)

# 試しに3D表現
fig = plt.figure()
ax_3d = Axes3D(fig)
ax_3d.view_init(elev=10, azim=15)
ax_3d.plot(x_std[:, 0], x_std[:, 1], x_std[:, 2],marker="o", linestyle='None', color='black' )
plt.show()


# In[14]:


# 主成分分析
pca = PCA(n_components=2)
pca.fit(x_std)
x_pca = pca.transform(x_std)

print('shape:',x_pca.shape) 
print('寄与率:',pca.explained_variance_ratio_) #2次元での30次元のデータをおよそ６割表現できている。 0.44272026 0.18971182


# In[ ]:





# In[15]:


df_pca = pd.DataFrame(x_pca, columns=['pc1','pc2'])
df_pca = pd.concat([df_pca, pd.DataFrame(cancer.target, columns=['target'])], axis=1)
df_pca_malignant = df_pca[df_pca['target']==0]
df_pca_benign = df_pca[df_pca['target']==1]


# In[16]:


plt.scatter(df_pca_malignant['pc1'], df_pca_malignant['pc2'], color='red', label = 'malignant')
plt.scatter(df_pca_benign['pc1'], df_pca_benign['pc2'], color='green', label = 'benign')
plt.xlabel("pc1") 
plt.ylabel ("pc2")
plt.legend(loc='upper left')


# # (2)

# In[17]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.datasets import load_iris
pd.set_option('display.max_rows', 10)


# In[18]:


iris = load_iris()


# In[19]:


display(iris.keys())
display(iris.target_names)
display(iris.target)
display(pd.DataFrame(iris.data).info())
display(iris.feature_names)
# print(iris.DESCR)


# In[20]:


df_iris = pd.DataFrame(iris.data)
df_iris.columns = iris.feature_names
display(df_iris)

fig = plt.figure(figsize=(10,6))
ax1 = fig.add_subplot(1,3,1, projection='3d',azim = 15, elev = 10)
ax2 = fig.add_subplot(1,3,3, projection='3d',azim = -15, elev = 10)
ax3 = fig.add_subplot(1,3,2, projection='3d',azim = 0, elev = 10)

# ax_3d = Axes3D(fig)
# ax_3d.set_xlabel(iris.feature_names[0])
ax1.set_title( 'azim = 15')
ax1.set_xlabel(iris.feature_names[1])
ax1.set_ylabel(iris.feature_names[2])
ax1.set_zlabel(iris.feature_names[3])
#ax1.view_init(elev=10, azim=-15)
ax1.plot( iris.data[:, 1], iris.data[:, 2],  iris.data[:, 2], marker="o", linestyle='None', color='black' )

ax3.set_title( 'azim = 0')
ax3.set_xlabel(iris.feature_names[1])
ax3.set_ylabel(iris.feature_names[2])
ax3.set_zlabel(iris.feature_names[3])
ax3.plot( iris.data[:, 1], iris.data[:, 2],  iris.data[:, 2], marker="o", linestyle='None', color='black' )

ax2.set_title( 'azim = -15')
ax2.set_xlabel(iris.feature_names[1])
ax2.set_ylabel(iris.feature_names[2])
ax2.set_zlabel(iris.feature_names[3])
ax2.plot( iris.data[:, 1], iris.data[:, 2],  iris.data[:, 2], marker="o", linestyle='None', color='black' )

plt.show()


# In[21]:


# 標準化
sc = StandardScaler()
sc.fit(iris.data)
x_std = sc.transform(iris.data)


# In[22]:


fig = plt.figure(figsize=(10,6))
ax_3d = Axes3D(fig)
# ax_3d.set_xlabel(iris.feature_names[1])
# ax_3d.set_ylabel(iris.feature_names[2])
# ax_3d.set_zlabel(iris.feature_names[3])
ax_3d.view_init(elev=10, azim=-15)
ax_3d.plot( x_std[:, 1], x_std[:, 2],  x_std[:, 2], marker="o", linestyle='None', color='black' )

plt.show()


# In[23]:


# 主成分分析
pca = PCA(n_components=2)
pca.fit(x_std)
x_pca = pca.transform(x_std)

df_pca = pd.DataFrame(x_pca, columns=['pc1','pc2'])
df_pca = pd.concat([df_pca, pd.DataFrame(cancer.target, columns=['target'])], axis=1)
df_pca_malignant = df_pca[df_pca['target']==0]
df_pca_benign = df_pca[df_pca['target']==1]


# In[24]:


# 主成分分析前の次元
display(iris.data.shape)
display(x_pca.shape)


# In[36]:


df_pca = pd.DataFrame(x_pca, columns = ['PC1','PC2'])
df_pca = pd.concat([df_pca, pd.DataFrame(iris.target, columns=['Target'])], axis =  1)
df_pca_0 = df_pca[df_pca.Target == 0]
df_pca_1 = df_pca[df_pca.Target == 1]
df_pca_2 = df_pca[df_pca.Target == 2]

df_pca


# In[38]:


plt.scatter(df_pca_0['PC1'], df_pca_0['PC2'], color='blue', label = 'setosa')
plt.scatter(df_pca_1['PC1'], df_pca_1['PC2'], color='red', label = 'versicolor')
plt.scatter(df_pca_2['PC1'], df_pca_2['PC2'], color='green', label = 'virginica')
plt.xlabel("pc1") 
plt.ylabel("pc2") 
plt.legend()
#  'setosa', 'versicolor', 'virginica']


# # ndarrayの要素指定方法を練習

# In[26]:


import numpy as np
ndarray = np.array([[1,2,3],[4,5,6],[7,8,9]])
ndarray


# In[27]:


ndarray[:,0]


# # DataFrameの行をSeriesのBoolで取り出す

# In[28]:


df_test = pd.DataFrame({'a': ['1st', '2nd','3rd']})
display(type(pd.Series([True,False,True])))
display(df_test[pd.Series([True,False,False])])
display(df_test[pd.Series([True,False,True])])


# In[ ]:




