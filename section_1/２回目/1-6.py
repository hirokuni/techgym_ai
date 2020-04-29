#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[4]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# In[6]:


from sklearn.datasets import load_breast_cancer


# In[12]:


cancer = load_breast_cancer()
print(type(cancer))
cancer.feature_names


# In[14]:


fig =plt.figure()
ax_3d = Axes3D(fig)
ax_3d.set_xlabel(cancer.feature_names[0])
ax_3d.set_ylabel(cancer.feature_names[1])
ax_3d.set_zlabel(cancer.feature_names[2])
ax_3d.view_init(elev=10, azim=-15)
ax_3d.plot(cancer.data[:,0], cancer.data[:,1], cancer.data[:,2], marker="o", linestyle='None', color='black')


# In[31]:


sc =StandardScaler()
sc.fit(cancer.data)
X_std = sc.transform(cancer.data)
print(type(cancer.data))


# In[17]:


X_std


# In[35]:


pca = PCA(n_components = 2, # 圧縮後の次元 : default None
                   copy = False, # fitやfit_transformで変換するデータを上書き : default True
                    )
pca.fit(X_std)
X_pca = pca.transform(X_std)
type(X_pca)


# In[34]:


# 属性表示
print('x pca shape: {}'.format(X_pca.shape))
print('explained_variance: {}'.format(pca.explained_variance_))
print('explained_variance: {}'.format(pca.explained_variance_ratio_))
print('平均: {}'.format(pca.mean_))
print('共分散行列: {}'.format(pca.get_covariance()))


# In[37]:


X_pca = pd.DataFrame(X_pca, columns = ['pc1', 'pc2'])
X_pca = pd.concat([X_pca, pd.DataFrame(cancer.target, columns = ['target'])], axis = 1)


# In[43]:


pca_malignant = X_pca[X_pca['target'] == 0]
pca_benign = X_pca[X_pca['target'] == 1]
type(pca_benign)


# In[44]:


ax = pca_malignant.plot.scatter(x='pc1', y='pc2', color='red', label='malignant')
pca_benign.plot.scatter(x='pc1', y='pc2', color='blue', label='benign', ax=ax)


# In[ ]:




