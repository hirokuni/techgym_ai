#!/usr/bin/env python
# coding: utf-8

# In[2]:


#フォントの準備
import urllib.request as req

def get_font_file_name():
    return './Osaka.ttc'

def retrieveOsaka():    
    url = "https://github.com/hokuto-HIRANO/Word2Vec/raw/master/font/Osaka.ttc"
    req.urlretrieve(url, get_font_file_name)
    return  file_name


# In[ ]:




