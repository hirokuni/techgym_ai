#!/usr/bin/env python
# coding: utf-8

# In[8]:


import urllib.request
import zipfile
import re


# In[11]:


def retrieve_rojin_to_umi():
    url = 'https://www.aozora.gr.jp/cards/001847/files/57347_ruby_57225.zip'
    zip = '57347_ruby_57225.zip'
    urllib.request.urlretrieve(url, zip)

    with zipfile.ZipFile(zip,'r') as myzip:
        myzip.extractall()
        for myfile in myzip.infolist():
            print('Extracted file: ', myfile)
            with open(myfile.filename, encoding='sjis') as f:
                text = f.read()
            
    text = re.split('\-{5,}',text)[2]   # ヘッダ部分の除去
    text = re.split('底本：',text)[0]   # フッタ部分の除去
    text = text.replace('|', '')        # | の除去
    text = re.sub('《.+?》', '', text)  # ルビの削除
    text = re.sub('［＃.+?］', '',text) # 入力注の削除
    text = re.sub('\n\n', '\n', text)   # 空行の削除
    text = re.sub('\r', '', text)
    
    return text


# In[ ]:




