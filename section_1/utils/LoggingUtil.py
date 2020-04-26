#!/usr/bin/env python
# coding: utf-8

# In[1]:


import logging
# フォーマットを定義
formatter = '%(levelname)s : %(asctime)s : %(message)s'
logging.basicConfig(level=logging.DEBUG, format=formatter)


# In[2]:


def logging_info(*args):
    logging.info(args)


# In[3]:


# logging_info('msg: {}','test')


# In[4]:


logging_info('start')


# In[ ]:




