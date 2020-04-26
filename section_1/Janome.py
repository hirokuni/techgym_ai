#!/usr/bin/env python
# coding: utf-8

# # 準備

# In[3]:


from janome.tokenizer import Tokenizer


# In[5]:


t = Tokenizer()
# Word2Vec :   https://www.excellence-blog.com/2018/05/06/word2vec-%E5%8D%98%E8%AA%9E%E3%81%AE%E3%83%99%E3%82%AF%E3%83%88%E3%83%AB%E8%A1%A8%E7%8F%BE/


# # インスタンス変数

# In[6]:


for token in t.tokenize('吾輩は猫である。'):
    print('表層系> ', token.surface)
    print('活用形> ', token.infl_type)
    print('活用形> ', token.infl_form)
    print('品詞    > ',token.part_of_speech) 
    print('基本形> ', token.base_form) # 単語の原型
    print('読み   >', token.reading)
    print('発音   >', token.phonetic)
    print('----')


# # 品詞の取得

# In[27]:


for token in t.tokenize('吾輩は猫である。'):
    for one_part_of_speech in token.part_of_speech.split(','):
        print(one_part_of_speech) 
    print('----')


# In[ ]:




