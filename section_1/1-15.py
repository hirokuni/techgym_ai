#!/usr/bin/env python
# coding: utf-8

# # (1)

# In[1]:


from gensim.models import KeyedVectors


# In[2]:


MODEL_FILENAME = "./stanby-jobs-200d-word2vector.bin"
w2v = KeyedVectors.load_word2vec_format(MODEL_FILENAME, binary = True)


# In[31]:


while True:
    print('単語を入力してください')
    word = input()
    if word == 'q':
        break;
    try:
        job = w2v.most_similar(word, topn=1)
        print('もっとも類似する単語は {} です'.format(job[0][0]))
    except:
        print('不明な単語です')

print("終了")


# In[ ]:


while True:
    print('> 単語を入力してください')
    word = input()
    if word == 'q':
        break;
    try:
        job = w2v.most_similar(word, topn=1)
        print('{}と言ったら{}です'.format(word, job[0][0]))
        print('{}と言ったら'.format(job[0][0]))
    except:
        print('不明な単語です. もう一度入力してください')

print("終了")


# In[ ]:




