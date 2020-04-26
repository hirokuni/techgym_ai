#!/usr/bin/env python
# coding: utf-8

# In[1]:


import utils.SampleDataRetriever as SampleDataRetriever
import time


# In[2]:


text = SampleDataRetriever.retrieve_rojin_to_umi()


# In[3]:


from gensim.models import word2vec
from janome.tokenizer import Tokenizer


# In[4]:


t = Tokenizer()


# # wakati書きを使う

# In[5]:


wakati_1= t.tokenize(text, wakati=True)


# In[6]:


wakati_1


# ## OKのケース
# Wakati書きをリストにWrapしてWord2Vecの学習データとしてINPUT

# In[18]:


# 　https://stackoverflow.com/questions/45420466/gensim-keyerror-word-not-in-vocabulary
#  model = word2vec.Word2Vec(wakati_1, size=100, min_count=1, window=3, iter=3) <-- wakati_1はiterative of sentencesではない!!

time_s  = time.time()
model = word2vec.Word2Vec([wakati_1], size=100, min_count=1, window=3, iter=3)
time_e  = time.time()
print(time_e - time_s)
print('--')

results = model.wv.most_similar(positive="彼", topn=3)
for result in results:
    print(result[0],result[1])
    
print("===")

# # # メキシコがVocabularyになくErrorになる ###
try:
    results = model.wv.most_similar(positive="メキシコ", topn=3)
    for result in results:
        print(result[0],result[1])
except KeyError as error:
        print("Error: ",error)


# ## ダメなケース
# 　Word2Vec学習のデータとして、Wakati書きされた単語で個々のword_listを使う。この場合Wort to Vecではコンテキストが取れないので、あまり意味が無い学習になっている。普通こういうのはやらない。

# In[8]:


#対象の品詞
obj_list = ['名詞', '動詞','形容詞']
def extract_words(s):
#     print('===')
#     print(s)
    tokens = t.tokenize(s)
    r = [token.base_form for token in tokens if token.part_of_speech.split(',')[0] in obj_list]
    return r

word_list = [extract_words(sentence) for sentence in wakati_1]

model3 = word2vec.Word2Vec(word_list, size=100, min_count=1, window=3, iter=3)

# results
results = model3.wv.most_similar(positive="彼", topn=3)
for result in results:
    print(result[0],result[1])
print('--')

results = model3.wv.most_similar(positive="メキシコ", topn=3)
for result in results:
    print(result[0],result[1])
    
print(word_list)


# # 　。でSplitしたSentencesを使う

# In[9]:


sentences = text.split('。')
sentences


# ## ダメなケース
# sentencesをそのまま学習 (INPUTが単語リストになってないので、このパターンは普通やらない！！）

# In[10]:


# word2vecは単語に分けて学習させるので、sentencesのままは普通INPUTしない。
model2 = word2vec.Word2Vec(sentences, size=100, min_count=1, window=3, iter=3)
results = model2.wv.most_similar(positive="彼", topn=5)
for result in results:
    print(result[0],result[1])

# # # メキシコがVocabularyになくErrorになる ###
# results = model2.wv.most_similar(positive="メキシコ", topn=5)
# for result in results:
#     print(result[0],result[1])


# ## OKのケース　Word2Vec学習のデータとして、品詞（名詞、動詞、形容詞）のみのWord listを使う

# In[21]:


#対象の品詞
# obj_list = ['名詞', '動詞','形容詞','形容動詞','副詞','連体詞','接続詞','感動詞', '助動詞', '助詞']
obj_list = ['名詞', '動詞','形容詞']
def extract_words(s):
    tokens = t.tokenize(s)
    r = [token.base_form for token in tokens if token.part_of_speech.split(',')[0] in obj_list]
    # https://www.excellence-blog.com/2018/05/06/word2vec-%E5%8D%98%E8%AA%9E%E3%81%AE%E3%83%99%E3%82%AF%E3%83%88%E3%83%AB%E8%A1%A8%E7%8F%BE/
    # 
    return r

word_list = [extract_words(sentence) for sentence in sentences]

time_s = time.time()
model3 = word2vec.Word2Vec(word_list, size=100, min_count=1, window=3, iter=3)
time_e  = time.time()
print(time_e - time_s)
print('--')

# results
results = model3.wv.most_similar(positive="彼", topn=3)
for result in results:
    print(result[0],result[1])
print('--')

results = model3.wv.most_similar(positive="メキシコ", topn=3)
for result in results:
    print(result[0],result[1])
print("---")
print(word_list)


# # Parameters

# In[ ]:


# https://www.kaggle.com/pierremegret/gensim-word2vec-tutorial#The-parameters:
# https://qiita.com/mergit/items/822dc49343c887019d44 
model3 = word2vec.Word2Vec(word_list, size=100, min_count=1, window=3, iter=3)
# min_count : この数より小さい出現頻度の単語は除外する
# window : windowサイズ(ターゲットの単語の前後の単語数。ゼロから始めるDL　Vol2のW2Vのコンテキストという説明を参照)
# size : 単語ベクトルの次元数

