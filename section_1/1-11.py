#!/usr/bin/env python
# coding: utf-8

# In[1]:


import urllib.request
import zipfile
import re
from janome.tokenizer import Tokenizer


# In[2]:


url = 'https://www.aozora.gr.jp/cards/001847/files/57347_ruby_57225.zip'
zip = '57347_ruby_57225.zip'
urllib.request.urlretrieve(url, zip)


# In[3]:


with zipfile.ZipFile(zip, 'r') as myzip:
    myzip.extractall()
    for myfile in myzip.infolist():
        print(myfile.filename)
        with open(myfile.filename, encoding='sjis') as file:
            text = file.read()
            print(text)


# In[4]:


text = re.split('\-{5,}',text)[2]   # ヘッダ部分の除去
text = re.split('底本：',text)[0]   # フッタ部分の除去
text = text.replace('|', '')        # | の除去
text = re.sub('《.+?》', '', text)  # ルビの削除
text = re.sub('［＃.+?］', '',text) # 入力注の削除
text = re.sub('\n\n', '\n', text)   # 空行の削除
text = re.sub('\r', '', text)


# In[5]:


outnum = 50
# 頭の100文字の表示 
print(text[:outnum])
print("…")
# 後ろの100文字の表示 
print(text[-outnum:])


# In[6]:


#対象の品詞
obj_list = ['名詞', '動詞','形容詞']
t = Tokenizer()


# In[7]:


# テキストを引数として、形態素解析の結果で対象のみを配列で抽出する関数を定義 
def extract_words(s):
    tokens = t.tokenize(s)
    r = [token.base_form for token in tokens if token.part_of_speech.split(',')[0] in obj_list]
#     for token in tokens:
#         print(token)
#         print(token.part_of_speech)
    return r


# In[8]:


# 全体のテキストを句点('。')で区切った配列
sentences = text.split('。')


# In[9]:


# それぞれの文章を単語リストに変換(処理に数分かかります)
# sentence毎の配列が格納されている
word_list = [extract_words(sentence) for sentence in sentences]


# In[10]:


# 一部確認
for word in word_list[0:3]:
    print(word)


# In[11]:


from gensim.models import word2vec


# In[12]:


model = word2vec.Word2Vec(word_list, size=100, min_count=5, window=5, iter=100)
model.save('./words.model')


# In[13]:


model.wv.most_similar('老人', topn=5)


# In[14]:


model.wv.most_similar('海', topn=5)


# In[15]:


model.wv.n_similarity([ '海'],['老人'])


# In[16]:


model.wv.most_similar(positive=['海'], negative=['老人'], topn=5)


# In[17]:


model.wv.most_similar(positive=['海', '老人'], topn=5)


# In[ ]:




