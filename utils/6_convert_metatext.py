"""FastTextを利用して文章を変換する"""
#文の前処理を入れる
#nltkによる分かち書き、ステミング、ストップワード除去
#titleを変換

import pickle
import numpy as np
from gensim.models.wrappers import FastText
from nltk import tokenize
from nltk import stem
from nltk.corpus import stopwords
import re
import gensim
import tensorflow as tf
import tensorflow_hub as hub

stop_words = set(stopwords.words('english'))
signals = re.compile('[^a-zA-Z0-9]+')
fasttext = FastText.load_fasttext_format('../raw_data/cc.en.300.bin')

with open('../raw_data/remap.pkl', 'rb') as f:
  pickle.load(f)
  pickle.load(f)
  pickle.load(f)
  pickle.load(f)
  pickle.load(f)
  tit_list, tit_key = pickle.load(f)

def sec2vec(sentence):
  sentence = sentence.replace('&#34;', '')
  sentence = signals.sub(' ', sentence)
  global fasttext
  # 文を単語に分ける
  words = tokenize.word_tokenize(sentence)
  # ストップワードフィルタリング
  filtered_words = [word for word in words if word not in stop_words]
  # 存在する単語のみ利用
  words_vectors = [fasttext[word] for word in filtered_words if word in fasttext]
  # 文のベクトルを平均で算出
  if len(words_vectors) == 0:
    return np.zeros((300,), dtype=np.float32)
  return np.mean(words_vectors, axis=0)

# レビュー文をベクトル化
r = np.ndarray((len(tit_key), 300), dtype=np.float32)

for i in range(len(tit_key)):
  r[i] = sec2vec(tit_key[i])
    
with open('../raw_data/text_embeddings.pkl', 'wb') as f:
  pickle.dump(r, f, pickle.HIGHEST_PROTOCOL)
