"""word2vecを利用して文章を変換する"""
#文の前処理を入れる
#nltkによる分かち書き、ステミング、ストップワード除去

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
word2vec = gensim.models.KeyedVectors.load_word2vec_format('../raw_data/GoogleNews-vectors-negative300.bin',binary=True)


with open('../raw_data/remap.pkl', 'rb') as f:
  pickle.load(f)
  pickle.load(f)
  pickle.load(f)
  pickle.load(f)
  pickle.load(f)
  texts = pickle.load(f)

def sec2vec(sentence):
  sentence = sentence.replace('&#34;', '')
  sentence = signals.sub(' ', sentence)
  global fasttext
  # 文を単語に分ける
  words = tokenize.word_tokenize(sentence)
  # ストップワードフィルタリング
  filtered_words = [word for word in words if word not in stop_words]
  # 存在する単語のみ利用
  words_vectors = [word2vec[word] for word in filtered_words if word in word2vec]
  # 文のベクトルを平均で算出
  if len(words_vectors) == 0:
    return np.zeros((300,), dtype=np.float32)
  return np.mean(words_vectors, axis=0)

# レビュー文をベクトル化
r = np.ndarray((len(texts), 300), dtype=np.float32)
for i, sent in enumerate(texts):
  r[i] = sec2vec(sent)

with open('../raw_data/text_embeddings.pkl', 'wb') as f:
  pickle.dump(r, f, pickle.HIGHEST_PROTOCOL)
