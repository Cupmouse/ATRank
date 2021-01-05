"""FastTextを利用して文章を変換する"""

import pickle
import numpy as np
from gensim.models.wrappers import FastText

with open('../raw_data/reviews.pkl', 'rb') as f:
  reviews_df = pickle.load(f)
  reviews_df = reviews_df['reviewText']

fasttext = FastText.load_fasttext_format('../raw_data/cc.en.300.bin')

def sec2vec(sentence):
  global fasttext
  # 文を単語に分ける
  words = sentence.split()
  # 存在する単語のみ利用
  words = [fasttext[word] for word in words if word in fasttext]
  # 文のベクトルを平均で算出
  if len(words) == 0:
    return np.zeros((300,))
  return np.mean(words, axis=0)

# レビュー文をベクトル化
texts = np.ndarray((len(reviews_df), 300))
for i, sent in enumerate(reviews_df):
  texts[i] = sec2vec(sent)

with open('../raw_data/texts.pkl', 'wb') as f:
  pickle.dump(texts, f, pickle.HIGHEST_PROTOCOL)
