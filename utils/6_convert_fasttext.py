"""FastTextを利用して文章を変換する"""

import pickle
import numpy as np
from gensim.models.wrappers import FastText

with open('../raw_data/remap.pkl', 'rb') as f:
  pickle.load(f)
  pickle.load(f)
  pickle.load(f)
  pickle.load(f)
  pickle.load(f)
  texts = pickle.load(f)

fasttext = FastText.load_fasttext_format('../raw_data/cc.en.300.bin')

def sec2vec(sentence):
  global fasttext
  # 文を単語に分ける
  words = sentence.split()
  # 存在する単語のみ利用
  words = [fasttext[word] for word in words if word in fasttext]
  # 文のベクトルを平均で算出
  if len(words) == 0:
    return np.zeros((300,), dtype=np.float32)
  return np.mean(words, axis=0)

# レビュー文をベクトル化
r = np.ndarray((len(texts), 300), dtype=np.float32)
for i, sent in enumerate(texts):
  r[i] = sec2vec(sent)

with open('../raw_data/text_embeddings.pkl', 'wb') as f:
  pickle.dump(r, f, pickle.HIGHEST_PROTOCOL)
