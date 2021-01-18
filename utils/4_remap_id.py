import os
import random
import pickle
import numpy as np

random.seed(1234)

# データセットの読み込みと利用する要素の選択
with open('../raw_data/small_reviews.pkl', 'rb') as f:
  reviews_df = pickle.load(f)
  reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime', 'reviewText']]
with open('../raw_data/meta.pkl', 'rb') as f:
  meta_df = pickle.load(f)
  meta_df = meta_df[['asin', 'categories', 'imUrl']]
meta_df['categories'] = meta_df['categories'].map(lambda x: x[-1][-1])
# URLを画像埋め込み表現を取得するためのキーへ変換する
meta_df['imUrl'] = meta_df['imUrl'].map(lambda url: os.path.basename(url) if isinstance(url, str) else 'not_available')

def build_map(df, col_name):
  """キーをユニークなIDに変換する。そのキーとそのIDをマッピングする辞書との逆処理の配列を返す"""
  key = sorted(df[col_name].unique().tolist())
  m = dict(zip(key, range(len(key))))
  df[col_name] = df[col_name].map(lambda x: m[x])
  return m, key

# 商品、カテゴリ、レビュアーのIDを整数へ変換
asin_map, asin_key = build_map(meta_df, 'asin')
cate_map, cate_key = build_map(meta_df, 'categories')
revi_map, revi_key = build_map(reviews_df, 'reviewerID')
img_map, img_key = build_map(meta_df, 'imUrl')

user_count, item_count, cate_count, example_count =\
    len(revi_map), len(asin_map), len(cate_map), reviews_df.shape[0]
print('user_count: %d\titem_count: %d\tcate_count: %d\texample_count: %d' %
      (user_count, item_count, cate_count, example_count))
print('image_count: %d' % len(img_map))

meta_df = meta_df.sort_values('asin')
meta_df = meta_df.reset_index(drop=True)
reviews_df['asin'] = reviews_df['asin'].map(lambda x: asin_map[x])
reviews_df = reviews_df.sort_values(['reviewerID', 'unixReviewTime'])
reviews_df = reviews_df.reset_index(drop=True)
texts = np.array(reviews_df['reviewText'], dtype=object)
reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]

# ASINの並び順に商品のカテゴリだけをとってきて配列にする
cate_list = [meta_df['categories'][i] for i in range(len(asin_map))]
cate_list = np.array(cate_list, dtype=np.int32)

# 商品の画像のID
img_list = np.array(meta_df['imUrl'], dtype=np.int32)

# 書き出し
with open('../raw_data/remap.pkl', 'wb') as f:
  pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL) # uid, iid
  pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL) # cid of iid line
  pickle.dump((user_count, item_count, cate_count, example_count),
              f, pickle.HIGHEST_PROTOCOL)
  pickle.dump((asin_key, cate_key, revi_key), f, pickle.HIGHEST_PROTOCOL)
  pickle.dump((img_list, img_key), f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(texts, f, pickle.HIGHEST_PROTOCOL)
