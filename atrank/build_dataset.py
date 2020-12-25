#データセットのファイルを読み込み、埋め込みしやすい形に直して保存する
import random
import pickle
import numpy as np
from sklearn import decomposition
import pandas as pd
import itertools
import gc

random.seed(1234)

#user_count, item_count, cate_count, example_countはそれぞれユーザ、アイテム、カテゴリ、レビュー履歴の数
with open('../raw_data/remap.pkl', 'rb') as f:
  # asinが整数になっているレビューのデータ
  reviews_df = pickle.load(f)
  # asinでソートされた商品に対するカテゴリ（商品のカテゴリ）の羅列
  cate_list = pickle.load(f)
  user_count, item_count, cate_count, example_count = pickle.load(f)
  pickle.load(f)
  img_list, img_key = pickle.load(f)

with open('../raw_data/image_embeddings.pkl', 'rb') as f:
  image_embeddings = pickle.load(f)

# 時間をカテゴリカルな値にするためのテーブル
# [1, 2) = 0, [2, 4) = 1, [4, 8) = 2, [8, 16) = 3...  need len(gap) hot
gap = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
# gap = [2, 7, 15, 30, 60,]
# gap.extend( range(90, 4000, 200) )
# gap = np.array(gap)
print(gap.shape[0])

def proc_time_emb(hist_t, cur_t):
  """
  絶対タイムスタンプを連続値→相対タイムスタンプのカテゴリ値に変換
  hist_t：行動した絶対時間（日）のリスト
  cur_t：最後の行動の絶対時間（日）
  """
  # 最後の行動との時間差をとってカテゴリ特徴へ
  hist_t = [cur_t - abs_time + 1 for abs_time in hist_t]
  # sum([2, 4, 8...] <= rel_time)＝カテゴリ値
  hist_t = [np.sum(rel_time >= gap) for rel_time in hist_t]
  return hist_t

train_set = []
test_set = []
# histは各reviewerIDについてのレビューデータ(reviewerID以外のカラム全て)
for reviewerID, hist in reviews_df.groupby('reviewerID'):
  # 正例のラベル
  pos_list = hist['asin'].tolist()
  # 時間のリスト
  tim_list = hist['unixReviewTime'].tolist()
  # 時間を1日単位に変換
  tim_list = [i // 3600 // 24 for i in tim_list]
  def gen_neg():
    """一つの行動に対する負例の生成"""
    # 他の正例とかぶらないように繰り返しランダムでカテゴリ整数を生成
    neg = pos_list[0]
    while neg in pos_list:
      neg = random.randint(0, item_count-1)
    return neg
  # 正例と同じ数だけ負例を生成
  neg_list = [gen_neg() for i in range(len(pos_list))]

  # 訓練データを増やすために、元データをスライスする
  for i in range(1, len(pos_list)):
    hist_i = pos_list[:i]
    hist_t = proc_time_emb(tim_list[:i], tim_list[i])
    # 一番最後の履歴はテストに、他は訓練に入れる
    if i != len(pos_list) - 1:
      # (ユーザー, 履歴, 履歴時間, ラベル, 正例なら1でないなら0)
      train_set.append((reviewerID, hist_i, hist_t, pos_list[i], 1, img_list[pos_list[i]]))
      train_set.append((reviewerID, hist_i, hist_t, neg_list[i], 0, img_list[neg_list[i]]))
    else:
      # (ユーザー, 履歴, 履歴時間, (正例,負例))
      label = (pos_list[i], neg_list[i])
      img = (img_list[label[0]], img_list[label[1]])
      test_set.append((reviewerID, hist_i, hist_t, label, img))

# 訓練データからtSVD(PCA)を訓練し次元削減
train_img = pd.Series(pd.Series([ts[5] for ts in train_set]).unique())
train_img = train_img.map(lambda id: img_key[id]) # in key
train_img = train_img[train_img != 'not_available']
# 辞書：キー→PCAで訓練された画像の配列のIndex
train_img_map = {key: id for id, key in enumerate(train_img)}
train_img = train_img.map(lambda key: image_embeddings[key]).to_list()
image_tsvd = decomposition.TruncatedSVD(n_components=64, random_state=1234)
train_img = image_tsvd.fit_transform(train_img)
# 画像が存在しない場合の代わり
image_placeholder = np.sum(train_img, axis=0)
# 学習セットの画像IDをPCAを通した特徴に置き換え
for i, ts in enumerate(train_set):
  key = img_key[ts[5]]
  ts = list(ts)
  if key in train_img_map:
    # 埋め込み表現がある
    ts[5] = train_img[train_img_map[key]]
  else:
    # 埋め込み表現がない
    ts[5] = image_placeholder
  train_set[i] = tuple(ts)

gc.collect()

# テストセットについても同様
test_img = itertools.chain.from_iterable([list(ts[4]) for ts in test_set])
test_img = pd.Series(pd.Series(test_img).unique())
test_img = test_img.map(lambda id: img_key[id])
test_img = test_img[test_img != 'not_available']
test_img_map = {key: id for id, key in enumerate(test_img)}
test_img = test_img.map(lambda key: image_embeddings[key]).to_list()
# メモリ対策
image_embeddings = None
test_img = image_tsvd.transform(test_img)
for i, ts in enumerate(test_set):
  key_1 = img_key[ts[4][0]]
  key_2 = img_key[ts[4][1]]
  ts = list(ts)
  img_1 = test_img[test_img_map[key_1]] if key_1 in test_img_map else image_placeholder
  img_2 = test_img[test_img_map[key_2]] if key_2 in test_img_map else image_placeholder
  ts[4] = (img_1, img_2)
  test_set[i] = tuple(ts)

# シャフル
random.shuffle(train_set)
random.shuffle(test_set)

assert len(test_set) == user_count
# assert(len(test_set) + len(train_set) // 2 == reviews_df.shape[0])

# train_set・test_setは一つ一つのレビュー履歴についてユーザID、レビューしたアイテムID、これ以前にレビューしたアイテム、過去の履歴との時間差を含む
with open('dataset.pkl', 'wb') as f:
  pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump((user_count, item_count, cate_count), f, pickle.HIGHEST_PROTOCOL)
