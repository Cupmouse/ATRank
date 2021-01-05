# データセットのファイルを読み込み、埋め込みしやすい形に直して保存する
import random
import pickle
import numpy as np
from sklearn import decomposition
import itertools

random.seed(1234)

# user_count, item_count, cate_count, example_countはそれぞれユーザ、アイテム、カテゴリ、レビュー履歴の数
with open('../raw_data/remap.pkl', 'rb') as f:
  # asinが整数になっているレビューのデータ
  reviews_df = pickle.load(f)
  # asinでソートされた商品に対するカテゴリ（商品のカテゴリ）の羅列
  cate_list = pickle.load(f)
  user_count, item_count, cate_count, example_count = pickle.load(f)
  pickle.load(f)
  img_list, img_key = pickle.load(f)

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
  # 正例のラベル(商品のID)
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
      train_set.append((reviewerID, hist_i, hist_t, pos_list[i], 1))
      train_set.append((reviewerID, hist_i, hist_t, neg_list[i], 0))
    else:
      # (ユーザー, 履歴, 履歴時間, (正例,負例))
      label = (pos_list[i], neg_list[i])
      test_set.append((reviewerID, hist_i, hist_t, label))

# 訓練データからPCAを訓練して全画像の次元圧縮を行う
print('processing images...')

IMG_DIMINISHED_SIZE = 64

with open('../raw_data/image_embeddings.pkl', 'rb') as f:
  image_embeddings = pickle.load(f)
  img_missing_mask = pickle.load(f)

# 同じ画像は訓練データに複数回現れるので、ユニークなものだけを集める
img_train_mask = np.zeros(len(image_embeddings), dtype=np.bool)
hist_i = list(itertools.chain.from_iterable([ts[1] for ts in train_set]))
print(len(hist_i))
hist_img = img_list[hist_i]
print(hist_img)
img_train_mask[hist_img] = True
# 訓練データにあって存在する埋め込みだけをとってくる
masking = np.logical_and(img_train_mask, np.logical_not(img_missing_mask))
print('number of selected images to train TruncatedSVD: %d' % masking.sum())
train_img = image_embeddings[masking]
# 訓練データからtSVD(PCA)を訓練
image_tsvd = decomposition.TruncatedSVD(n_components=IMG_DIMINISHED_SIZE, random_state=1234)
train_img = image_tsvd.fit_transform(train_img)
# 画像が存在しない場合の代わり
image_placeholder = np.mean(train_img, axis=0)
# メモリ対策
train_img = None
# 次元圧縮
image_converted = image_tsvd.transform(image_embeddings)
# 存在しない画像の場合はプレースホルダで置き換え
image_converted[img_missing_mask] = image_placeholder

# シャフル
random.shuffle(train_set)
random.shuffle(test_set)

assert len(test_set) == user_count
# assert(len(test_set) + len(train_set) // 2 == reviews_df.shape[0])

with open('../raw_data/text_embeddings.pkl', 'rb') as f:
  r = pickle.load(f)

# train_set・test_setは一つ一つのレビュー履歴についてユーザID、レビューしたアイテムID、これ以前にレビューしたアイテム、過去の履歴との時間差を含む
with open('dataset.pkl', 'wb') as f:
  pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump((user_count, item_count, cate_count), f, pickle.HIGHEST_PROTOCOL)
  pickle.dump((img_list, image_converted), f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(r, f, pickle.HIGHEST_PROTOCOL)
