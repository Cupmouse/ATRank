#データセットのファイルを読み込み、埋め込みしやすい形に直して保存する
import random
import pickle
import numpy as np

random.seed(1234)

#user_count, item_count, cate_count, example_countはそれぞれユーザ、アイテム、カテゴリ、レビュー履歴の数
with open('../raw_data/remap.pkl', 'rb') as f:
  reviews_df = pickle.load(f)
  cate_list = pickle.load(f)
  user_count, item_count, cate_count, example_count = pickle.load(f)

# [1, 2) = 0, [2, 4) = 1, [4, 8) = 2, [8, 16) = 3...  need len(gap) hot
gap = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
# gap = [2, 7, 15, 30, 60,]
# gap.extend( range(90, 4000, 200) )
# gap = np.array(gap)
print(gap.shape[0])

#タイムスタンプを連続値→カテゴリ特徴に変換
#gapで時間を区切る領域を指定し、領域ごとにカテゴリの数値を割り当てている。
def proc_time_emb(hist_t, cur_t):
  hist_t = [cur_t - i + 1 for i in hist_t]
  hist_t = [np.sum(i >= gap) for i in hist_t]
  return hist_t

train_set = []
test_set = []
#histは各reviewerIDについてのレビューデータ(reviewerID以外のカラム全て)
for reviewerID, hist in reviews_df.groupby('reviewerID'):
  pos_list = hist['asin'].tolist()
  tim_list = hist['unixReviewTime'].tolist()
  tim_list = [i // 3600 // 24 for i in tim_list]
  #ユーザがレビューしたアイテムにランダムな数値を割り当てる
  def gen_neg():
    neg = pos_list[0]
    while neg in pos_list:
      neg = random.randint(0, item_count-1)
    return neg
  neg_list = [gen_neg() for i in range(len(pos_list))]

  #レビュー履歴とそれ以前の全てのレビュー履歴との時間差をとってカテゴリ特徴に直す
  #一番最後のレビュー履歴はテストデータセットに、他は訓練データセットに入れている
  for i in range(1, len(pos_list)):
    hist_i = pos_list[:i]
    hist_t = proc_time_emb(tim_list[:i], tim_list[i])
    if i != len(pos_list) - 1:
      train_set.append((reviewerID, hist_i, hist_t, pos_list[i], 1))
      train_set.append((reviewerID, hist_i, hist_t, neg_list[i], 0))
    else:
      label = (pos_list[i], neg_list[i])
      test_set.append((reviewerID, hist_i, hist_t, label))

random.shuffle(train_set)
random.shuffle(test_set)

assert len(test_set) == user_count
# assert(len(test_set) + len(train_set) // 2 == reviews_df.shape[0])

#train_set・test_setは一つ一つのレビュー履歴についてユーザID、レビューしたアイテムID、これ以前にレビューしたアイテム、過去の履歴との時間差を含む
with open('dataset.pkl', 'wb') as f:
  pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump((user_count, item_count, cate_count), f, pickle.HIGHEST_PROTOCOL)
