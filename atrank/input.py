import numpy as np

class DataInput:
  """
  整形済み学習・テストデータから指定されたバッチを生成
  Iterable & Iterator
  """
  def __init__(self, data, batch_size, imgs, img_list, texts):
    """
    data：学習またはテストデータ
    batch_size：バッチサイズ
    """
    self.batch_size = batch_size
    self.data = data
    self.imgs = imgs
    self.img_list = img_list
    self.texts = texts
    # エポック数の計算
    self.epoch_size = len(self.data) // self.batch_size
    if self.epoch_size * self.batch_size < len(self.data):
      self.epoch_size += 1
    # 現在のバッチインデックス
    self.i = 0

  def __iter__(self):
    """イテレーターとして自分自身を返す"""
    return self

  def __next__(self):
    """
    イテレーターが呼ばれた
    次のバッチを返す
    """
    if self.i == self.epoch_size:
      # データが尽きた
      raise StopIteration

    # このバッチに関係のあるデータをスライス
    # 「tsには行動の一覧」がリストになっている、一つの問題と同じ
    ts = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size, len(self.data))]
    # 次のバッチインデックスへ
    self.i += 1

    # 行列へ入れ込む
    # u:ユーザーID
    # hi:履歴
    # ht:履歴の時間
    # i:予測する商品のID(ラベル)
    # y:正例なら1、負例なら0
    # hr:テキストIDの履歴
    # sl:履歴の長さ
    u, hi, ht, i, y, hr = zip(*ts)
    sl = [len(h) for h in hi]
    max_sl = max(sl)
    # iは配列のインデックスに使われるのでリストでなければならない
    i = np.array(i, dtype=np.int32)
    
    # 最大の履歴の長さに合わせて行列を初期化する
    # hist_i:行動履歴
    hist_i = np.zeros([len(ts), max_sl], np.int32)
    # hist_t:行動の時間（hist_i[i]の時間）
    # 行動時間は最後の行動からの相対時間になっている
    hist_t = np.zeros([len(ts), max_sl], np.int32)
    # hist_r:行動のテキストID
    hist_r = np.zeros((len(ts), max_sl), np.int32)

    for j in range(len(ts)):
      hist_i[j, :len(hi[j])] = hi[j]
      hist_t[j, :len(ht[j])] = ht[j]
      hist_r[j, :len(hr[j])] = hr[j]

    im = self.imgs[self.img_list[hist_i]]
    r = self.texts[hist_r]
    im_i = self.imgs[self.img_list[i]]

    return self.i, (u, i, y, hist_i, hist_t, sl, im, r, im_i)

class DataInputTest:
  """DataInputのテストデータバージョン"""
  def __init__(self, data, batch_size, imgs, img_list, texts):

    # epoch_sizeを決定
    self.batch_size = batch_size
    self.data = data
    self.imgs = imgs
    self.img_list = img_list
    self.texts = texts
    self.epoch_size = len(self.data) // self.batch_size
    if self.epoch_size * self.batch_size < len(self.data):
      self.epoch_size += 1
    self.i = 0

  def __iter__(self):
    return self

  # 次の状態(i)とデータを出力
  def __next__(self):

    if self.i == self.epoch_size:
      raise StopIteration

    # 今回入力するデータを現在の状態iに従って持ってくる
    ts = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size, len(self.data))]
    # 次の状態
    self.i += 1

    # 今回入力するデータを整形
    # DataInputと違うのはこの部分
    # u:ユーザーID
    # i:正例のラベル(予測する商品のID)
    # j:負例のラベル(予測する商品のID)
    # hr:テキストIDの履歴
    # sl:履歴の長さ
    u, hi, ht, ij, hr = zip(*ts)
    sl = [len(h) for h in hi]
    max_sl = max(sl)
    i, j = zip(*ij)
    i, j = np.array(i, dtype=np.int32), np.array(j, dtype=np.int32)
    
    #hist_i、hist_tの形を入力長で固定させる
    hist_i = np.zeros([len(ts), max_sl], np.int32)
    hist_t = np.zeros([len(ts), max_sl], np.int32)
    hist_r = np.zeros((len(ts), max_sl), np.int32)
    
    for k in range(len(ts)):
      hist_i[k, :len(hi[k])] = hi[k]
      hist_t[k, :len(ht[k])] = ht[k]
      hist_r[k, :len(hr[k])] = hr[k]

    im = self.imgs[self.img_list[hist_i]]
    r = self.texts[hist_r]
    im_i = self.imgs[self.img_list[i]]
    im_j = self.imgs[self.img_list[j]]

    return self.i, (u, i, j, hist_i, hist_t, sl, im, r, im_i, im_j)
