import os
import json
import numpy as np
import tensorflow.compat.v1 as tf



class Model(object):
  """モデルのクラスを定義"""

  def __init__(self, config, cate_list):
    """config：設定、cate_list：商品のカテゴリ（ASINのID順）"""
    self.config = config

    # Summary Writer
    self.train_writer = tf.summary.FileWriter(config['model_dir'] + '/train')
    self.eval_writer = tf.summary.FileWriter(config['model_dir'] + '/eval')

    # Building network
    self.init_placeholders()
    self.build_model(cate_list)
    self.init_optimizer()


  def init_placeholders(self):
    """
    プレースホルダーの定義
    プレースホルダーは変数Variableと違い、初期化時にインスタンスを与える必要がない。
    セッション毎にデータが与えられる場合に用いられる。
    """
    # [B] user id
    self.u = tf.placeholder(tf.int32, [None,])

    # [B] item id
    self.i = tf.placeholder(tf.int32, [None,])

    # [B] item label
    self.y = tf.placeholder(tf.float32, [None,])
    
    # [B, T] user's history item id
    self.hist_i = tf.placeholder(tf.int32, [None, None])

    # [B, T] user's history item purchase time
    self.hist_t = tf.placeholder(tf.int32, [None, None])

    self.im = tf.placeholder(tf.float32, [None, None, self.config['input_image_emb_size']])

    self.r = tf.placeholder(tf.float32, [None, None, self.config['input_text_emb_size']])

    # [B] valid length of `hist_i`
    self.sl = tf.placeholder(tf.int32, [None,])

    # learning rate
    # float64と32同士の演算を行うとエラーになるため、学習率をfloat32に変える
    # self.lr = tf.placeholder(tf.float64, [])
    self.lr = tf.placeholder(tf.float32, [])

    # whether it's training or not
    self.is_training = tf.placeholder(tf.bool, [])


  def build_model(self, cate_list):
    """モデルの構築"""

    modal_emb_size = self.config['modal_embedding_size']
    dropout_rate = self.config['dropout']

    # 変数の定義
    # 商品の埋め込み表現が保存される行列 [|I|, di]
    # ２次元のルックアップテーブル
    item_emb_w = tf.get_variable(
        "item_emb_w",
        [self.config['item_count'], modal_emb_size])
    # 類似度のバイアスのベクトル [|I|]
    item_b = tf.get_variable(
        "item_b",
        [self.config['item_count'],],
        initializer=tf.constant_initializer(0.0))
    # カテゴリの埋め込み表現が保存される行列 [|A|, da]
    # ２次元のルックアップテーブル
    cate_emb_w = tf.get_variable(
        "cate_emb_w",
        [self.config['cate_count'], modal_emb_size])
    # 各商品のIDとカテゴリIDのマップ（リスト） [|I|]
    cate_list = tf.convert_to_tensor(cate_list, dtype=tf.int32)
    # 時間の埋め込み表現
    time_emb_w = tf.get_variable(
        "time_emb_w",
        (self.config['time_gap_categoized'], modal_emb_size))

    # アイテム埋め込みとカテゴリ埋め込みと時間の埋め込みを結合、それをDenseで写像する
    # 論文：p3左のu_ij=h_emb
    # 予測すべきアイテムの埋め込み表現 [B, 2d]
    i_emb = tf.concat([
        tf.nn.embedding_lookup(item_emb_w, self.i),
        tf.nn.embedding_lookup(cate_emb_w, tf.gather(cate_list, self.i)),
        ], 1)
    # 予測すべきアイテムの重み [B]
    i_b = tf.gather(item_b, self.i)

    # embedding_lookupでルックアップテーブルから該当する埋め込み表現を持ってくる
    item_emb = tf.nn.embedding_lookup(item_emb_w, self.hist_i) # [B, T, d]
    item_emb = tf.expand_dims(item_emb, 2) # [B, T, 1, d]
    
    cat_emb = tf.nn.embedding_lookup(cate_emb_w, tf.gather(cate_list, self.hist_i)) # [B, T, d]
    cat_emb = tf.expand_dims(cat_emb, 2) # [B, T, 1, d]

    img_emb = tf.layers.dense(self.im, modal_emb_size, activation=tf.nn.relu) # [B, T, d]
    img_emb = tf.layers.dropout(img_emb, rate=dropout_rate, training=tf.convert_to_tensor(self.is_training))
    img_emb = tf.expand_dims(img_emb, 2) # [B, T, 1, d]

    text_emb = tf.layers.dense(self.r, modal_emb_size, activation=tf.nn.relu) # [B, T, d]
    text_emb = tf.expand_dims(text_emb, 2) # [B, T, 1, d]

    t_emb = tf.nn.embedding_lookup(time_emb_w, self.hist_t) # [B, T, d]
    t_emb = tf.expand_dims(t_emb, 2) # [B, T, 1, d]

    # [B, T, M, d]
    h_emb = tf.concat((item_emb, cat_emb, img_emb, text_emb, t_emb), axis=2)

    # アテンション機構を重ねる数
    num_blocks = self.config['num_blocks']
    num_heads = self.config['num_heads']

    # トランスフォーマー
    # 論文：p4左数式(3)
    # u_emb [B, C]
    u_emb, self.att, self.stt = transformer(
        # uij
        h_emb,
        # ユーザーの履歴の長さ
        self.sl,
        # デコーダーへの入力
        i_emb,
        num_blocks,
        num_blocks,
        num_heads,
        dropout_rate,
        self.is_training,
        False)

    # 予測
    # 論文：p4右数式(7)&(8) f(h_t, et_u) reduce_sum([B, C]) [B]
    self.logits = i_b + tf.reduce_sum(tf.multiply(u_emb, i_emb), 1)

    # ============== Eval ===============
    self.eval_logits = self.logits
  
    # Step variable
    self.global_step = tf.Variable(0, trainable=False, name='global_step')
    self.global_epoch_step = \
        tf.Variable(0, trainable=False, name='global_epoch_step')
    self.global_epoch_step_op = \
        tf.assign(self.global_epoch_step, self.global_epoch_step+1)

    # Loss
    # L2正規化
    l2_norm = tf.add_n([
        tf.nn.l2_loss(u_emb),
        tf.nn.l2_loss(i_emb),
        ])

    # ロス定義、ペアワイズ、シグモイド相互情報量
    self.loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.logits,
            labels=self.y)
        ) + self.config['regulation_rate'] * l2_norm

    self.train_summary = tf.summary.merge([
        tf.summary.histogram('embedding/1_item_emb', item_emb_w),
        tf.summary.histogram('embedding/2_cate_emb', cate_emb_w),
        tf.summary.histogram('embedding/3_time_raw', self.hist_t),
        tf.summary.histogram('embedding/3_time_dense', t_emb),
        tf.summary.histogram('embedding/4_final', h_emb),
        tf.summary.histogram('attention_output', u_emb),
        tf.summary.scalar('L2_norm Loss', l2_norm),
        tf.summary.scalar('Training Loss', self.loss),
        ])


  def init_optimizer(self):
    """最適化アルゴリズムの設定"""
    # 論文だとSGDを利用、lrは1.0→0.1に変化するようになっている
    # Gradients and SGD update operation for training the model
    trainable_params = tf.trainable_variables()
    if self.config['optimizer'] == 'adadelta':
      self.opt = tf.train.AdadeltaOptimizer(learning_rate=self.lr)
    elif self.config['optimizer'] == 'adam':
      self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)
    elif self.config['optimizer'] == 'rmsprop':
      self.opt = tf.train.RMSPropOptimizer(learning_rate=self.lr)
    else:
      self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)

    # Compute gradients of loss w.r.t. all trainable variables
    gradients = tf.gradients(self.loss, trainable_params)

    # Clip gradients by a given maximum_gradient_norm
    clip_gradients, _ = tf.clip_by_global_norm(
        gradients, self.config['max_gradient_norm'])

    # Update the model
    self.train_op = self.opt.apply_gradients(
        zip(clip_gradients, trainable_params), global_step=self.global_step)



  def train(self, sess, uij, l, add_summary=False):
    """行動とラベルを入力して学習する"""

    # uij = (u, i, j, hist_i, hist_t, sl, im, r)
    input_feed = {
        self.u: uij[0],
        self.i: uij[1],
        self.y: uij[2],
        self.hist_i: uij[3],
        self.hist_t: uij[4],
        self.sl: uij[5],
        self.im: uij[6],
        self.r: uij[7],
        self.lr: l,
        self.is_training: True,
        }

    output_feed = [self.loss, self.train_op]

    if add_summary:
      output_feed.append(self.train_summary)

    outputs = sess.run(output_feed, input_feed)

    if add_summary:
      self.train_writer.add_summary(
          outputs[2], global_step=self.global_step.eval())

    return outputs[0]

  def eval(self, sess, uij):
    """モデルの評価を行う"""
    res1 = sess.run(self.eval_logits, feed_dict={
        self.u: uij[0],
        self.i: uij[1],
        self.hist_i: uij[3],
        self.hist_t: uij[4],
        self.sl: uij[5],
        self.im: uij[6],
        self.r: uij[7],
        self.is_training: False,
        })
    res2 = sess.run(self.eval_logits, feed_dict={
        self.u: uij[0],
        self.i: uij[2],
        self.hist_i: uij[3],
        self.hist_t: uij[4],
        self.sl: uij[5],
        self.im: uij[6],
        self.r: uij[7],
        self.is_training: False,
        })
    return np.mean(res1 - res2 > 0)

  def test(self, sess, uij):
    """uijを使ってテスト用の結果を生成"""
    res1, att_1, stt_1 = sess.run([self.eval_logits, self.att, self.stt], feed_dict={
        self.u: uij[0],
        self.i: uij[1],
        self.hist_i: uij[3],
        self.hist_t: uij[4],
        self.sl: uij[5],
        self.im: uij[6],
        self.r: uij[7],
        self.is_training: False,
        })
    res2, att_2, stt_2 = sess.run([self.eval_logits, self.att, self.stt], feed_dict={
        self.u: uij[0],
        self.i: uij[2],
        self.hist_i: uij[3],
        self.hist_t: uij[4],
        self.sl: uij[5],
        self.im: uij[6],
        self.r: uij[7],
        self.is_training: False,
        })
    return res1, res2, att_1, stt_1, att_2, stt_1


     
  def save(self, sess):
    checkpoint_path = os.path.join(self.config['model_dir'], 'atrank')
    saver = tf.train.Saver()
    save_path = saver.save(
        sess, save_path=checkpoint_path, global_step=self.global_step.eval())
    json.dump(self.config,
              open('%s-%d.json' % (checkpoint_path, self.global_step.eval()), 'w'),
              indent=2)
    print('model saved at %s' % save_path, flush=True)

  def restore(self, sess, path):
    saver = tf.train.Saver()
    saver.restore(sess, save_path=path)
    print('model restored from %s' % path, flush=True)


def transformer(enc, sl, dec, enc_blocks, dec_blocks, dec_heads, dropout_rate, is_training, reuse):
  """
  トランスフォーマー
  論文：p4 Self-Attention Layer
  enc：履歴のバッチ [B, T, M, di+da+dt or di+da]
  sl：各履歴の長さ [B, T]
  dec：デコーダーへの入力 [B, di+da]
  """
  with tf.variable_scope("all", reuse=reuse):
    with tf.variable_scope("encoder"):
      # エンコーダー
      for i in range(enc_blocks):
        with tf.variable_scope("block_{}".format(i)):
          # セルフマルチヘッドアテンション
          ### Multihead Attention
          # enc [B, Tq, M, C]
          enc, stt_vec = modal_head_attention(queries=enc,
              queries_length=sl,
              keys=enc,
              keys_length=sl,
              dropout_rate=dropout_rate,
              is_training=is_training)

    # enc [B, T, C*M]
    enc = tf.reshape(enc, (tf.shape(enc)[0], tf.shape(enc)[1], enc.get_shape()[2]*enc.get_shape()[3]))
    # dec [B, 1, di+da]
    dec = tf.expand_dims(dec, 1)
    with tf.variable_scope("decoder"):
      # デコーダー
      for i in range(dec_blocks):
        with tf.variable_scope("block_{}".format(i)):
          ## Multihead Attention ( vanilla attention)
          # decを使ってencにアテンション
          # dec [B, 1, C]
          dec, att_vec = multihead_attention(queries=dec,
              queries_length=tf.ones_like(dec[:, 0, 0], dtype=tf.int32),
              keys=enc,
              keys_length=sl,
              num_heads=dec_heads,
              dropout_rate=dropout_rate,
              is_training=is_training,
              scope="vanilla_attention")

    # [B, C]
    dec = tf.squeeze(dec)
    return dec, att_vec, stt_vec


def modal_dense(input_tensor,
            num_units,
            activation=None,
            scope="modal_dense",
            reuse=None):
  """各モーダルに対して異なるDenseを適用します。

  Args:
    input_tensor: [B, T, M, C]の形をした最低2次元のテンソルです。
    scope: スコープ。
    num_units: 出力のベクトルの長さ。O
    reuse: 重みを再利用するかどうか。
  """
  with tf.variable_scope(scope, reuse=reuse):
    s = input_tensor.get_shape().as_list()
    # 入力のベクトルの長さ　モーダル数
    input_size, modality = s[-1], s[-2]
    # 重みとバイアステンソル
    # [M, O, C]
    weight = tf.get_variable('weight', (modality, num_units, input_size))
    # [M, O]
    bias = tf.get_variable('bias', (modality, num_units))

    # [B, T, M, O]
    output = tf.einsum('mik,btmk->btmi', weight, input_tensor)
    output += bias

    # アクティベーションが指定されているなら適用
    if activation is not None:
      output = activation(output)

    return output


def leave_one_dense(input_tensor,
            num_units,
            activation=None,
            scope='leave_one_dense',
            reuse=None):
  """各モーダルに対して、それ以外のモーダルをDenseに通します。
  例：M=3 (Ma, Mb, Mc)
  output = [dense(input[Mb]⊕input[Mc]), dense(input[Ma]⊕input[Mc]), dense(input[Ma]⊕input[Mb])]

  Args:
    input_tensor: [N, T, M, C]の形をした4次元入力のテンソルです。
    scope: スコープ。
    num_units: 出力のベクトルサイズ。
    reuse: 重みを再利用するかどうか。
  """
  with tf.variable_scope(scope, reuse=reuse):
    batch_size = tf.shape(input_tensor)[0]
    series_size = tf.shape(input_tensor)[1]
    s = input_tensor.get_shape().as_list()
    vector_size = s[-1]
    modality = s[-2]
    
    # [N, T, M-1, C]*M
    output = [tf.stack([input_tensor[:, :, j] for j in range(modality) if i != j], 2) for i in range(modality)]
    output = tf.stack(output, 2) # [N, T, M, M-1, C]

    # [N, T, M, (M-1)*C]
    output = tf.reshape(output, (batch_size, series_size, modality, (modality-1)*vector_size))

    # [N, T, M, O]
    output = modal_dense(output, num_units, activation=activation, reuse=reuse)

    return output


def modal_head_attention(queries,
            queries_length,
            keys,
            keys_length,
            dropout_rate=0,
            is_training=True,
            scope="modal_head_attention",
            reuse=None):
  '''Applies multihead attention.

  Args:
    queries: A 4d tensor with shape of [N, T_q, M, C_q].
    queries_length: A 1d tensor with shape of [N].
    keys: A 4d tensor with shape of [N, T_k, M, C_k].
    keys_length:  A 1d tensor with shape of [N].
    dropout_rate: A floating point number.
    is_training: Boolean. Controller of mechanism for dropout.
    scope: Optional scope for `variable_scope`.
    reuse: Boolean, whether to reuse the weights of a previous layer
    by the same name.

  Returns
    A 4d tensor with shape of (N, T_q, M, C)
  '''
  with tf.variable_scope(scope, reuse=reuse):
    q_shape = queries.get_shape().as_list()
    # ベクトルの次元
    num_units = q_shape[-1]
    # モーダルの数
    modality = q_shape[-2]

    # queries ----> leave-one-dense(relu) ----> Q
    #
    # keys    --+-> modal-dense(relu)     ----> K
    #           |
    #           +-> modal-dense(relu)     ----> V
    # Linear projections, C = # dim or column, T_x = # vectors or actions
    Q = modal_dense(queries, num_units, activation=tf.nn.relu, scope='query_dense', reuse=reuse)  # (N, T_q, M, C)
    K = leave_one_dense(keys, num_units, activation=tf.nn.relu, scope='key_dense', reuse=reuse)  # (N, T_k, M, C)
    V = leave_one_dense(keys, num_units, activation=tf.nn.relu, scope='value_dense', reuse=reuse)  # (N, T_k, M, C)

    # アテンションスコアの算出
    # Multiplication
    # query-key score matrix
    # each big score matrix is then split into h score matrix with same size
    # w.r.t. different part of the feature
    att = tf.einsum('bqmi,bkni->bqmkn', Q, K) # (N, T_q, M, T_k, M)
    
    # Key Masking
    # keysは可変の長さを持つ(keys_lengthで定義)、値のある要素だけを示すマスクを生成する
    key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])   # (N, T_k)
    # 複製する
    key_masks = tf.expand_dims(key_masks, 1)  # (N, 1, T_k)
    key_masks = tf.expand_dims(key_masks, 2)  # (N, 1, 1, T_k)
    key_masks = tf.expand_dims(key_masks, 4)  # (N, 1, 1, T_k, 1)
    key_masks = tf.tile(key_masks, [1, tf.shape(queries)[1], modality, 1, modality])  # (N, T_q, M, T_k, M)
    # -2^32+1は無限大として考える、attと同じ形を持つすべての要素が無限大の行列paddingsを作る
    paddings = tf.ones_like(att) * (-2 ** 32 + 1)
    # attにマスクを適用する、key_masksがTrueの要素はoutputsで、Falseはpaddingsで塗りつぶす
    att = tf.where(key_masks, att, paddings)  # (N, T_q, M, T_k, M)

    # 文章のトランスフォーマーとは違い、予測に必要な以外の将来の行動をマスクする必要はないので実装していない
    # Causality = Future blinding: No use, removed

    # Softmaxを最後の2ランクに適用
    exp = tf.exp(att) # (N, Tq, M, Tk, M)
    reduced = tf.reduce_sum(exp, axis=(-1, -2), keepdims=True) # (N, T_q, M)
    att = exp / reduced # (N, T_q, M, T_k, M)

    # keysのマスキングは行ったが、queryのマスキングは行っていないので行う
    # Query Masking
    query_masks = tf.sequence_mask(queries_length, tf.shape(queries)[1], dtype=tf.float32)   # (N, T_q)
    query_masks = tf.expand_dims(query_masks, 2)  # (N, T_q, 1)
    query_masks = tf.expand_dims(query_masks, 3)  # (N, T_q, 1, 1)
    query_masks = tf.expand_dims(query_masks, 4)  # (N, T_q, 1, 1, 1)
    query_masks = tf.tile(query_masks, (1, 1, modality, tf.shape(keys)[1], modality))  # (N, T_q, M, T_k, M)
    att *= query_masks  # broadcasting. (N, T_q, M, T_k, M)

    # Dropouts
    outputs = tf.layers.dropout(att, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

    # Weighted sum
    outputs = tf.einsum('bqni,bqmkn->bqmi', V, outputs) # (N, T_q, M, C)
    
    # Residual connection
    outputs += queries

    # Normalize and Feed Forward
    unstacked = tf.unstack(outputs, axis=2)
    unstacked = [normalize(u, scope='norm_%d' % i) for i, u in enumerate(unstacked)]
    unstacked = [feedforward(u,
                  num_units=[num_units // 4, num_units],
                  scope='ff_modal_%d' % i,
                  reuse=reuse) for i, u in enumerate(unstacked)]
    outputs = tf.stack(unstacked, axis=2)

  return outputs, att


def multihead_attention(queries,
            queries_length,
            keys,
            keys_length,
            num_heads=8,
            dropout_rate=0,
            is_training=True,
            scope="multihead_attention",
            reuse=None):
  '''Applies multihead attention.

  Args:
    queries: A 3d tensor with shape of [N, T_q, C_q].
    queries_length: A 1d tensor with shape of [N].
    keys: A 3d tensor with shape of [N, T_k, C_k].
    keys_length:  A 1d tensor with shape of [N].
    num_units: A scalar. Attention size.
    dropout_rate: A floating point number.
    is_training: Boolean. Controller of mechanism for dropout.
    num_heads: An int. Number of heads.
    scope: Optional scope for `variable_scope`.
    reuse: Boolean, whether to reuse the weights of a previous layer
    by the same name.

  Returns
    A 3d tensor with shape of (N, T_q, C)
  '''
  with tf.variable_scope(scope, reuse=reuse):
    num_units = queries.get_shape().as_list()[-1]

    # queries ----> dense(relu) ----> Q
    #
    # keys    --+-> dense(relu) ----> K
    #           |
    #           +-> dense(relu) ----> V
    # Linear projections, C = # dim or column, T_x = # vectors or actions
    Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
    K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
    V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)

    # ヘッドで分割
    # Split and concat
    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

    # アテンションスコアの算出（QK^T）
    # Multiplication
    # query-key score matrix
    # each big score matrix is then split into h score matrix with same size
    # w.r.t. different part of the feature
    outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

    # sqrt(C/h)で割る
    # Scale
    outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

    # Key Masking
    # keysは可変の長さを持つ(keys_lengthで定義)、値のある要素だけを示すマスクを生成する
    key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])   # (N, T_k)
    # ヘッドの数だけ複製する
    key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
    # クエリの数だけマスクを複製
    key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

    # -2^32+1は無限大として考える、outputsと同じ形を持つすべての要素が無限大の行列paddingsを作る
    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
    # outputsにマスクを適用する、key_masksがTrueの要素はoutputsで、Falseはpaddingsで塗りつぶす
    outputs = tf.where(key_masks, outputs, paddings)  # (h*N, T_q, T_k)

    # 文章のトランスフォーマーとは違い、予測に必要な以外の将来の行動をマスクする必要はないので実装していない
    # Causality = Future blinding: No use, removed

    # 最後の次元、T_kのaxisでsoftmaxが実行される
    # Activation
    outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

    # keysのマスキングは行ったが、queryのマスキングは行っていないので行う
    # Query Masking
    query_masks = tf.sequence_mask(queries_length, tf.shape(queries)[1], dtype=tf.float32)   # (N, T_q)
    query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
    query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
    outputs *= query_masks  # broadcasting. (h*N, T_q, T_k)

    # Attention vector
    att_vec = outputs

    # Dropouts
    outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

    # Weighted sum
    outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

    # Restore shape
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

    # Residual connection
    outputs += queries

    # Normalize
    outputs = normalize(outputs)  # (N, T_q, C)

    ## Feed Forward
    outputs = feedforward(outputs, num_units=[num_units // 4, num_units], reuse=reuse)

  return outputs, att_vec

def feedforward(inputs,
        num_units=[2048, 512],
        scope="feedforward",
        reuse=None):
  '''Point-wise feed forward net.

  Args:
    inputs: A 3d tensor with shape of [N, T, C].
    num_units: A list of two integers.
    scope: Optional scope for `variable_scope`.
    reuse: Boolean, whether to reuse the weights of a previous layer
    by the same name.

  Returns:
    A 3d tensor with the same shape and dtype as inputs
  '''
  with tf.variable_scope(scope, reuse=reuse):
    # Inner layer
    params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
          "activation": tf.nn.relu, "use_bias": True}
    outputs = tf.layers.conv1d(**params)

    # Readout layer
    params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
          "activation": None, "use_bias": True}
    outputs = tf.layers.conv1d(**params)

    # Residual connection
    outputs += inputs

    # Normalize
    outputs = normalize(outputs)

  return outputs  # [N, T, C]

def normalize(inputs,
        epsilon=1e-8,
        scope="normalize",
        reuse=None):
  '''Applies layer normalization.

  Args:
    inputs: A tensor with 2 or more dimensions, where the first dimension has
    `batch_size`.
    epsilon: A floating number. A very small number for preventing ZeroDivision Error.
    scope: Optional scope for `variable_scope`.
    reuse: Boolean, whether to reuse the weights of a previous layer
    by the same name.

  Returns:
    A tensor with the same shape and data dtype as `inputs`.
  '''
  with tf.variable_scope(scope, reuse=reuse):
    # 入力の形
    inputs_shape = inputs.get_shape()
    params_shape = inputs_shape[-1:]

    # 入力の平均と分散
    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
    # バイアス
    beta = tf.Variable(tf.zeros(params_shape))
    # 重み
    gamma = tf.Variable(tf.ones(params_shape))
    # 正規化、eplisonは0除算を防ぐためのパラメーター
    # TODO gamma、betaはブロードキャストされる？
    normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
    outputs = gamma * normalized + beta

  return outputs

# この関数は使われていない
# def extract_axis_1(data, ind):
#   batch_range = tf.range(tf.shape(data)[0])
#   indices = tf.stack([batch_range, ind], axis=1)
#   res = tf.gather_nd(data, indices)
#   return res

