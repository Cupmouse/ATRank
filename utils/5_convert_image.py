import torch
from torchvision import models, transforms as T
from PIL import Image
import os
import numpy as np
import pickle

# AlexNetを読み込み
alexnet = models.alexnet(pretrained=True)
# 分類層の1番目の出力を特徴ベクトルとして利用
alexnet.classifier = torch.nn.Sequential(*list(alexnet.classifier.children())[:3])
#alexnet.to('cuda')

def unflatten(l, batch_size):
  last_i = int(len(l)/batch_size)
  uf = [None]*(last_i+1)
  for i in range(last_i):
    uf[i] = l[batch_size*i:batch_size*(i+1)]
  uf[last_i] = l[batch_size*last_i:len(l)]
  return uf

preprocess = T.Compose([
  T.Resize(256),
  T.CenterCrop(224),
  T.ToTensor(),
  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 画像の一覧を取得
with open('../raw_data/remap_meta.pkl', 'rb') as f:
  pickle.load(f)
  pickle.load(f)
  pickle.load(f)
  pickle.load(f)
  img_list, img_key = pickle.load(f)

PREDICTION_BATCH_SIZE = 512

print('processing with batch size %d' % PREDICTION_BATCH_SIZE)

outputs = np.ndarray((len(img_key), 4096), dtype=np.float32)
# 画像が存在しない場合Trueになる行列
missing_mask = np.zeros(len(img_key), dtype=np.bool)

for i, batch in enumerate(unflatten(img_key, PREDICTION_BATCH_SIZE)):
  input_batch = torch.zeros([PREDICTION_BATCH_SIZE, 3, 224, 224], dtype=torch.float32)
  for j, key in enumerate(batch):
    path = '../raw_data/images/'+key
    if not os.path.exists(path):
      # ファイルが存在しないならスキップ
      missing_mask[PREDICTION_BATCH_SIZE*i+j] = True
      print('%s does not exist, skipping' % key)
      continue
    input_image = Image.open(path)
    try:
      input_batch[j] = preprocess(input_image)
    except KeyboardInterrupt as e:
      raise e
    except Exception as e:
      # グレースケールの可能性がある
      print('an error encountered while loading %s as an RGB image, might be an grayscale image, falling back...' % key)
      rgb_image = Image.new("RGB", input_image.size)
      rgb_image.paste(input_image)
      # ココでエラーが起きたら致命的、終了させる
      input_batch[j] = preprocess(rgb_image)
    
  if len(input_batch) != len(batch):
    # 最後のバッチではバッファーのサイズより小さい入力の可能性がある
    input_batch = input_batch[:len(batch)]

  # 入力テンソルをGPUへ
  #input_batch = input_batch.to('cuda')
  
  # パラメーターが変化しないようにAlexNetへ入力
  with torch.no_grad():
    output = alexnet(input_batch)
  
  # 結果を主メモリに移して大きい行列に保存
  outputs[PREDICTION_BATCH_SIZE*i:PREDICTION_BATCH_SIZE*i+len(output)] = output.cpu().numpy()
  print("batch %d processed" % i)

# 画像の埋め込み表現とそもそも画像が存在しない場合Trueが記録されている行列
with open('../raw_data/image_embeddings.pkl', 'wb') as f:
  pickle.dump(outputs, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(missing_mask, f, pickle.HIGHEST_PROTOCOL)
