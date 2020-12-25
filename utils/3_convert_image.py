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
alexnet.to('cuda')

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

imagepaths = os.listdir('../raw_data/images/')

# "no image available" image
# imagepaths.remove('no-img-sm._CB192198896_.gif')
# "update page" button image
# imagepaths.remove('update-page._CB192192236_.gif')

PREDICTION_BATCH_SIZE = 512

outputs = np.ndarray((len(imagepaths), 4096))

for i, batch in enumerate(unflatten(imagepaths, PREDICTION_BATCH_SIZE)):
  input_batch = torch.zeros([PREDICTION_BATCH_SIZE, 3, 224, 224], dtype=torch.float32)
  for j, filename in enumerate(batch):
    input_image = Image.open('../raw_data/images/'+filename)
    try:
      input_batch[j] = preprocess(input_image)
    except KeyboardInterrupt as e:
      raise e
    except Exception as e:
      # グレースケールの可能性がある
      print('%s produced error, might be an grayscale image, failling back...')
      rgb_image = Image.new("RGB", input_image.size)
      rgb_image.paste(input_image)
      try:
        input_batch[j] = preprocess(rgb_image)
      except KeyboardInterrupt as e:
        raise e
      except Exception as e:
        print("skipping:" + filename)
        print(e)
    
  if len(input_batch) != len(batch):
    # 最後のバッチではバッファーのサイズより小さい入力の可能性がある
    input_batch = input_batch[:len(batch)]

  # 入力テンソルをGPUへ
  input_batch = input_batch.to('cuda')
  
  # パラメーターが変化しないようにAlexNetへ入力
  with torch.no_grad():
    output = alexnet(input_batch)
  
  outputs[PREDICTION_BATCH_SIZE*i:PREDICTION_BATCH_SIZE*i+len(output)] = output.cpu().numpy()
  print("batch %d processed" % i)

embeddings = {filename: outputs[i] for i, filename in enumerate(imagepaths)}

with open('../raw_data/image_embeddings.pkl', 'wb') as f:
  pickle.dump(embeddings, f, pickle.HIGHEST_PROTOCOL)
