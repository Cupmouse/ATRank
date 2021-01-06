import os
from concurrent import futures
from urllib import request
import pickle

with open('../raw_data/meta.pkl', 'rb') as f:
  meta_df = pickle.load(f)
  imUrl = meta_df.imUrl

if not os.path.exists('../raw_data/images/'):
  os.makedirs('../raw_data/images/')

def download_image(url, timeout):
  """urlの画像をダウンロードする"""
  request.urlretrieve(url, "../raw_data/images/"+os.path.basename(url))

with futures.ThreadPoolExecutor(max_workers=5) as executor:
  future_to_url = {executor.submit(download_image, url, 60): url for url in imUrl}
  for future in futures.as_completed(future_to_url):
    url = future_to_url[future]
    try:
      data = future.result()
    except KeyboardInterrupt as ke:
      print(ke)
      exit(1)
    except Exception as exc:
      print('downloading %r failed: %s' % (url, exc))
