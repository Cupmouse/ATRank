import csv
import pprint
import pandas as pd
import gzip
import json
import gensim
import seaborn as sns
import numpy as np

df = pd.read_pickle('reviews.pkl')
grouped_df = df.groupby('reviewerID').count()

#元のデータセットからランダムにユーザ15000人分を選択
grouped_df = grouped_df[0:15000].reset_index()

#データセットを選択したユーザ15000人分に縮小して保存
reviewer_list = grouped_df['reviewerID'].values.tolist()
small_df = df[df['reviewerID'].isin(reviewer_list)]
small_df.to_pickle('small_reviews.pkl')