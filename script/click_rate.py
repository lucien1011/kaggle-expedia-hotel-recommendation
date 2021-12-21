import os
import pandas as pd
import pickle

from data.tools import train_test_split

io_config = dict(
    input_csv_path='storage/train.csv',
    output_dir='storage/output/211220_click_rate/',
)

df = pd.read_csv(io_config['input_csv_path'])
df,_ = train_test_split(df)
click_rate = dict(df.hotel_cluster.value_counts() / len(df))
if not os.path.exists(io_config['output_dir']): os.makedirs(io_config['output_dir'])
pickle.dump(click_rate,open(os.path.join(io_config['output_dir'],'click_rate.p'),'wb'))
