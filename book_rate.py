import os
import pandas as pd
import pickle

from data.tools import train_test_split

io_config = dict(
    input_csv_path='storage/train_is_booking_category.csv',
    output_dir='storage/output/211220_book_rate/',
)

df = pd.read_csv(io_config['input_csv_path'])
df,_ = train_test_split(df)
book_rate = dict(df.hotel_cluster.value_counts() / len(df))
if not os.path.exists(io_config['output_dir']): os.makedirs(io_config['output_dir'])
pickle.dump(book_rate,open(os.path.join(io_config['output_dir'],'book_rate.p'),'wb'))
