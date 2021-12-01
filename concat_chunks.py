import numpy as np
import os
import pandas as pd

io_config = dict(
    input_csv_dir="/cmsuf/data/store/user/t2/users/klo/MiscStorage/ForLucien/Kaggle/expedia-hotel-recommendations/preprocess/211201_baseline/",
    output_csv_dir="/cmsuf/data/store/user/t2/users/klo/MiscStorage/ForLucien/Kaggle/expedia-hotel-recommendations/preprocess/211201_baseline/",
    nchunk=4,
)

if __name__ == "__main__":
    train = pd.concat([pd.read_csv(os.path.join(io_config['input_csv_dir'],'chunk{:d}_train.csv'.format(i)),index_col=0) for i in range(io_config['nchunk'])],axis=0)
    valid = pd.concat([pd.read_csv(os.path.join(io_config['input_csv_dir'],'chunk{:d}_valid.csv'.format(i)),index_col=0) for i in range(io_config['nchunk'])],axis=0)

    train.to_csv(os.path.join(io_config['output_csv_dir'],'train.csv'))
    valid.to_csv(os.path.join(io_config['output_csv_dir'],'valid.csv'))
