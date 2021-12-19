import datetime
import numpy as np
import os
import pandas as pd
import pickle
import torch
from tqdm import tqdm

from utils import Timer

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('mode.chained_assignment', None)

n_hotel_cluster = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device ',device)

io_config = dict(
    input_csv_path='storage/train_is_booking_category.csv',
    fm_mapping_path='storage/output/211216_catvar_hotel_cluster/mapping.p',
    fm_embed_dim=4,
    output_csv_dir='storage/output/211218_baseline/',
    chunksize=int(5e5),
)

def explode_is_booking(is_booking,hotel_cluster,):
    out = np.zeros(n_hotel_cluster)
    if is_booking:
        out[hotel_cluster] = 1
    return out

def read_fm_mapping(path):
    return pickle.load(open(path,'rb'))

def baseline_preprocess(df, progressbar=True):

    timer = Timer()

    tqdm.write(f'Processing timestamp features')
    timer.start()
    df['srch_id'] = df.apply(lambda x: '_'.join([str(x['date_time']),str(x['user_id']),str(x['srch_destination_id'])]),axis=1)
    df["srch_ci"] = pd.to_datetime(df["srch_ci"])
    df["srch_co"] = pd.to_datetime(df["srch_co"])
    df['year'] = df['date_time'].apply(lambda x: x.year)
    df['month'] = df['date_time'].apply(lambda x: x.month)
    df['srch_duration'] = (df["srch_co"] - df["srch_ci"]).dt.days
    df['srch_trip_time_diff'] = (df["srch_co"] - df["date_time"]).dt.days
    timer.print_reset()

    tqdm.write(f'Processing book/click-rate features')
    timer.start()
    booking_rate = dict(df.hotel_cluster.value_counts() / len(df))
    df['book_rate'] = df['hotel_cluster'].apply(lambda x: booking_rate[x])
    df.fillna(value=-1,inplace=True)
    timer.print_reset()

    tqdm.write(f'Explode with is_booking and hotel_cluster')
    timer.start()
    df['is_booking'] = df.apply(lambda x: explode_is_booking(x['is_booking'],x['hotel_cluster']),axis=1)
    df['hotel_cluster'] = df.apply(lambda x: np.arange(n_hotel_cluster),axis=1)    
    df = df.explode(['is_booking','hotel_cluster']).reset_index()
    df['is_booking'] = df['is_booking'].astype(np.int64)
    df['hotel_cluster'] = df['hotel_cluster'].astype(np.int64)
    timer.print_reset()

    tqdm.write(f'fm mapping on categorical features')
    timer.start()
    fm_mapping = read_fm_mapping(io_config['fm_mapping_path'])
    timer.print_reset()
    category_columns = [
        'srch_destination_id','srch_destination_type_id',
        'user_id','user_location_country','user_location_region','user_location_city',
        'site_name','channel',
        'is_mobile','is_package',
        ]
    for column in category_columns:
        timer.start()
        tqdm.write(f'Processing embedding for column {column}')
        df[column+f'_embedvec'] = df[['hotel_cluster',column]].apply(lambda x: fm_mapping[x['hotel_cluster']][column][x[column]],axis=1)
        df = pd.concat([df,pd.DataFrame(df[column+f'_embedvec'].tolist(), columns=[column+f'_dim{embeddim}' for embeddim in range(io_config['fm_embed_dim'])])],axis=1)
        timer.print_reset()
    
    df = df[
        [
            'date_time',
            'year',
            'month',
            'srch_id',
            'hotel_market',
            'orig_destination_distance',
            'srch_adults_cnt',
            'srch_children_cnt',
            'srch_rm_cnt',
            'srch_duration',
            'srch_trip_time_diff',
            'book_rate',
            'hotel_cluster',
            'is_booking',
        ] + [ column+f'_dim{embed_dim}' for embed_dim in range(io_config['fm_embed_dim']) for column in category_columns]
    ]
    
    return df

def train_test_split(df,timestamp=(2014,8,1)):
    print('train test split with datatime')
    train_test_timestamp = pd.Timestamp(datetime.datetime(*timestamp))
    df['date_time'] = pd.to_datetime(df['date_time'])
    X_train_inds = df.date_time < train_test_timestamp
    X_test_inds = df.date_time > train_test_timestamp
    print('baseline_preprocess train data')
    train_data = baseline_preprocess(df[X_train_inds])
    print('baseline_preprocess validation data')
    valid_data = baseline_preprocess(df[X_test_inds])
    return train_data,valid_data

if __name__ == "__main__":
    if 'nsample' in io_config:
        df = pd.read_csv(io_config['input_csv_path']).sample(io_config['nsample'])
        if not os.path.exists(io_config['output_csv_dir']):
            os.makedirs(io_config['output_csv_dir'])
        train,valid = train_test_split(df)
        train.to_csv(os.path.join(io_config['output_csv_dir'],'train.csv'))
        valid.to_csv(os.path.join(io_config['output_csv_dir'],'valid.csv'))
    elif 'chunksize' in io_config:
        dfs = pd.read_csv(io_config['input_csv_path'],chunksize=io_config['chunksize'])
        if not os.path.exists(io_config['output_csv_dir']):
            os.makedirs(io_config['output_csv_dir'])
        for i,df in enumerate(dfs):
            train,valid = train_test_split(df)
            train.to_csv(os.path.join(io_config['output_csv_dir'],'chunk{:d}_train.csv'.format(i)))
            valid.to_csv(os.path.join(io_config['output_csv_dir'],'chunk{:d}_valid.csv'.format(i)))
