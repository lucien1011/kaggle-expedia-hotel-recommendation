import datetime
import numpy as np
import os
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('mode.chained_assignment', None)

n_hotel_cluster = 100

io_config = dict(
    input_csv_path='/cmsuf/data/store/user/t2/users/klo/MiscStorage/ForLucien/Kaggle/expedia-hotel-recommendations/train_is_booking.csv',
    output_csv_dir='/cmsuf/data/store/user/t2/users/klo/MiscStorage/ForLucien/Kaggle/expedia-hotel-recommendations/preprocess/211201_baseline/',
    chunksize=int(1e6),
)

def explode_is_booking(is_booking,hotel_cluster,):
    out = np.zeros(n_hotel_cluster)
    if is_booking:
        out[hotel_cluster] = 1
    return out

def baseline_preprocess(df):
    df['srch_id'] = df.apply(lambda x: '_'.join([str(x['date_time']),str(x['user_id']),str(x['srch_destination_id'])]),axis=1)
    df["srch_ci"] = pd.to_datetime(df["srch_ci"])
    df["srch_co"] = pd.to_datetime(df["srch_co"])
    df['year'] = df['date_time'].apply(lambda x: x.year)
    df['month'] = df['date_time'].apply(lambda x: x.month)
    df['srch_duration'] = (df["srch_co"] - df["srch_ci"]).dt.days
    df['srch_trip_time_diff'] = (df["srch_co"] - df["date_time"]).dt.days

    booking_rate = dict(df.hotel_cluster.value_counts() / len(df))
    df['book_rate'] = df['hotel_cluster'].apply(lambda x: booking_rate[x])
    df.fillna(value=-1,inplace=True)
    
    df['is_booking'] = df.apply(lambda x: explode_is_booking(x['is_booking'],x['hotel_cluster']),axis=1)
    df['hotel_cluster'] = df.apply(lambda x: np.arange(n_hotel_cluster),axis=1)
    
    df = df.explode(['is_booking','hotel_cluster'])
    df['is_booking'] = df['is_booking'].astype(np.int64)
    df['hotel_cluster'] = df['hotel_cluster'].astype(np.int64)
    
    df = df[
        [
            'date_time',
            'year',
            'month',
            'srch_id',
            'user_id',
            'hotel_market',
            'orig_destination_distance',
            'is_mobile',
            'is_package',
            'channel',
            'srch_adults_cnt',
            'srch_children_cnt',
            'srch_rm_cnt',
            'srch_duration',
            'srch_trip_time_diff',
            'book_rate',
            'hotel_cluster',
            'is_booking',
        ]
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
    dfs = pd.read_csv(io_config['input_csv_path'],chunksize=io_config['chunksize'])
    if not os.path.exists(io_config['output_csv_dir']):
        os.makedirs(io_config['output_csv_dir'])
    for i,df in enumerate(dfs):
        train,valid = train_test_split(df)
        train.to_csv(os.path.join(io_config['output_csv_dir'],'chunk{:d}_train.csv'.format(i)))
        valid.to_csv(os.path.join(io_config['output_csv_dir'],'chunk{:d}_valid.csv'.format(i)))
