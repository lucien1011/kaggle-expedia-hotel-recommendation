import datetime
import numpy as np
import os
import pandas as pd
import pickle
import torch
import torch.utils.data as data
from tqdm import tqdm

from data.tools import train_test_split
from model.fm import FactorizationMachineModel
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
    fm_model_dir='storage/output/211220_catvar_hotel_cluster/',
    book_rate_path='storage/output/211220_book_rate/book_rate.p',
    output_csv_dir='storage/output/211220_baseline+fm_nsample1e5/',
    nsample=int(1e5),
)

def explode_is_booking(is_booking,hotel_cluster,):
    out = np.zeros(n_hotel_cluster)
    if is_booking:
        out[hotel_cluster] = 1
    return out

def read_fm_model(model_dir):
    return {hotel_cluster: torch.load(os.path.join(model_dir,'hotel_cluster_{:d}/'.format(hotel_cluster),'saved.model')) for hotel_cluster in range(n_hotel_cluster)}

def preprocess(df, progressbar=False):
    
    def prepare_tensor(df,feature_columns):
        return torch.tensor(df[feature_columns].values).long()

    def construct_df_fm(df,group,feature_columns,fm_model_dir):
        fm_models = read_fm_model(fm_model_dir)
        x = prepare_tensor(df,feature_columns)
        dataloader = data.DataLoader(x,batch_size=int(2**18),shuffle=False)
        init = {}
        for hotel_cluster in range(n_hotel_cluster):
            fm_models[hotel_cluster].eval()
            s = []
            for x in tqdm(dataloader) if progressbar else dataloader:
                with torch.no_grad():
                    y_hat = fm_models[hotel_cluster](x.to(device))
                s.extend(y_hat.cpu().detach().numpy().tolist())
            init[hotel_cluster] = s
        df_group = pd.DataFrame({group:list(map(list,zip(*init.values())))})
        return df_group

    feature_groups = {
            'srch_destination': ['srch_destination_id','srch_destination_type_id',],
            #'user': ['user_id','user_location_country','user_location_region','user_location_city',],
            #'hotel': ['hotel_continent','hotel_country','hotel_market',],
            #'market': ['site_name','channel','is_mobile','is_package',],
            #'category': [
            #    'srch_destination_id','srch_destination_type_id',
            #    'user_id','user_location_country','user_location_region','user_location_city',
            #    'site_name','channel',
            #    'is_mobile','is_package',
            #],
    }
    features_to_explode = [k for k in feature_groups] + ['book_rate','is_booking',]
    features_to_select = features_to_explode + ['srch_id','year','month','srch_duration','srch_trip_time_diff',]

    timer = Timer()

    df = df.fillna(value=-1).reset_index()

    tqdm.write(f'Processing timestamp features')
    timer.start()
    df['srch_id'] = df.apply(lambda x: '_'.join([str(x['date_time']),str(x['user_id']),str(x['srch_destination_id'])]),axis=1)
    df["srch_ci"] = pd.to_datetime(df["srch_ci"])
    df["srch_co"] = pd.to_datetime(df["srch_co"])
    df['year'] = df['date_time'].apply(lambda x: x.year).astype('category').cat.codes
    df['month'] = df['date_time'].apply(lambda x: x.month).astype('category').cat.codes
    df['srch_duration'] = (df["srch_co"] - df["srch_ci"]).dt.days
    df['srch_trip_time_diff'] = (df["srch_co"] - df["date_time"]).dt.days
    timer.print_reset()
    
    tqdm.write(f'fm on categorical features')
    timer.start()
    for group,feature_columns in feature_groups.items():
        df[group] = construct_df_fm(df,group,feature_columns,os.path.join(io_config['fm_model_dir'],group+'/'))
    timer.print_reset()

    tqdm.write(f'construct book_rate, is_booking')
    timer.start()
    booking_rate_map = pickle.load(open(os.path.join(io_config['book_rate_path']),'rb')) 
    booking_rate = [booking_rate_map[hotel_cluster] for hotel_cluster in range(n_hotel_cluster)]
    df['book_rate'] = df.apply(lambda x: booking_rate,axis=1)
    df['is_booking'] = df.apply(lambda x: explode_is_booking(x['is_booking'],x['hotel_cluster']),axis=1)
    timer.print_reset()
    
    tqdm.write(f'Explode df')
    timer.start()
    df = df.explode(features_to_explode).reset_index()
    df['book_rate'] = df['book_rate'].astype(np.float64)
    df['is_booking'] = df['is_booking'].astype(np.int64)
    timer.print_reset()
    
    df = df[features_to_select]
    
    return df

if __name__ == "__main__":
    if 'nsample' in io_config:
        df = pd.read_csv(io_config['input_csv_path']).sample(io_config['nsample'])
        if not os.path.exists(io_config['output_csv_dir']):
            os.makedirs(io_config['output_csv_dir'])
        train,valid = train_test_split(df)
        train = preprocess(train)
        train.to_csv(os.path.join(io_config['output_csv_dir'],'train.csv'),index=False)
        del train
        valid = preprocess(valid)
        valid.to_csv(os.path.join(io_config['output_csv_dir'],'valid.csv'),index=False)
        del valid
    elif 'chunksize' in io_config:
        dfs = pd.read_csv(io_config['input_csv_path'],chunksize=io_config['chunksize'])
        if not os.path.exists(io_config['output_csv_dir']):
            os.makedirs(io_config['output_csv_dir'])
        for i,df in enumerate(dfs):
            train,valid = train_test_split(df)
            train.to_csv(os.path.join(io_config['output_csv_dir'],'chunk{:d}_train.csv'.format(i)))
            valid.to_csv(os.path.join(io_config['output_csv_dir'],'chunk{:d}_valid.csv'.format(i)))
