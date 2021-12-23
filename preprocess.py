import argparse
import datetime
import glob
import numpy as np
import os
import pandas as pd
import pickle
import sys
import torch
import torch.utils.data as data
from tqdm import tqdm

from data.tools import train_test_split
from model.fm import FactorizationMachineModel
from utils import Timer,mkdir_from_filepath,read_attr_conf

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('mode.chained_assignment', None)

n_hotel_cluster = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device ',device)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('conf',action='store')
    return parser.parse_args()
    
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)

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

    feature_groups = io_config['fm_feature_groups']
    features_from_map = io_config['book_click_rate_category']

    features_to_explode = ['{:s}_book_rate'.format(f) for f in features_from_map] + ['{:s}_click_rate'.format(f) for f in features_from_map] + [k for k in feature_groups] + ['is_booking',]
    features_to_select = features_to_explode + ['srch_id'] 

    timer = Timer()

    df = df.fillna(value=-1).reset_index()

    tqdm.write(f'Processing timestamp features')
    timer.start()
    df['srch_id'] = df.apply(lambda x: '_'.join([str(x['date_time']),str(x['user_id']),str(x['srch_destination_id'])]),axis=1)
    timer.print_reset()
    
    tqdm.write(f'fm on categorical features')
    timer.start()
    for group,feature_columns in feature_groups.items():
        df[group] = construct_df_fm(df,group,feature_columns,os.path.join(io_config['fm_model_dir'],group+'/'))
    timer.print_reset()

    tqdm.write(f'construct book_rate, is_booking')
    timer.start()

    for feature in features_from_map: 
        rate_uniform_prior = [1./n_hotel_cluster for i in range(n_hotel_cluster)]

        if 'book_rate_dir' in io_config:
            book_rate_map = pickle.load(open(os.path.join(io_config['book_rate_dir'],'{:s}_book_rate.p'.format(feature)),'rb')) 
            df['{:s}_book_rate'.format(feature)] = df.apply(lambda x: book_rate_map.get(str(x[feature]),rate_uniform_prior),axis=1)

        if 'click_rate_dir' in io_config:
            click_rate_map = pickle.load(open(os.path.join(io_config['click_rate_dir'],'{:s}_click_rate.p'.format(feature)),'rb')) 
            df['{:s}_click_rate'.format(feature)] = df.apply(lambda x: click_rate_map.get(str(x[feature]),rate_uniform_prior),axis=1)
        
        if 'book_var_dir' in io_config:
            book_var_map = pickle.load(open(os.path.join(io_config['book_var_dir'],'{:s}_book_var.p'.format(feature)),'rb')) 
            df['{:s}_book_var'.format(feature)] = df.apply(lambda x: book_var_map.get(str(x[feature]),var_uniform_prior),axis=1)

        if 'click_var_dir' in io_config:
            click_var_map = pickle.load(open(os.path.join(io_config['click_var_dir'],'{:s}_click_var.p'.format(feature)),'rb')) 
            df['{:s}_click_var'.format(feature)] = df.apply(lambda x: click_var_map.get(str(x[feature]),var_uniform_prior),axis=1)
        
    df['is_booking'] = df.apply(lambda x: explode_is_booking(x['is_booking'],x['hotel_cluster']),axis=1)
    
    timer.print_reset()
    
    tqdm.write(f'Explode df')
    timer.start()
    df = df.explode(features_to_explode).reset_index()
    df['is_booking'] = df['is_booking'].astype(np.int64)
    timer.print_reset()
    
    df = df[features_to_select]
    
    return df

def preprocess_df(df):
    train,valid = train_test_split(df)
    train = preprocess(train)
    train.to_csv(os.path.join(io_config['output_csv_dir'],'train.csv'),index=False)
    valid = preprocess(valid)
    valid.to_csv(os.path.join(io_config['output_csv_dir'],'valid.csv'),index=False)
    return train,valid

def run():
    if not os.path.exists(io_config['output_csv_dir']):
        os.makedirs(io_config['output_csv_dir'])
    if 'nsample' in io_config:
        df = pd.read_csv(io_config['input_csv_path']).sample(io_config['nsample'])
        preprocess_df(df)
    elif 'chunksize' in io_config:
        dfs = pd.read_csv(io_config['input_csv_path'],chunksize=io_config['chunksize'])
        trains,valids = [],[]
        for i,df in enumerate(dfs):
            train,valid = preprocess_df(df)
            trains.append(train)
            valids.append(valid)
        train = pd.concat(trains,axis=0)
        train.to_csv(os.path.join(io_config['output_csv_dir'],'train.csv'))
        del train
        valid = pd.concat(valids,axis=0)
        valid.to_csv(os.path.join(io_config['output_csv_dir'],'valid.csv'))
        del valid
    else:
        df = pd.read_csv(io_config['input_csv_path'])
        preprocess_df(df)

if __name__ == "__main__":
    args = parse_arguments()
    global io_config
    io_config = read_attr_conf(args.conf,'preprocess_conf')
    run()
