import argparse
import glob
import numpy as np
import os
import pandas as pd
import pprint
import random
import xgboost as xgb

io_config = dict(
    base_dir='storage/output/211217_baseline_nsample1e5/',
    nchunk=4,
)
io_config['input_csv_dir'] = io_config['base_dir']
io_config['save_model_path']=os.path.join(io_config['base_dir'],'pairwise.model')

xgb_config = {'eval_metric': 'map@5', 'gamma': 5.137244663650965, 'learning_rate': 0.3586551065993189, 'max_depth': 9, 'min_child_weight': 7.0, 'n_estimators': 180, 'objective': 'rank:pairwise', 'reg_alpha': 130.0, 'reg_lambda': 0.8589062653908213, 'seed': 42} 
    
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_chunks',action='store_true')
    return parser.parse_args()

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def read_chunk(input_dir,ichunk):
    train = pd.read_csv(os.path.join(input_dir,'chunk{:d}_train.csv'.format(ichunk)),index_col=0)
    valid = pd.read_csv(os.path.join(input_dir,'chunk{:d}_valid.csv'.format(ichunk)),index_col=0)
    return train,valid

def x_y_group(data):
    columns_to_drop = ['srch_id','is_booking','user_id','date_time',]
    print('make x y group')
    x = data.loc[:, ~data.columns.isin(columns_to_drop)]
    y = data.loc[:, data.columns.isin(['is_booking'])]
    group = data.groupby('srch_id').size().to_frame('size')['size'].to_numpy()
    print('shape (x,y,group): ',x.shape,y.shape,group.shape)
    return x,y,group

def train_model(x_train,y_train,group_train,x_valid,y_valid,group_valid,xgb_config):
    seed_everything()
    model = xgb.sklearn.XGBRanker(**xgb_config)
    model.fit(
        x_train, y_train, group_train, verbose=True,
        eval_set=[(x_valid, y_valid)], eval_group=[group_valid],
        early_stopping_rounds=2,
    )
    model.save_model(io_config['save_model_path'])
    return model

def train_chunks():
    for ichunk in range(io_config['nchunk']):
        print("reading chunk",ichunk) 
        train,valid = read_chunk(io_config['input_csv_dir'],ichunk)
        x_train,y_train,group_train = x_y_group(train)
        x_valid,y_valid,group_valid = x_y_group(valid)
        print("fitting chunk",ichunk)
        model = train_model(x_train,y_train,group_train,x_valid,y_valid,group_valid,xgb_config)
    return model

def train():
    train_df = pd.read_csv(os.path.join(io_config['input_csv_dir'],'train.csv'),index_col=0)
    x_train,y_train,group_train = x_y_group(train_df)
    valid_df = pd.read_csv(os.path.join(io_config['input_csv_dir'],'valid.csv'),index_col=0)
    x_valid,y_valid,group_valid = x_y_group(valid_df)
    del train_df,valid_df
    model = train_model(x_train,y_train,group_train,x_valid,y_valid,group_valid,xgb_config)

if __name__ == "__main__":
    args = parse_arguments()
    if args.train_chunks:
        train_chunks()
    else:
        train()
