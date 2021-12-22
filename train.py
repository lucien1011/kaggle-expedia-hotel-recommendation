import argparse
import glob
import numpy as np
import os
import pandas as pd
import pprint
import random
import xgboost as xgb

from data.tools import x_y_group

io_config = dict(
    base_dir='storage/output/211222_baseline_nsample1e5/',
    rev_dir='train_rev_01/',
)
io_config['save_model_path']=os.path.join(io_config['base_dir'],io_config['rev_dir'],'pairwise.model')

xgb_config = {'eval_metric': 'map@5', 'gamma': 4.646432672836216, 'learning_rate': 0.4973514335339059, 'max_depth': 7, 'min_child_weight': 10.0, 'n_estimators': 180, 'objective': 'rank:pairwise', 'reg_alpha': 177.0, 'reg_lambda': 0.24106060721446132, 'seed': 42}
    
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

def train_model(x_train,y_train,group_train,x_valid,y_valid,group_valid,xgb_config):
    seed_everything()
    model = xgb.sklearn.XGBRanker(**xgb_config)
    model.fit(
        x_train, y_train, group_train, verbose=True,
        eval_set=[(x_valid, y_valid)], eval_group=[group_valid],
        early_stopping_rounds=5,
    )
    base_dir = os.path.join(io_config['base_dir'],io_config['rev_dir'])
    if not os.path.exists(base_dir): os.makedirs(base_dir)
    model.save_model(io_config['save_model_path'])
    return model

def train_chunks():
    for ichunk in range(io_config['nchunk']):
        print("reading chunk",ichunk) 
        train,valid = read_chunk(io_config['base_dir'],ichunk)
        x_train,y_train,group_train = x_y_group(train,['srch_id','is_booking'])
        x_valid,y_valid,group_valid = x_y_group(valid,['srch_id','is_booking'])
        print("fitting chunk",ichunk)
        model = train_model(x_train,y_train,group_train,x_valid,y_valid,group_valid,xgb_config)
    return model

def train():
    train_df = pd.read_csv(os.path.join(io_config['base_dir'],'train.csv'))
    x_train,y_train,group_train = x_y_group(train_df,['srch_id','is_booking'])
    valid_df = pd.read_csv(os.path.join(io_config['base_dir'],'valid.csv'))
    x_valid,y_valid,group_valid = x_y_group(valid_df,['srch_id','is_booking'])
    del train_df,valid_df
    model = train_model(x_train,y_train,group_train,x_valid,y_valid,group_valid,xgb_config)

if __name__ == "__main__":
    args = parse_arguments()
    if args.train_chunks:
        train_chunks()
    else:
        train()
