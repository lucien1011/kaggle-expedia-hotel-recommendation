import argparse
import glob
import numpy as np
import os
import pandas as pd
import pprint
import random
import xgboost as xgb

from data.tools import x_y_group
from utils import mkdir_p,read_attr_conf
 
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('conf',action='store')
    parser.add_argument('--train_chunks',action='store_true')
    return parser.parse_args()

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def read_chunk(input_dir,ichunk):
    train = pd.read_csv(os.path.join(input_dir,'chunk{:d}_train.csv'.format(ichunk)),index_col=0)
    valid = pd.read_csv(os.path.join(input_dir,'chunk{:d}_valid.csv'.format(ichunk)),index_col=0)
    return train,valid

def train_model(x_train,y_train,group_train,x_valid,y_valid,group_valid,io_config):
    seed_everything()
    model = xgb.sklearn.XGBRanker(**io_config['xgb_config'])
    model.fit(
        x_train, y_train, group_train, verbose=True,
        eval_set=[(x_valid, y_valid)], eval_group=[group_valid],
        early_stopping_rounds=io_config['early_stopping_rounds'],
    )
    base_dir = os.path.join(io_config['base_dir'],io_config['rev_dir'])
    mkdir_p(base_dir)
    model.save_model(os.path.join(base_dir,io_config['saved_model_name']))
    return model

def train_chunks():
    for ichunk in range(io_config['nchunk']):
        print("reading chunk",ichunk) 
        train,valid = read_chunk(io_config['base_dir'],ichunk)
        x_train,y_train,group_train = x_y_group(train,['srch_id','is_booking'])
        x_valid,y_valid,group_valid = x_y_group(valid,['srch_id','is_booking'])
        print("fitting chunk",ichunk)
        model = train_model(x_train,y_train,group_train,x_valid,y_valid,group_valid,io_config)
    return model

def train():
    train_df = pd.read_csv(os.path.join(io_config['base_dir'],'train.csv'))
    x_train,y_train,group_train = x_y_group(train_df,['srch_id','is_booking'])
    valid_df = pd.read_csv(os.path.join(io_config['base_dir'],'valid.csv'))
    x_valid,y_valid,group_valid = x_y_group(valid_df,['srch_id','is_booking'])
    del train_df,valid_df
    model = train_model(x_train,y_train,group_train,x_valid,y_valid,group_valid,io_config)

if __name__ == "__main__":
    args = parse_arguments()
    global io_config
    io_config = read_attr_conf(args.conf,'train_conf')
    if args.train_chunks:
        train_chunks()
    else:
        train()
