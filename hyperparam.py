import argparse
import copy
import glob
import numpy as np
import os
import pandas as pd
import pickle
import pprint
import random
import xgboost as xgb

from data.tools import x_y_group
from utils import mkdir_p,read_attr_conf

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('conf',action='store')
    return parser.parse_args()

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def hyperopt():
    from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
        
    train_df = pd.read_csv(os.path.join(io_config['base_dir'],'train.csv'))
    x_train,y_train,group_train = x_y_group(train_df,['srch_id','is_booking'])
    valid_df = pd.read_csv(os.path.join(io_config['base_dir'],'valid.csv'))
    x_valid,y_valid,group_valid = x_y_group(valid_df,['srch_id','is_booking'])
    del train_df,valid_df
 
    def objective(space):
    
        seed_everything(io_config['space']['seed'])
        model = xgb.sklearn.XGBRanker(**io_config['space'])
        model.fit(
            x_train, y_train, group_train, verbose=True,
            eval_set=[(x_valid, y_valid)], eval_group=[group_valid],
            early_stopping_rounds=io_config['early_stopping_rounds'],
        )    
        print(space)
        print("SCORE:", model.best_score)
        return {'loss': 1-model.best_score, 'status': STATUS_OK }

    def save_best_hyperparams(best_hyperparams):
        base_dir = os.path.join(io_config['base_dir'],io_config['rev_dir'])
        mkdir_p(base_dir)
        space = copy.deepcopy(io_config['space'])
        for k,v in best_hyperparams.items():
            space[k] = v
        pickle.dump(space,open(os.path.join(base_dir,'best_hyperparams.p'),'wb'))

    trials = Trials()
    best_hyperparams = fmin(
        fn = objective,
        space = io_config['space'],
        algo = tpe.suggest,
        max_evals = io_config['max_evals'],
        trials = trials
    )
    print('Best hyperparameters: ',best_hyperparams)
    save_best_hyperparams(best_hyperparams)

if __name__ == "__main__":
    args = parse_arguments()
    global io_config
    io_config = read_attr_conf(args.conf,'hyperopt_conf')
    hyperopt()
