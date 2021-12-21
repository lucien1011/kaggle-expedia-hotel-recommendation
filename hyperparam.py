import argparse
import glob
import numpy as np
import os
import pandas as pd
import pickle
import pprint
import random
import xgboost as xgb

from data.tools import x_y_group

io_config = dict(
    base_dir='storage/output/211221_baseline+fm_nsample1e5/',
    rev_dir='hyperopt_rev_01/',
)
io_config['save_model_path']=os.path.join(io_config['base_dir'],'pairwise.model')

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def hyperopt():
    from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
    
    space = {
        'objective': 'rank:pairwise',
        'max_depth': hp.choice("max_depth",np.arange(2, 10, dtype=int) ),
        'gamma': hp.uniform('gamma', 1,9),
        'learning_rate': hp.uniform('learning_rate',0.1,1.0),
        'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': 180,
        'seed': 42,
        'eval_metric':'map@5',
    }
    
    train_df = pd.read_csv(os.path.join(io_config['base_dir'],'train.csv'))
    x_train,y_train,group_train = x_y_group(train_df,['srch_id','is_booking'])
    valid_df = pd.read_csv(os.path.join(io_config['base_dir'],'valid.csv'))
    x_valid,y_valid,group_valid = x_y_group(valid_df,['srch_id','is_booking'])
    del train_df,valid_df
 
    def objective(space):
    
        seed_everything(space['seed'])
        model = xgb.sklearn.XGBRanker(**space)
        model.fit(
            x_train, y_train, group_train, verbose=True,
            eval_set=[(x_valid, y_valid)], eval_group=[group_valid],
            early_stopping_rounds=5,
        )    
        print(space)
        print("SCORE:", model.best_score)
        return {'loss': 1-model.best_score, 'status': STATUS_OK }

    trials = Trials()
    best_hyperparams = fmin(
        fn = objective,
        space = space,
        algo = tpe.suggest,
        max_evals = 20,
        trials = trials
    )
    print('Best hyperparameters: ',best_hyperparams)

    base_dir = os.path.join(io_config['base_dir'],io_config['rev_dir'])
    if not os.path.exists(base_dir): os.makedirs(base_dir)
    for k,v in best_hyperparams.items():
        space[k] = v
    pickle.dump(space,open(os.path.join(base_dir,'best_hyperparams.p'),'wb'))

if __name__ == "__main__":
    hyperopt()
