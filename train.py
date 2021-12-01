import glob
import numpy as np
import os
import pandas as pd
import xgboost as xgb

io_config = dict(
    input_csv_dir='/cmsuf/data/store/user/t2/users/klo/MiscStorage/ForLucien/Kaggle/expedia-hotel-recommendations/preprocess/211201_baseline/',
    nchunk=4,
)

xgb_config = {
    'colsample_bytree': 0.7959932314624918, 
    'gamma': 3.236981174565596, 
    'learning_rate': 0.8092969260411637, 
    'min_child_weight': 10.0, 
    'reg_alpha': 83.0, 
    'reg_lambda': 0.9226958452956067, 
    'max_depth': 6,
    'n_estimators': 180,
    'seed': 0,
    'eval_metric':'map@5',
    'objective': 'rank:map',
}
    
fit_config = {
    'early_stopping_rounds': 5,
    'xgb_model': None,
}

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

def train_model(x_train,y_train,group_train,x_valid,y_valid,group_valid,xgb_config,fit_config):
    model = xgb.sklearn.XGBRanker(**xgb_config)
    es = xgb.callback.EarlyStopping(
        rounds=fit_config.get('early_stopping_rounds',5),
        save_best=True,
    )
    model.fit(
        x_train, y_train, group_train, verbose=True,
        eval_set=[(x_valid, y_valid)], eval_group=[group_valid],
        early_stopping_rounds=fit_config.get('early_stopping_rounds',5),
        xgb_model=fit_config.get('xgb_model',None),
        callbacks=[es],
    )
    return model

if __name__ == "__main__":
    for ichunk in range(io_config['nchunk']):
        print("reading chunk",ichunk) 
        train,valid = read_chunk(io_config['input_csv_dir'],ichunk)
        x_train,y_train,group_train = x_y_group(train)
        x_valid,y_valid,group_valid = x_y_group(valid)
        print("fitting chunk",ichunk)
        model = train_model(x_train,y_train,group_train,x_valid,y_valid,group_valid,xgb_config,fit_config)
        fit_config['xgb_model'] = model.get_booster()
