import datetime
import pandas as pd
import math
import numpy as np
import os
import time
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model.fm import FactorizationMachineModel 

torch.manual_seed(1)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('mode.chained_assignment', None)

n_hotel_cluster = 100
embedding_dim = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device ',device)

io_config = dict(
    input_csv_path='/cmsuf/data/store/user/t2/users/klo/MiscStorage/ForLucien/Kaggle/expedia-hotel-recommendations/train_lite.csv',
    save_model_path='/cmsuf/data/store/user/t2/users/klo/MiscStorage/ForLucien/Kaggle/expedia-hotel-recommendations/output/211216_catvar_booking.model',
)

train_config = dict(
    lr=0.5,
    nepoch=100,
    embedding_dim=4,
)

feature_columns = [
        'srch_destination_id','srch_destination_type_id',
        'user_id','user_location_country','user_location_region','user_location_city',
        'site_name','channel',
        'is_mobile','is_package',
        'hotel_cluster',
        ]

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess',action='store_true')
    parser.add_argument('--train',action='store_true')
    return parser.parse_args()

def read_csv(path):
    df = pd.read_csv(path)
    return df

def train_test_split(df,timestamp=(2014,8,1)):
    print('train test split with datatime')
    train_test_timestamp = pd.Timestamp(datetime.datetime(*timestamp))
    df['date_time'] = pd.to_datetime(df['date_time'])
    X_train_inds = df.date_time < train_test_timestamp
    X_test_inds = df.date_time > train_test_timestamp
    print('baseline_preprocess train data')
    train_data = df[X_train_inds]
    print('baseline_preprocess validation data')
    valid_data = df[X_test_inds]
    return train_data,valid_data

def preprocess(df):
    for feature_column in feature_columns:
        df[feature_column] = df[feature_column].astype('category').cat.codes
    features_to_keep = feature_columns + ['is_booking',]
    df = df[features_to_keep+['date_time']]
    train,valid = train_test_split(df)
    train = train[features_to_keep]
    valid = valid[features_to_keep]
    return train,valid,[len(df[feature_column].unique()) for feature_column in feature_columns]

def prepare_tensor(df):
    x = torch.tensor(df[feature_columns].values).long()
    y = torch.tensor(df['is_booking'].values).float()
    dataset = data.TensorDataset(x,y)
    return x,y,dataset

def fit(iterator, model, optimizer, criterion):
    train_loss = 0
    model.train()
    for x,y in iterator:
        optimizer.zero_grad()
        y_hat = model(x.to(device))
        loss = criterion(y_hat, y.to(device))
        train_loss += loss.item()*x.shape[0]
        loss.backward()
        optimizer.step()
    return train_loss / len(iterator.dataset)

def test(iterator, model, criterion):
    train_loss = 0
    model.eval()
    for x,y in iterator:                    
        with torch.no_grad():
            y_hat = model(x.to(device))
        loss = criterion(y_hat, y.to(device))
        train_loss += loss.item()*x.shape[0]
    return train_loss / len(iterator.dataset)

def train(train_df,valid_df,field_dims):

    train_x,train_y,trainset = prepare_tensor(train_df)
    valid_x,valid_y,validset = prepare_tensor(valid_df)
    
    bs=1024
    train_dataloader = data.DataLoader(trainset,batch_size=bs,shuffle=True)
    valid_dataloader = data.DataLoader(validset,batch_size=bs,shuffle=True)

    model = FactorizationMachineModel(field_dims,embedding_dim).to(device)
    wd=1e-5
    lr=0.001
    epochs=50
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[7], gamma=0.1)
    criterion = nn.BCELoss().to(device)
    for epoch in range(epochs):
        start_time = time.time()
        train_loss = fit(train_dataloader, model, optimizer, criterion)
        valid_loss = test(valid_dataloader, model, criterion)
        scheduler.step()
        secs = int(time.time() - start_time)
        print(f'epoch {epoch}. time: {secs}[s]')
        print(f'\ttrain bce: {(math.sqrt(train_loss)):.4f}')
        print(f'\tvalidation bce: {(math.sqrt(valid_loss)):.4f}')

    save_dir = os.path.dirname(io_config['save_model_path'])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model,io_config['save_model_path'])

def run():
    df = read_csv(io_config['input_csv_path'])
    train_df,valid_df,field_dims = preprocess(df)
    train(train_df,valid_df,field_dims)

if __name__ == '__main__':
    run()
