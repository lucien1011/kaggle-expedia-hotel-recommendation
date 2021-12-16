import argparse
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
from torchmetrics import AUROC
from tqdm import tqdm

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

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target',action='store',default='hotel_cluster')
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

def preprocess(df,feature_columns,target):
    for feature_column in feature_columns:
        df[feature_column] = df[feature_column].astype('category').cat.codes
    features_to_keep = feature_columns + [target,]
    df = df[features_to_keep+['date_time']]
    train,valid = train_test_split(df)
    train = train[features_to_keep]
    valid = valid[features_to_keep]
    return train,valid,[len(df[feature_column].unique()) for feature_column in feature_columns]

def fit(iterator, model, optimizer, criterion):
    train_loss = 0
    model.train()
    for x,y in tqdm(iterator):
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
    for x,y in tqdm(iterator):                    
        with torch.no_grad():
            y_hat = model(x.to(device))
        loss = criterion(y_hat, y.to(device))
        train_loss += loss.item()*x.shape[0]
    return train_loss / len(iterator.dataset)

def train(train_x,train_y,trainset,valid_x,valid_y,validset,field_dims,train_config):
    
    bs = train_config['bs']
    wd = train_config['wd']
    lr = train_config['lr']
    epochs = train_config['epochs']

    train_dataloader = data.DataLoader(trainset,batch_size=bs,shuffle=True)
    valid_dataloader = data.DataLoader(validset,batch_size=bs,shuffle=True)

    model = FactorizationMachineModel(field_dims,train_config['embedding_dim']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[7], gamma=0.1)

    trn_losses,val_losses,trn_metrics,val_metrics = [],[],[],[]

    for epoch in range(epochs):
        start_time = time.time()
        train_loss = fit(train_dataloader, model, optimizer, train_config['loss'])
        valid_loss = test(valid_dataloader, model, train_config['loss'])
        train_metric = test(train_dataloader, model, train_config['metric'])
        valid_metric = test(valid_dataloader, model, train_config['metric'])
        trn_losses.append(train_loss)
        val_losses.append(valid_loss)
        trn_metrics.append(train_metric)
        val_metrics.append(valid_metric)
        scheduler.step()
        secs = int(time.time() - start_time)
        tqdm.write(f'epoch {epoch}. time: {secs}[s]')
        tqdm.write(f'\ttrain loss: {(math.sqrt(train_loss)):.4f}')
        tqdm.write(f'\tvalidation loss: {(math.sqrt(valid_loss)):.4f}')

    return model,trn_losses,val_losses,trn_metrics,val_metrics

def save(model,trn_losses,val_losses,trn_metrics,val_metrics,train_config):
    save_dir = os.path.dirname(train_config['save_model_path'])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model,train_config['save_model_path'])
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig,ax = plt.subplots()
    n_trn_epoch,n_val_epoch = len(trn_losses),len(val_losses)
    ax.plot(range(n_trn_epoch),trn_losses,label='train')
    ax.plot(range(n_val_epoch),val_losses,label='valid')
    fig.savefig(os.path.join(save_dir,train_config.get('loss_plot_fname','loss.png')))

    plt.clf()
    fig,ax = plt.subplots()
    n_trn_epoch,n_val_epoch = len(trn_metrics),len(val_metrics)
    ax.plot(range(n_trn_epoch),trn_metrics,label='train')
    ax.plot(range(n_val_epoch),val_metrics,label='valid')
    fig.savefig(os.path.join(save_dir,train_config.get('metric_plot_fname','metrics.png')))

def run_is_booking():

    def prepare_tensor(df):
        x = torch.tensor(df[feature_columns].values).long()
        y = torch.tensor(df['is_booking'].values).float()
        dataset = data.TensorDataset(x,y)
        return x,y,dataset

    io_config = dict(
        input_csv_path='/cmsuf/data/store/user/t2/users/klo/MiscStorage/ForLucien/Kaggle/expedia-hotel-recommendations/train.csv',
    )
    
    train_config = dict(
        lr=0.5,
        wd=1e-5,
        bs=1024,
        epochs=100,
        embedding_dim=4,
        train_criterion=nn.BCELoss(),
        valid_criterion=F1(),
        save_model_path='/cmsuf/data/store/user/t2/users/klo/MiscStorage/ForLucien/Kaggle/expedia-hotel-recommendations/output/211216_catvar_booking.model',
    )
    
    feature_columns = [
            'srch_destination_id','srch_destination_type_id',
            'user_id','user_location_country','user_location_region','user_location_city',
            'site_name','channel',
            'is_mobile','is_package',
            'hotel_cluster',
            ]

    df = read_csv(io_config['input_csv_path'])
    train_df,valid_df,field_dims = preprocess(df,feature_columns,'is_booking')
    train_x,train_y,trainset = prepare_tensor(train_df)
    valid_x,valid_y,validset = prepare_tensor(valid_df)
    train(train_x,train_y,trainset,valid_x,valid_y,validset,field_dims,train_config)

def run_hotel_cluster():

    bce_loss_fn = nn.BCELoss()
    def bce(y_hat,y):
        return bce_loss_fn(y_hat,y.float())

    def prepare_tensor(df,hotel_cluster):
        x = torch.tensor(df[feature_columns].values).long()
        y = torch.tensor((df['hotel_cluster']==hotel_cluster).values).long()
        dataset = data.TensorDataset(x,y)
        return x,y,dataset

    io_config = dict(
        input_csv_path='/cmsuf/data/store/user/t2/users/klo/MiscStorage/ForLucien/Kaggle/expedia-hotel-recommendations/train_lite.csv',
    )

    feature_columns = [
            'srch_destination_id','srch_destination_type_id',
            'user_id','user_location_country','user_location_region','user_location_city',
            'site_name','channel',
            'is_mobile','is_package',
            ]

    train_config = dict(
        lr=0.1,
        wd=1e-5,
        bs=1024,
        epochs=5,
        embedding_dim=4,
        loss=bce,
        metric=AUROC(),
        save_model_dir='/cmsuf/data/store/user/t2/users/klo/MiscStorage/ForLucien/Kaggle/expedia-hotel-recommendations/output/211216_catvar_hotel_cluster/',
    )

    df = read_csv(io_config['input_csv_path'])
    train_df,valid_df,field_dims = preprocess(df,feature_columns,'hotel_cluster')
    for hotel_cluster in range(n_hotel_cluster):
        tqdm.write('-'*100)
        tqdm.write('Processing hotel cluster {:d}'.format(hotel_cluster))
        train_x,train_y,trainset = prepare_tensor(train_df,hotel_cluster)
        valid_x,valid_y,validset = prepare_tensor(valid_df,hotel_cluster)
        train_config['save_model_path'] = os.path.join(train_config['save_model_dir'],'hotel_cluster_{:d}/'.format(hotel_cluster),'saved.model')
        model,trn_losses,val_losses,trn_metrics,val_metrics = train(train_x,train_y,trainset,valid_x,valid_y,validset,field_dims,train_config)
        save(model,trn_losses,val_losses,trn_metrics,val_metrics,train_config)

if __name__ == '__main__':
    args = parse_arguments()
    if args.target == 'is_booking':
        run_is_booking()
    elif args.target == 'hotel_cluster':
        run_hotel_cluster()
    else:
        raise RuntimeError('target {:s} not supported'.format(args.target))
