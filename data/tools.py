import datetime
import pandas as pd

def train_test_split(df,timestamp=(2014,8,1)):
    print('train test split with datatime')
    train_test_timestamp = pd.Timestamp(datetime.datetime(*timestamp))
    df['date_time'] = pd.to_datetime(df['date_time'])
    X_train_inds = df.date_time < train_test_timestamp
    X_test_inds = df.date_time > train_test_timestamp
    return df[X_train_inds],df[X_test_inds]

def x_y_group(
        data,
        x_column_to_drop=['srch_id','is_booking','user_id','date_time',],
        target_column='is_booking',
        group_column='srch_id',
        verbose=True,
        ): 
    if verbose: print('make x y group')
    x = data.loc[:, ~data.columns.isin(x_column_to_drop)]
    y = data.loc[:, data.columns.isin([target_column])]
    group = data.groupby(group_column).size().to_frame('size')['size'].to_numpy()
    if verbose: print('shape (x,y,group): ',x.shape,y.shape,group.shape)
    return x,y,group
