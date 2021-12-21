import datetime
import numpy as np
import os
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('mode.chained_assignment', None)

n_hotel_cluster = 100

io_config = dict(
    input_csv_path='storage/train_is_booking.csv',
)

def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

def train_test_split(df,timestamp=(2014,8,1)):
    print('train test split with datatime')
    train_test_timestamp = pd.Timestamp(datetime.datetime(*timestamp))
    df['date_time'] = pd.to_datetime(df['date_time'])
    X_train_inds = df.date_time < train_test_timestamp
    X_test_inds = df.date_time > train_test_timestamp
    return df[X_train_inds],df[X_test_inds]

def predict(train,valid):
    print('top clusters')
    def make_key(items):
        return "_".join([str(i) for i in items])
    
    match_cols = ["srch_destination_id"]
    cluster_cols = match_cols + ['hotel_cluster']
    groups = train.groupby(cluster_cols)
    top_clusters = {}
    for name, group in groups:
        clicks = len(group.is_booking[group.is_booking == False])
        bookings = len(group.is_booking[group.is_booking == True])
    
        score = bookings + .15 * clicks
    
        clus_name = make_key(name[:len(match_cols)])
        if clus_name not in top_clusters:
            top_clusters[clus_name] = {}
        top_clusters[clus_name][name[-1]] = score

    import operator
    
    cluster_dict = {}
    for n in top_clusters:
        tc = top_clusters[n]
        top = [l[0] for l in sorted(tc.items(), key=operator.itemgetter(1), reverse=True)[:5]]
        cluster_dict[n] = top 

    print('predict')
    preds = []
    for index, row in valid.iterrows():
        key = make_key([row[m] for m in match_cols])
        if key in cluster_dict:
            preds.append(cluster_dict[key])
        else:
            preds.append([])
    return preds

def run():
    df = pd.read_csv(io_config['input_csv_path'])
    train,valid = train_test_split(df)
    preds = predict(train,valid)
    trues = [[l] for l in valid["hotel_cluster"]]
    print("map@5: ",mapk(trues, preds, k=5))

if __name__ == "__main__":
    run()
