import math
import os
import pandas as pd
import pickle
from tqdm import tqdm

from data.tools import train_test_split

n_hotel_cluster = 100

io_config = dict(
    input_csv_path='storage/output/211222_train_test_categorized/train.csv',
    output_dir='storage/output/211223_book+click+rate+variance_by+category/',
)

def book_rate(train):
    return dict(df.hotel_cluster.value_counts() / len(df))

def book_rate_by_group(train,match_cols=["srch_destination_id"]):
    def make_key(items):
        return "_".join([str(i) for i in items])
    
    cluster_cols = match_cols + ['hotel_cluster']
    groups = train.groupby(cluster_cols)
    book_rate_map,click_rate_map = {},{}
    book_var_map,click_var_map = {},{}
    for name, group in tqdm(groups):
        clicks = len(group.is_booking[group.is_booking == False])
        bookings = len(group.is_booking[group.is_booking == True])
     
        clus_name = make_key(name[:len(match_cols)])
        if clus_name not in book_rate_map:
            book_rate_map[clus_name] = {}
        if clus_name not in click_rate_map:
            click_rate_map[clus_name] = {}

        book_rate_map[clus_name][name[-1]] = bookings
        click_rate_map[clus_name][name[-1]] = clicks

    for clus_name,clus_map in book_rate_map.items():
        clus_sum = sum(clus_map.values())
        book_rate_map[clus_name] = [clus_map.get(i,0.)/clus_sum if clus_sum != 0 else 1./n_hotel_cluster for i in range(n_hotel_cluster)]
        book_var_map[clus_name] = [math.sqrt(book_rate_map[clus_name][i]*(1.-book_rate_map[clus_name][i])/clus_sum) if clus_sum != 0 else 1. for i in range(n_hotel_cluster)]

    for clus_name,clus_map in click_rate_map.items():
        clus_sum = sum(clus_map.values())
        click_rate_map[clus_name] = [clus_map.get(i,0.)/clus_sum if clus_sum != 0 else 1./n_hotel_cluster for i in range(n_hotel_cluster)]
        click_var_map[clus_name] = [math.sqrt(click_rate_map[clus_name][i]*(1.-click_rate_map[clus_name][i])/clus_sum) if clus_sum != 0 else 1. for i in range(n_hotel_cluster)]

    return book_rate_map,click_rate_map,book_var_map,click_var_map

def pickle_save(obj,path):
    out_dir = os.path.dirname(path)
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    pickle.dump(obj,open(path,'wb'))

def run():
    df = pd.read_csv(io_config['input_csv_path'])
    df,_ = train_test_split(df)
    features = [
            'srch_destination_id',
            #'srch_destination_id','srch_destination_type_id',
            #'user_location_country','user_location_region','user_location_city',
            #'hotel_continent','hotel_country','hotel_market',
            #'site_name','channel','is_mobile','is_package',
            #'srch_adults_cnt','srch_children_cnt','srch_rm_cnt',
            #'year','month',
            #'srch_duration',
            ]
    for feature in features:
        book_rate,click_rate,book_var,click_var = book_rate_by_group(df,match_cols=[feature])
        pickle_save(book_rate,os.path.join(io_config['output_dir'],'{:s}_book_rate.p'.format(feature)))
        pickle_save(click_rate,os.path.join(io_config['output_dir'],'{:s}_click_rate.p'.format(feature)))
        pickle_save(book_var,os.path.join(io_config['output_dir'],'{:s}_book_var.p'.format(feature)))
        pickle_save(click_var,os.path.join(io_config['output_dir'],'{:s}_click_var.p'.format(feature)))

if __name__ == "__main__":
    run()
