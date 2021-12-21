import os
import pandas as pd
import pickle

from data.tools import train_test_split

n_hotel_cluster = 100

io_config = dict(
    input_csv_path='storage/output/211212_train_test_categorized/train.csv',
    output_dir='storage/output/211221_book_rate_by+srch+dest+id/',
)

def book_rate(train):
    return dict(df.hotel_cluster.value_counts() / len(df))

def book_rate_by_group(train,match_cols=["srch_destination_id"]):
    def make_key(items):
        return "_".join([str(i) for i in items])
    
    cluster_cols = match_cols + ['hotel_cluster']
    groups = train.groupby(cluster_cols)
    book_rate_map = {}
    for name, group in groups:
        clicks = len(group.is_booking[group.is_booking == False])
        bookings = len(group.is_booking[group.is_booking == True])
     
        clus_name = make_key(name[:len(match_cols)])
        if clus_name not in book_rate_map:
            book_rate_map[clus_name] = {}

        book_rate_map[clus_name][name[-1]] = bookings + 0.15 * clicks

    for clus_name,clus_map in book_rate_map.items():
        clus_sum = sum(clus_map.values())
        book_rate_map[clus_name] = [clus_map.get(i,0.)/clus_sum if clus_sum != 0 else 1./n_hotel_cluster for i in range(n_hotel_cluster)]
    return book_rate_map 

def run():
    df = pd.read_csv(io_config['input_csv_path'])
    df,_ = train_test_split(df)
    book_rate = book_rate_by_group(df)
    if not os.path.exists(io_config['output_dir']): os.makedirs(io_config['output_dir'])
    pickle.dump(book_rate,open(os.path.join(io_config['output_dir'],'book_rate.p'),'wb'))

if __name__ == "__main__":
    run()
