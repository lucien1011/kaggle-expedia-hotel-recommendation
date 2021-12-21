import os
import pandas as pd
import pickle

from data.tools import train_test_split

def categorize(df):
    columns = [
            'srch_destination_id','srch_destination_type_id',
            'user_id','user_location_country','user_location_region','user_location_city',
            'hotel_continent','hotel_country','hotel_market',
            'site_name','channel',
            'is_mobile','is_package',
            ]
    for column in columns:
        df[column] = df[column].astype('category').cat.codes
        print(f'{column}: {len(df[column].unique())} {df[column].unique().max()}')
    field_dims = {column: len(df[column].unique()) for column in columns}
    return df,field_dims

def filter_by_booking(df):
    return df[df.is_booking == 1],df[df.is_booking == 0]

def save(
        train,ftrain,
        book,fbook,
        click,fclick,
        fielddims,ffielddims,
        ):
    train.to_csv(ftrain)
    book.to_csv(fbook)
    click.to_csv(fclick)
    pickle.dump(fielddims,open(ffielddims,'wb'))

def prepare(input_path,ftrain,fbook,fclick,ffielddims):
    df = pd.read_csv(input_path)
    df,fielddims = categorize(df)
    book,click = filter_by_booking(df)
    save(train,ftrain,book,fbook,click,fclick,fielddims,ffielddims)

def run():
    in_dir = 'storage/'
    out_dir = 'storage/output/211212_train_test_categorized/'
    ftrain = 'train.csv'
    fbook = 'book.csv'
    fclick = 'click.csv'
    ffielddims = 'fielddims.p'

    prepare(
            os.path.join(in_dir,'train.csv'),
            os.path.join(out_dir,ftrain),
            os.path.join(out_dir,fbook),
            os.path.join(out_dir,fclick),
            os.path.join(out_dir,ffielddims),
            )

if __name__ == '__main__':
    run()
