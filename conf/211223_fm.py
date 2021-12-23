fm_conf = dict(
    input_csv_path='storage/output/211222_train_test_categorized/train.csv',
    
    fm_model_dir=None,
    fm_feature_groups={
        'category': [
            'user_id','user_location_country','user_location_region','user_location_city',
            'hotel_continent','hotel_country',
            'site_name','channel','is_mobile','is_package',
            'year','month',
        ],
    },
    fielddims_path='storage/output/211222_train_test_categorized/fielddims.p',

    lr=0.1,
    wd=1e-5,
    bs=1024,
    epochs=1,
    embedding_dim=4,
    save_model_dir='storage/output/211223_fm/',
)
