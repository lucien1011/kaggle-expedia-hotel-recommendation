preprocess_conf = dict(
    input_csv_path='storage/output/211222_train_test_categorized/book.csv',
    
    fm_model_dir=None,
    fm_feature_groups={},
    fielddims_path=None,

    click_rate_dir='storage/output/211221_book_click_rate_by+category/',
    book_rate_dir='storage/output/211221_book_click_rate_by+category/',
    book_click_rate_category = ['srch_destination_id'],


    output_csv_dir='storage/output/211223_baseline_nsample1e6/',
    nsample=int(1e6),
    seed=42,
)

fm_conf = dict(
    input_csv_path='storage/output/211222_train_test_categorized/train.csv',
    
    fm_model_dir=None,
    fm_feature_groups={},
    fielddims_path=None,

    lr=0.1,
    wd=1e-5,
    bs=1024,
    epochs=1,
    embedding_dim=4,
    save_model_dir=None,
)

from hyperopt import hp
import numpy as np
hyperopt_conf = dict(
    base_dir='storage/output/211222_baseline_nsample1e6_test/',
    rev_dir='hyperopt_rev_01/',
    early_stopping_rounds=5,
    max_evals=20,
    seed=42,
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
)

train_conf = dict(
    base_dir='storage/output/211222_baseline_nsample1e6_test/',
    rev_dir='train_rev_01/',
    early_stopping_rounds=5,
    saved_model_name='pairwise.model',
    xgb_config={'eval_metric': 'map@5', 'gamma': 8.759836634327673, 'learning_rate': 0.751319389003732, 'max_depth': 6, 'min_child_weight': 3.0, 'n_estimators': 180, 'objective': 'rank:pairwise', 'reg_alpha': 47.0, 'reg_lambda': 0.060633636599996765, 'seed': 42},
)
