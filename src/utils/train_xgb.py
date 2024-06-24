import json
import numpy as np
import xgboost as xgb

from tqdm import tqdm

from src.utils.amex_metric import amex_metric
from src.utils.amex_metric import amex_scorer

def train_xgb(X_train, y_train, X_v, y_v, n=10, params=None):
    '''
    Because of subsampling and early stopping, training XGBoost on the same data multiple times
    will lead to slightly different results. To mitigate this, we train n times with different 
    seeds and average the results.
    '''
    if params is None:
        with open('../config/xgboost.json', 'r') as f:
            params = json.load(f)
    dtrain = xgb.DMatrix(X_train, y_train)
    dv = xgb.DMatrix(X_v, y_v)
    amex_metrics = []
    for _ in range(n):
        bst = xgb.train(params, dtrain, seed=_, num_boost_round=9999, verbose_eval=0,
                        evals=[(dtrain, 'train'), (dv, 'v')], custom_metric=amex_scorer, 
                        early_stopping_rounds=100, maximize=True)
        amex_metrics.append(amex_metric(y_v, bst.predict(dv, iteration_range=(0, bst.best_iteration + 1))))
    return np.mean(amex_metrics), np.std(amex_metrics)

def remove_hard(X_train, y_train, X_v, y_v, ind, n, k):
    '''
    Removes the hardest points and retrains.
    '''
    with open('../config/xgboost.json', 'r') as f:
        params = json.load(f)
    amex_metrics_means = []
    amex_metrics_stds = []
    for _ in tqdm(range(k + 1)):
        X_train_sub = X_train[ind[(n * _):]]
        y_train_sub = y_train[ind[(n * _):]]
        means, stds = train_xgb(X_train_sub, y_train_sub, X_v, y_v, n=10)
        amex_metrics_means.append(means)
        amex_metrics_stds.append(stds)
    return amex_metrics_means, amex_metrics_stds
        