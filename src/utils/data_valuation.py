import json
import numpy as np
import xgboost as xgb

from tqdm import tqdm

from src.utils.data_iq import DataIQ_xgb
from src.utils.amex_metric import amex_scorer

def compute_knn_shapley(X_train, y_train, X_test, y_test, k=5):
    '''
    Implementation of the exact formula for data shapley for nearest neighbor classifiers from Jia et al., 2019.
    '''
    N = len(X_train)
    N_test = len(X_test)
    ds = np.zeros((N, N_test), dtype=np.float32) # shapley for each test point

    for n, (X, y) in tqdm(enumerate(zip(X_test, y_test))):
        diff = (X_train - X).reshape(N, -1)
        dist = np.einsum('ij, ij->i', diff, diff)
        idx = np.argsort(dist)
        ans = y_train[idx]
        ds[idx[N - 1]][n] = float(ans[N - 1] == y) / N
        cur = N - 2
        for _ in range(N - 1):
            ds[idx[cur]][n] = ds[idx[cur + 1]][n] + float(int(ans[cur] == y) - int(ans[cur + 1] == y)) \
                / k * min(cur + 1, k) / (cur + 1)
            cur -= 1 
    return np.mean(ds, axis=1) # average shapley across test set

def compute_forget(X_train, y_train, X_test, y_test):
    '''
    Forgetting events counter over boosting rounds adapted from Toneva et al., 2019.
    '''
    with open('../config/xgboost.json', 'r') as f:
        params = json.load(f)
    dtrain = xgb.DMatrix(X_train, y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    bst = xgb.train(params, dtrain, num_boost_round=9999, verbose_eval=0,
                    evals=[(dtrain, 'train'), (dtest, 'test')], custom_metric=amex_scorer, 
                    early_stopping_rounds=100, maximize=True)
    forget = np.zeros(len(X_train)) # FORGETTING EVENTS COUNTER
    learnt = np.zeros(len(X_train)) # BINARY FLAG FOR LEARNT DATAPOINTS
    prev_pred = np.round(bst.predict(dtrain, iteration_range=(0, 0)))
    learnt[(prev_pred == 1)] += 1 
    for ix in tqdm(range(1, bst.best_iteration + 1)):
        y_pred = np.round(bst.predict(dtrain, iteration_range=(0, ix)))
        forget[(y_pred == 0) & (prev_pred == 1)] += 1
        learnt[(y_pred == 1) & (learnt == 0)] += 1
        prev_pred = y_pred
    return forget, learnt

def compute_dataiq(X_train, y_train, X_test, y_test):
    '''
    Estimation of aleatoric and epistematic uncertainty for xgboost as per Seedat et al., 2022.
    '''
    with open('../config/xgboost.json', 'r') as f:
        params = json.load(f)
    dataiq_xgb = DataIQ_xgb(X_train=X_train, y_train=y_train)
    dtrain = xgb.DMatrix(X_train, y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    bst = xgb.train(params, dtrain, num_boost_round=9999, verbose_eval=0,
                    evals=[(dtrain, 'train'), (dtest, 'test')], custom_metric=amex_scorer, 
                    early_stopping_rounds=100, maximize=True)
    for ix in tqdm(range(1, bst.best_iteration + 1)):
        dataiq_xgb.on_epoch_end(bst=bst, iteration=ix)
    return dataiq_xgb.aleatoric, dataiq_xgb.confidence
    