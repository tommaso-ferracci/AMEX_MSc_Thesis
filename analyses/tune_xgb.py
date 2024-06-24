import pickle
import numpy as np

import sys, os
sys.path.append(os.path.join(os.getcwd(), '..'))

from src.utils.train_xgb import train_xgb

X_train = np.load('../data/processed/train.npz')['x']
y_train = np.load('../data/processed/train.npz')['y']
X_v = np.load('../data/processed/v.npz')['x']
y_v = np.load('../data/processed/v.npz')['y']

# Define the parameter grid (108 combinations)
max_depths = [2, 4, 6, 8]
learning_rates = [0.01, 0.03, 0.05]
subsamples = [0.6, 0.8, 1.]
colsamples_bytree = [0.5, 0.6, 0.7]

metrics_list = []
params_list = []
for max_depth in max_depths:
    for learning_rate in learning_rates:
        for subsample in subsamples:
            for colsample_bytree in colsamples_bytree:
                params = {
                    'objective': 'binary:logistic',
                    'device': 'cuda',
                    'max_depth': max_depth,
                    'learning_rate': learning_rate, 
                    'subsample': subsample,
                    'colsample_bytree': colsample_bytree, 
                }
                mean, _ = train_xgb(X_train, y_train, X_v, y_v, n=5, params=params)
                metrics_list.append(mean)
                params_list.append(params)

with open('../outputs/results/params_list.pkl', 'wb') as f:
    pickle.dump(params_list, f)
with open('../outputs/results/metrics_list.pkl', 'wb') as f:
    pickle.dump(metrics_list, f)
    