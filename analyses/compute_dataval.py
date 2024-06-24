import numpy as np
import pandas as pd

import sys, os
sys.path.append(os.path.join(os.getcwd(), '..'))

from src.utils.data_valuation import *

X_train = np.load('../data/processed/train.npz')['x']
y_train = np.load('../data/processed/train.npz')['y']
X_test = np.load('../data/processed/test.npz')['x']
y_test = np.load('../data/processed/test.npz')['y']

# KNN requires standardized features!
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
std_X_train = sc.transform(X_train)
std_X_test = sc.transform(X_test)

forget, learnt = compute_forget(X_train, y_train, X_test, y_test)
pd.Series(forget).to_csv('../outputs/results/forget.csv')
pd.Series(learnt).to_csv('../outputs/results/learnt.csv')
aleatoric, confidence = compute_dataiq(X_train, y_train, X_test, y_test)
pd.Series(aleatoric).to_csv('../outputs/results/aleatoric.csv')
pd.Series(confidence).to_csv('../outputs/results/confidence.csv')
knn_1 = compute_knn_shapley(std_X_train, y_train, std_X_test, y_test, k=1)
pd.Series(knn_1).to_csv('../outputs/results/knn_1.csv')
knn_5 = compute_knn_shapley(std_X_train, y_train, std_X_test, y_test, k=5)
pd.Series(knn_5).to_csv('../outputs/results/knn_5.csv')
knn_100 = compute_knn_shapley(std_X_train, y_train, std_X_test, y_test, k=100)
pd.Series(knn_100).to_csv('../outputs/results/knn_100.csv')
