import sys, os
sys.path.append(os.path.join(os.getcwd(), '..'))

import pickle
import numpy as np
import pandas as pd

from sdv.metadata import SingleTableMetadata
from sdv.single_table import TVAESynthesizer, CTGANSynthesizer
from baytune import BTBSession
from baytune.tuning import Tunable
from baytune.tuning import hyperparams as hp

from src.utils.train_xgb import train_xgb

def append_to_pickle_file(pickle_file, new_data):
    try:
        with open(pickle_file, 'rb') as f:
            existing_data = pickle.load(f)
        if not isinstance(existing_data, list):
            existing_data = [existing_data]
        existing_data.append(new_data)
        with open(pickle_file, 'wb') as f:
            pickle.dump(existing_data, f)
    except FileNotFoundError:
        with open(pickle_file, 'wb') as f:
            pickle.dump([new_data], f)

# Import data and create DataFrame of data to augment
X_train = np.load('../data/processed/train.npz')['x']
y_train = np.load('../data/processed/train.npz')['y']
X_v = np.load('../data/processed/v.npz')['x']
y_v = np.load('../data/processed/v.npz')['y']

knn_raw = pd.read_csv('../outputs/results/knn.csv')['0']
ind_raw = knn_raw.sort_values(ascending=False).index
X_worst = X_train[ind_raw[-50000:]]
y_worst = y_train[ind_raw[-50000:]]

X_worst_df = pd.DataFrame(X_worst)
X_worst_df['target'] = y_worst

# Infer metadata: categorical and numeric features
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(X_worst_df)

# Compute average score on all data
tot_score, _ = train_xgb(X_train, y_train, X_v, y_v, n=5)

def get_xgboost_score(X_train, y_train):
    # Compute difference in average score with respect to all data
    new_score, _ = train_xgb(X_train, y_train, X_v, y_v, n=5)
    return new_score - tot_score

mods = {
    'TVAE': TVAESynthesizer,
    'CTGAN': CTGANSynthesizer,
}

def scoring_function(mod_name, hyperparams):
    '''
    Scorer for a synthesizer: quantifies validation performance variation 
    when synthetic data is added to training dataset.
    '''
    mod_class = mods[mod_name]
    mod_instance = mod_class(metadata, **hyperparams)
    mod_instance.fit(X_worst_df)
    scores = []
    for _ in range(10): # Repeat 10 times to mitigate randomness
        synthetic_data = mod_instance.sample(num_rows=50000)
        X_synth = synthetic_data.drop('target', axis=1).values
        y_synth = np.array(synthetic_data['target'])
        X_train_aug = np.vstack((X_train, X_synth))
        y_train_aug = np.concatenate((y_train, y_synth))
        scores.append(get_xgboost_score(X_train_aug, y_train_aug))
    append_to_pickle_file('../outputs/results/hyperparams_auc_hard.pkl', hyperparams)
    append_to_pickle_file('../outputs/results/scores_auc_hard.pkl', scores)
    return np.mean(scores)

# Candidate models and their hyperparameter sets
tunables = {
    'TVAE': Tunable({
    'batch_size': hp.IntHyperParam(min=100, max=1000, default=500, step=100),
    'epochs': hp.IntHyperParam(min=50, max=300, default=100, step=50),
    'embedding_dim': hp.IntHyperParam(min=64, max=512,default=128, step=64),
}),
    'CTGAN': Tunable({
    'batch_size': hp.IntHyperParam(min=100, max=1000,default=500, step=100),
    'epochs': hp.IntHyperParam(min=20, max=200, default=50, step=10),
    'embedding_dim': hp.IntHyperParam(min=64, max=512,default=128, step=64),
})
}

session = BTBSession(
    tunables=tunables,
    scorer=scoring_function,
    verbose=True
)

best_prop = session.run(30)
print(best_prop)

# Dump session results
with open('../outputs/synthesizers/session_hard.pkl', "wb") as f:
    pickle.dump(session, f)
    