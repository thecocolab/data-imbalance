from imbalance.data import eegbci
from imbalance.data.eeg import get_info
from imbalance.pipeline import Pipeline
from imbalance import viz
import pickle
from sklearn.linear_model import LogisticRegression
import os
import numpy as np

pipeline_path='data/eeg_roi.pickle'
features_path ="data/eeg_features.npy"
chans = get_info().ch_names

if not os.path.isfile(features_path):
    x, y, groups = eegbci('data',roi=lambda x: x[0] in ['P','O'])
    np.save(features_path,dict(x=x, y=y, groups=groups))
else:
    features = np.load(features_path,allow_pickle=True).item()
    x, y, groups = features["x"] , features["y"] , features["groups"]

if not os.path.isfile(pipeline_path):
    pl = Pipeline(
        x,
        y,
        groups,
        dataset_balance = np.linspace(0.1, 0.9, 25),
        classifiers = ["lda","svm",LogisticRegression(max_iter=1000)],#,'rf'],
        n_permutations = 10,
        n_init = 2,
    )
    # fit and evaluate classifiers on dataset configurations
    pl.evaluate()

    # Store data (serialize)
    with open(pipeline_path, 'wb') as handle:
        pickle.dump(pl, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    pl = pickle.load(open(pipeline_path,"rb"))
