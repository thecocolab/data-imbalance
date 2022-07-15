from imbalance.data import eegbci
from imbalance.data.eeg import get_info
from imbalance.pipeline import Pipeline
from imbalance import viz
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import os
import numpy as np
import sys

from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    StratifiedGroupKFold,
)
cvsnames = ["KFold", "Stratified", "Stratified-Group"]
cvs = [KFold(n_splits=5), StratifiedKFold(n_splits=5), StratifiedGroupKFold(n_splits=5)]
single_balanced_split = True
if len(sys.argv) > 1:
    print('cmd entry:', sys.argv)
    cvs_idx = int(sys.argv[-1])
    cvs_idx = [cvs_idx]
else:
    cvs_idx = [1,0,2] # because we have groups by default

for ccc in cvs_idx:
    cv = cvs[ccc]

    for n_features in ['single','multi']:
        chans = get_info().ch_names

        pipeline_path=f'data/eeg_roi_{n_features}_{cvsnames[ccc]}.pickle'
        features_path =f"data/eeg_features_{n_features}.npy"

        if single_balanced_split:
            pipeline_path.replace('.pickle','_single_balanced_split.pickle')

        if not os.path.isfile(features_path):
            x, y, groups = eegbci('data',roi=lambda x: x[0] in ['P','O'],n_features=n_features)
            np.save(features_path,dict(x=x, y=y, groups=groups))
        else:
            print("loading features from a previous run")
            features = np.load(features_path,allow_pickle=True).item()
            x, y, groups = features["x"] , features["y"] , features["groups"]


        if not os.path.isfile(pipeline_path):
            pl = Pipeline(
                x,
                y,
                groups,
                dataset_balance = np.linspace(0.1, 0.9, 25),
                classifiers = ["lda","svm",LogisticRegression(max_iter=1000),RandomForestClassifier(n_estimators=25)],
                n_permutations = 100,
                n_init = 10,
                cross_validation=cvs[ccc],
                single_balanced_split=single_balanced_split
            )
            # fit and evaluate classifiers on dataset configurations
            pl.evaluate()

            # Store data (serialize)
            with open(pipeline_path, 'wb') as handle:
                pickle.dump(pl, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print("loading pipeline from a previous run")

            pl = pickle.load(open(pipeline_path,"rb"))


    # clfs = [key for key,val in viz.CLASSIFIERS.items()]
    # for clf in clfs:
    #     viz.metric_balance(pl,clf)

    # viz.data_distribution(pl)
