from imbalance.data import eegbci
from imbalance.pipeline import Pipeline
from imbalance import viz
import pickle
from sklearn.linear_model import LogisticRegression
import os
import numpy as np

pipeline_path='data/eeg.pickle'
features_path ="data/eeg_features.npy"

if not os.path.isfile(pipeline_path):

    if not os.path.isfile(features_path):
        # load or generate dataset
        x, y, groups = eegbci('data',roi=lambda x: x[0] in ['P','O'])
        np.save(features_path,dict(x=x, y=y, groups=groups))
    else:
        features = np.load(features_path,allow_pickle=True).item()
        x, y, groups = features["x"] , features["y"] , features["groups"]

    pl = Pipeline(
        x,
        y,
        groups,
        dataset_balance=np.linspace(0.037, 1-0.037, 100)[1:-1],
        classifiers=['rf',"lda","svm",LogisticRegression(max_iter=1000)],
        metrics=[ "roc_auc","accuracy", "f1", "balanced_accuracy"],
    )
    # fit and evaluate classifiers on dataset configurations
    pl.evaluate()

    # Store data (serialize)
    with open(pipeline_path, 'wb') as handle:
        pickle.dump(pl, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    pl = pickle.load(open(pipeline_path,"rb"))

viz.metric_balance(pl)