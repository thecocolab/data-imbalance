"""
EEG Test 1
Multi-feature experiment with Stratified Group K-Fold
"""

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold

from imbalance.data import eegbci
from imbalance.data.eeg import get_info
from imbalance.pipeline import Pipeline
from imbalance import viz

###############################################################################
# Configuration of the test
###############################################################################

cross_validation = StratifiedGroupKFold(n_splits=5)
cross_validation_label = "Stratified-Group"
n_features = 'multi'
dataset_balance=np.linspace(0.1, 0.9, 25)
classifiers=[
    "lda",
    "svm",
    LogisticRegression(max_iter=1000),
    RandomForestClassifier(n_estimators=25),
]
n_permutations=100
n_init=10

###############################################################################
# Region of Interest
###############################################################################
# Any electrode that starts with
#   P -> Parietal
#   O -> Occipital
roi = lambda x: x[0] in ["P", "O"] 

###############################################################################
#%% Path for results
###############################################################################

pipeline_path = f"data/eeg_roi_{n_features}_{cross_validation_label}.pickle"
features_path = f"data/eeg_features_{n_features}.npy"

###############################################################################
#%% Generate Features from the EEG signals
###############################################################################

if not os.path.isfile(features_path):
    print('Generating features!')
    x, y, groups = eegbci(
        "data", roi = roi, n_features=n_features
    )
    np.save(features_path, dict(x=x, y=y, groups=groups))
else:
    # Reuse pre-computed features
    print("Loading features from a previous run")
    features = np.load(features_path, allow_pickle=True).item()
    x, y, groups = features["x"], features["y"], features["groups"]

###############################################################################
#%% Run Pipeline on the features
###############################################################################

if not os.path.isfile(pipeline_path):
    pl = Pipeline(
        x,
        y,
        groups,
        dataset_balance=dataset_balance,
        classifiers=classifiers,
        n_permutations=n_permutations,
        n_init=n_init,
        cross_validation=cross_validation,
    )

    pl.evaluate()

    # Store pipeline for later re-use
    with open(pipeline_path, "wb") as handle:
        pickle.dump(pl, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    # Reuse a pre-computed pipeline
    print("Loading pipeline from a previous run")
    pl = pickle.load(open(pipeline_path, "rb"))

###############################################################################
#%% Plot Results
###############################################################################

# Feature Distribution

print('Plotting feature distribution')
viz.data_distribution(pl, show=False)
fig = plt.gcf()
fig.savefig(pipeline_path.replace(".pickle", f"_dist.png"))

# Balance vs Score Curves for each Classifier
print('Plotting Balance vs Score Curves')
clfs = [key for key, val in viz.CLASSIFIERS.items()]
for clf in clfs:
    viz.metric_balance(pl, clf, show=False)
    fig = plt.gcf()
    fig.savefig(pipeline_path.replace(".pickle", f"_{clf}.png"))

print('EEG multi-feature test done!')
