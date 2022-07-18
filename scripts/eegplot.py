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
import matplotlib.pyplot as plt

pipeline_path = f"data/eeg_roi_multi_ Stratified-Group2.pickle"

pl = pickle.load(open(pipeline_path, "rb"))


clfs = [key for key, val in viz.CLASSIFIERS.items()]
for clf in clfs:
    viz.metric_balance(pl, clf, show=False)
    fig = plt.gcf()
    fig.savefig(pipeline_path.replace(".pickle", f"_{clf}.png"))

viz.data_distribution(pl, show=False)
fig = plt.gcf()
fig.savefig(pipeline_path.replace(".pickle", f"_dist.png"))
