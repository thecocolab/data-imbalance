import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from imbalance import viz
from imbalance.data import fmri_haxby
from imbalance.pipeline import Pipeline

pipeline_path = f"data/fmri.pickle"
cross_validation = StratifiedKFold(n_splits=5)
classifiers = {
    "lr": LogisticRegression(max_iter=1000),
}
n_permutations = 100
n_init = 10

# load the data
x, y = fmri_haxby("data")

# run the pipeline
pl = Pipeline(
    x,
    y,
    classifiers=list(classifiers.values()),
    n_permutations=n_permutations,
    n_init=n_init,
    cross_validation=cross_validation,
)
pl.evaluate()

# save the pipeline
with open(pipeline_path, "wb") as handle:
    pickle.dump(pl, handle)

# Feature Distribution Plot
print("Plotting feature distribution")
viz.data_distribution(pl, show=False)
plt.gcf().savefig(pipeline_path.replace(".pickle", f"_dist.png"))

# Balance vs Score Curves for each Classifier
print("Plotting Balance vs Score Curves")
for clf in classifiers.keys():
    viz.metric_balance(pl, clf, show=False)
    plt.gcf().savefig(pipeline_path.replace(".pickle", f"_{clf}.png"))
