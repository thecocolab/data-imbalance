import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

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
