from imbalance.pipeline import Pipeline
from imbalance import viz
import numpy as np
from imbalance.data import gaussian_binary
# load or generate dataset

x, y, groups = gaussian_binary(mean_distance = 1,
                               n_samples_per_class = 1500)

# initialize pipeline
pl = Pipeline(x, y,
              classifiers=["lr"],
              metrics=["accuracy", "balanced_accuracy", "roc_auc", "f1"],
              dataset_balance=np.linspace(0.1, 0.9, 25),
              n_permutations = 100,
              dataset_size=(1, 0.1, 0.333),
              n_init = 10)


# fit and evaluate classifiers on dataset configurations
pl.evaluate()

viz.plot_different_n(pl, classifier = "lr", metric="accuracy")

# visualize classification scores
#viz.metric_balance(pl, classifier = "lda")
#viz.metric_balance(pl, classifier = "svm")
#viz.metric_balance(pl, classifier = "lr")

breakpoint()
