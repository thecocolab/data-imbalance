import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class Pipeline:
    """
    A pipeline to measure robustness of different metrics under different amounts of unbalance of the data.

    Args:
        x (array-like): dataset with shape (num-samples x num-variables) or (num-samples,)
        y (array-like): labels with shape (num-samples,), each class should correspond to a unique integer
        groups (array-like): array with the same shape as y, containing group indices for each sample (optional)
        classifiers (list): a list of strings or scikit-learn classifier instances (for options see Pipeline.CLASSIFIERS.keys())
        metrics (list): a list of strings with metrics that should be compared (for options see Pipeline.METRICS)
        cross_validation (str): 
    """

    METRICS = ["acc", "auc", "f1"]
    CLASSIFIERS = {"logistic-regression": LogisticRegression, "svm": SVC}
    CROSS_VALIDATIONS = ["LeavePOut", "KFold", "StratifiedKFold"]

    def __init__(
        self,
        x,
        y,
        groups=None,
        classifiers=["logistic-regression"],
        metrics=["acc", "auc"],
        cross_validation="KFold",
    ):
        # check x and y
        x, y = np.asarray(x), np.asarray(y)
        assert x.ndim in [
            1,
            2,
        ], f"x must be 1- or 2-dimensional, got {x.ndim}D"
        assert (
            y.ndim == 1 and y.dtype == int
        ), f"y must be a 1D integer array, got {y.ndim}D with type {y.dtype}"
        n_classes = len(np.unique(y))
        assert (
            n_classes == 2
        ), f"for now, only binary classification is implemented (got {n_classes} classes)"

        class_counts = np.unique(y, return_counts=True)
        if class_counts[1][0] != class_counts[1][1]:
            warnings.warn(
                f"the dataset is unbalanced, results might be biased "
                f"({', '.join(f'{c}: {count} samples' for c, count in zip(*class_counts))})"
            )
        self.x = x.reshape(x.shape[0], -1)
        self.y = y

        # check groups
        if groups is not None:
            groups = np.asarray(groups)
            assert (
                groups.ndim == 1 and groups.dtype == int
            ), f"groups must be a 1D integer array, got {groups.ndim}D with type {groups.dtype}"
        self.groups = groups

        # initialize classifiers
        self.classifiers = []
        for clf in classifiers:
            if isinstance(clf, str):
                self.classifiers.append(Pipeline.get_classifier(clf))
            else:
                self.classifiers.append(clf)

        # check metrics
        for metric in metrics:
            assert (
                metric in Pipeline.METRICS
            ), f"unknown metric {metric}, choose from {', '.join(Pipeline.METRICS)}"
        self.metrics = metrics

    def fit(self):
        for clf in self.classifiers:
            clf.fit(self.x, self.y)

    def get_classifier(name):
        assert (
            name in Pipeline.CLASSIFIERS
        ), f"unknown classifier {name}, choose from {', '.join(Pipeline.CLASSIFIERS.keys())}"
        return Pipeline.CLASSIFIERS[name]()


if __name__ == "__main__":
    x = np.random.normal(size=(100, 3))
    y = np.zeros(x.shape[0], dtype=int)
    y[50:] = 1
    groups = np.arange(20, dtype=int).repeat(5)

    p = Pipeline(x, y, groups)
    p.fit()
