import warnings
from tqdm import tqdm
from typing import Sequence, List, Union, Optional, Dict, Any
import numpy as np
from sklearn.base import clone, BaseEstimator
from sklearn.model_selection import BaseCrossValidator, KFold, GroupKFold
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC


class Pipeline:
    """A pipeline to measure robustness of different metrics under different amounts of
    data unbalance, dataset size, classifier type and cross-validation strategy.

    Args:
        x (array-like): dataset with shape (num-samples, num-variables) or (num-samples,)
        y (array-like): labels with shape (num-samples,), each class should correspond to
                        a unique integer
        groups (array-like): array with the same shape as y, containing group indices for
                             each sample (optional)
        classifiers (str, list): a list of strings or scikit-learn classifier instances
                                 (see Pipeline.CLASSIFIERS.keys())
        metrics (list): a list of strings with metrics that should be compared (see
                        Pipeline.METRICS)
        cross_validation (CrossValidator): an instantiated scikit-learn cross validation
                                           strategy (if None, uses (Group)KFold with n=5)
        dataset_balance (array-like): sequence of balance ratios where 0 corresponds to
                                      100% of class 0, 0.5 is balanced and 1 is 100% of
                                      class 1
        dataset_size (str, array-like): "full" or sequence of ratios to evaluate different
                                        dataset sizes
    """

    METRICS: List[str] = ["acc", "auc", "f1"]
    CLASSIFIERS: Dict[str, BaseEstimator] = {
        "logistic-regression": LogisticRegression,
        "svm": SVC,
        "lda": LinearDiscriminantAnalysis,
    }

    def __init__(
        self,
        x: Sequence[float],
        y: Sequence[int],
        groups: Optional[Sequence[int]] = None,
        classifiers: Union[str, BaseEstimator, List[Any]] = "logistic-regression",
        metrics: List[str] = ["acc", "auc"],
        cross_validation: BaseCrossValidator = None,
        dataset_balance: Sequence[float] = [0.1, 0.3, 0.5, 0.7, 0.9],
        dataset_size: Union[str, Sequence[float]] = "full",
    ):
        # check x and y parameters
        x, y = np.asarray(x), np.asarray(y)
        assert x.ndim in [
            1,
            2,
        ], f"x must be 1- or 2-dimensional, got {x.ndim}D"
        assert (
            y.ndim == 1 and y.dtype == int
        ), f"y must be a 1D integer array, got {y.ndim}D with type {y.dtype}"
        n_classes = len(np.unique(y))
        assert n_classes == 2, (
            f"for now, only binary classification is "
            f"implemented (got {n_classes} classes)"
        )
        assert len(x) == len(
            y
        ), f"x and y must have the same length (got {len(x)} and {len(y)})"

        # warn if the dataset is unbalanced from the start
        class_counts = np.unique(y, return_counts=True)
        if class_counts[1][0] != class_counts[1][1]:
            class_counts_str = ", ".join(
                f"{c}: {count} samples" for c, count in zip(*class_counts)
            )
            warnings.warn(
                f"the dataset is unbalanced, which is currently not handled "
                f"properly. Results might be biased ({class_counts_str})"
            )

        # store dataset
        self.x = x.reshape(x.shape[0], -1)
        self.y = y

        # check groups parameter
        if groups is not None:
            groups = np.asarray(groups)
            assert groups.ndim == 1 and groups.dtype == int, (
                f"groups must be a 1D integer array, "
                f"got {groups.ndim}D with type {groups.dtype}"
            )
            assert len(x) == len(groups), (
                f"groups must have the same length as x and y "
                f"(got {len(x)} and {len(groups)})"
            )
        self.groups = groups

        # initialize classifiers
        self.classifiers: List[BaseEstimator] = []
        if not isinstance(classifiers, list):
            classifiers = [classifiers]

        for clf in classifiers:
            if isinstance(clf, str):
                # instantiate a new classifier object
                self.classifiers.append(Pipeline.get_classifier(clf))
            else:
                # clone the classifier object
                self.classifiers.append(clone(clf))

        # check metrics parameter
        for metric in metrics:
            assert metric in Pipeline.METRICS, (
                f"unknown metric {metric}, choose from "
                f"{', '.join(Pipeline.METRICS)}"
            )
        self.metrics = metrics

        # initialize cross-validation
        if cross_validation is None and self.groups is None:
            cross_validation = KFold(n_splits=5)
        elif cross_validation is None:
            cross_validation = GroupKFold(n_splits=5)
        self.cross_validation = cross_validation

        # store balance and dataset size ratios
        for bal in dataset_balance:
            assert 0 < bal < 1, f"expected dataset balance to be > 0 and < 1, got {bal}"
        self.dataset_balance = dataset_balance

        if isinstance(dataset_size, str):
            assert dataset_size == "full", f"unknown dataset size '{dataset_size}'"
            dataset_size = [1.0]
        for size in dataset_size:
            assert (
                0 < size <= 1
            ), f"expected dataset size to be > 0 and <= 1, got {size}"
        self.dataset_size = dataset_size

        # initialize results as None
        self.scores = None

    def fit(self):
        """Fits all classifiers on different configurations of data and
        evaluates their performance using the specified list of metrics.

        The dataset is shuffled randomly after adjusting class balance and dataset size.

        This function fits classifiers inside a nested loop structure as follows:
        dataset balance -> dataset size -> cross validation split -> classifier type

        The results are saved in nested dictionaries and can be accessed through
        pipeline_instance.scores. The dictionary is structued in the following way:
        scores = {
            # dataset balance ratio
            0.3: {
                # dataset size ratio
                0.5: {
                    # classifier name
                    clf1: {
                        # metric name
                        metric1: score-metric1,
                        metric2: score-metric2,
                        ...
                    },
                    clf2: ...
                },
                0.7: ...
            },
            0.5: ...
        }
        """
        num_loops = (
            len(self.dataset_balance)
            * len(self.dataset_size)
            * self.cross_validation.get_n_splits(self.x, self.y, self.groups)
            * len(self.classifiers)
        )
        pbar = tqdm(desc="fitting classifiers", total=num_loops)

        # initialize result dictionary
        results = {}

        # nested loops over data configurations, classifiers and metrics
        for dset_balance in self.dataset_balance:
            results[dset_balance] = {}

            # make sure we start with a balanced dataset
            unique_y, class_counts = np.unique(self.y, return_counts=True)
            assert (class_counts == class_counts[0]).all(), (
                f"classes have different numbers of "
                f"samples (counts: {list(class_counts)})"
            )

            # adjust class balance by dropping as few samples as possible
            alpha = dset_balance / (1 - dset_balance)
            n_first_class = int(alpha * class_counts[0])
            indices = np.concatenate(
                [np.where(self.y == 0)[0][:n_first_class], np.where(self.y == 1)[0]]
            )
            curr_x = self.x[indices]
            curr_y = self.y[indices]
            curr_groups = None if self.groups is None else self.groups[indices]

            for dset_size in self.dataset_size:
                # limit the size of the dataset while maintaining class distribution
                indices = np.concatenate(
                    [
                        np.where(curr_y == c)[0][: int(count * dset_size)]
                        for c, count in zip(*np.unique(curr_y, return_counts=True))
                    ]
                )

                curr_x = curr_x[indices]
                curr_y = curr_y[indices]
                curr_groups = None if curr_groups is None else curr_groups[indices]

                # make sure none of the classes have 0 samples
                assert (np.unique(curr_y, return_counts=True)[1] > 0).all(), (
                    f"one class has no samples for balance={dset_balance} "
                    f"and size={dset_size}. Increase the total dataset size "
                    f"or choose less extreme configurations"
                )

                # shuffle the dataset randomly
                permutation = np.random.permutation(len(curr_x))
                curr_x = curr_x[permutation]
                curr_y = curr_y[permutation]
                curr_groups = None if curr_groups is None else curr_groups[permutation]

                # create a cross-validation split
                cv_splits = self.cross_validation.split(curr_x, curr_y, curr_groups)

                cv_results = {}
                for idx_train, idx_test in cv_splits:
                    # split dataset according to cross-validation split
                    x_train, y_train = curr_x[idx_train], curr_y[idx_train]
                    x_test, y_test = curr_x[idx_test], curr_y[idx_test]

                    for clf in self.classifiers:
                        clf_name = type(clf).__name__
                        if clf_name not in cv_results:
                            cv_results[clf_name] = {}

                        # add current configuration to progress bar
                        pbar.set_postfix(
                            dict(
                                size=dset_size,
                                balance=dset_balance,
                                classifier=clf_name,
                            )
                        )

                        # make sure we get a new and untrained classifier
                        clf = clone(clf)
                        # train the current classifier
                        clf.fit(x_train, y_train)
                        # evaluate the classifier using all specified metrics
                        scores = self.evaluate_classifier(clf, x_test, y_test)

                        # append scores to a list to accumulate results
                        # from individual cross-validation splits
                        for metric, val in scores.items():
                            if metric not in cv_results[clf_name]:
                                cv_results[clf_name][metric] = []
                            cv_results[clf_name][metric].append(val)

                        # update the progress bar
                        pbar.update()

                # aggregate results over cross-validation splits
                for clf_name in cv_results.keys():
                    for metric in cv_results[clf_name].keys():
                        avg_score = np.mean(cv_results[clf_name][metric])
                        cv_results[clf_name][metric] = avg_score
                # store the aggregated scores in the results dict
                results[dset_balance][dset_size] = cv_results
        self.scores = results

    def evaluate_classifier(
        self, clf: BaseEstimator, x: np.ndarray, y: np.ndarray
    ) -> Dict[str, float]:
        """Evaluates different metrics on a fitted classifier.

        Args:
            clf (Estimator): a fitted scikit-learn classifier
            x (array): the testing data with shape (num-samples x num-variables)
            y (array): the testing labels with shape (num-samples,)

        Returns:
            result (dict): a dictionary with metric names as keys and scores as values
        """
        result = {}
        for metric in self.metrics:
            if metric == "acc":
                result[metric] = clf.score(x, y)
            elif metric == "auc":
                result[metric] = roc_auc_score(y, clf.decision_function(x))
            elif metric == "f1":
                result[metric] = f1_score(y, clf.predict(x))
        return result

    def get_classifier(name: str) -> BaseEstimator:
        """Instantiates a new classifier based on its name.

        Args:
            name (str): the name of the classifier to be created (see Pipeline.CLASSIFIERS)

        Returns:
            clf (BaseEstimator): unfitted scikit-learn classifier
        """
        assert name in Pipeline.CLASSIFIERS, (
            f"unknown classifier {name}, choose from "
            f"{', '.join(Pipeline.CLASSIFIERS.keys())}"
        )
        return Pipeline.CLASSIFIERS[name]()


if __name__ == "__main__":
    # generate random data
    n = 10000
    x = np.random.normal(size=(n, 3))
    y = np.concatenate((np.zeros(n // 2), np.ones(n // 2))).astype(int)
    groups = np.arange(n // 5, dtype=int).repeat(5)

    # initialize the pipeline
    p = Pipeline(x, y, groups)
    # fit the classifiers with different configurations
    p.fit()
