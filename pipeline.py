import warnings
from tqdm import tqdm
from typing import Sequence, List, Tuple, Union, Optional, Dict, Any
import numpy as np
from sklearn.base import clone, BaseEstimator
from sklearn.model_selection import (
    BaseCrossValidator,
    KFold,
    GroupKFold,
    permutation_test_score,
)
from sklearn.metrics import SCORERS
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


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
                        sklearn.metrics.SCORERS.keys())
        cross_validation (CrossValidator): an instantiated scikit-learn cross validation
                                           strategy (if None, uses (Group)KFold with n=5)
        dataset_balance (array-like): sequence of balance ratios where 0 corresponds to
                                      100% of class 0, 0.5 is balanced and 1 is 100% of
                                      class 1
        dataset_size (str, array-like): "full" or sequence of ratios to evaluate different
                                        dataset sizes
        n_permutations (int): number of permutations to run for evaluating statistical significance
                              (set to 0 to disable permutation tests)
    """

    CLASSIFIERS: Dict[str, BaseEstimator] = {
        "lr": LogisticRegression,
        "svm": SVC,
        "lda": LinearDiscriminantAnalysis,
        "rf": RandomForestClassifier,
    }

    def __init__(
        self,
        x: Sequence[float],
        y: Sequence[int],
        groups: Optional[Sequence[int]] = None,
        classifiers: Union[str, BaseEstimator, List[Any]] = "lr",
        metrics: List[str] = ["accuracy", "roc_auc"],
        cross_validation: BaseCrossValidator = None,
        dataset_balance: Sequence[float] = [0.5],  # [0.1, 0.3, 0.5, 0.7, 0.9],
        dataset_size: Union[str, Sequence[float]] = "full",
        n_permutations: int = 0,
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

        # other parameters
        self.n_permutations = n_permutations
        self.metrics = metrics

        # initialize results as None
        self.scores = None

    def evaluate(self):
        """Fits all classifiers on different configurations of data and
        evaluates their performance using the specified list of metrics.

        If n_permutations is greated than zero, runs permutation tests to assess statistical significance.

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
                        metric1: (score-metric1, p-value-metric1),
                        metric2: (score-metric2, p-value-metric2),
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
            * len(self.classifiers)
            * len(self.metrics)
        )
        pbar = tqdm(desc="fitting classifiers", total=num_loops)

        # initialize result dictionary
        results = {}

        # nested loops over data configurations, classifiers and metrics
        for dset_balance in self.dataset_balance:
            results[dset_balance] = {}

            # adjust class balance by dropping as few samples as possible
            unbalanced_x, unbalanced_y, unbalanced_groups = Pipeline.unbalance_data(
                dset_balance, self.x, self.y, self.groups
            )

            for dset_size in self.dataset_size:
                results[dset_balance][dset_size] = {}

                # limit the size of the dataset while maintaining class distribution
                curr_x, curr_y, curr_groups = Pipeline.limit_dataset_size(
                    dset_size, unbalanced_x, unbalanced_y, unbalanced_groups
                )

                # make sure none of the classes have 0 samples
                assert (np.unique(curr_y, return_counts=True)[1] > 0).all(), (
                    f"one class has no samples for balance={dset_balance} "
                    f"and size={dset_size}. Increase the total dataset size "
                    f"or choose less extreme configurations"
                )

                # shuffle the dataset randomly
                curr_x, curr_y, curr_groups = Pipeline.shuffle(
                    curr_x, curr_y, curr_groups
                )

                for clf in self.classifiers:
                    clf_name = type(clf).__name__
                    results[dset_balance][dset_size][clf_name] = {}

                    for metric in self.metrics:
                        # add current configuration to progress bar
                        pbar.set_postfix(
                            dict(
                                size=dset_size,
                                balance=dset_balance,
                                classifier=clf_name,
                                metric=metric,
                            )
                        )

                        # run a permutation test for the current combination of
                        # imbalance, sample size, classifier and metric
                        score, _, pvalue = permutation_test_score(
                            clone(clf),
                            curr_x,
                            curr_y,
                            groups=curr_groups,
                            scoring=metric,
                            cv=self.cross_validation,
                            n_permutations=self.n_permutations,
                            n_jobs=-1,
                        )

                        # we don't have a p-value if the number of permutations is zero
                        if self.n_permutations == 0:
                            pvalue = None

                        # store current results
                        results[dset_balance][dset_size][clf_name][metric] = (
                            score,
                            pvalue,
                        )

                        # update the progress bar
                        pbar.update()
        pbar.close()
        self.scores = results

    def unbalance_data(
        ratio: float, x: np.ndarray, y: np.ndarray, groups: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Artificially unbalances a given dataset according to a balance ratio

        Args:
            ratio (float): balance ratio where 0 corresponds to 100% of class 0, 0.5 is balanced and 1 is
                           100% of class 1 (0 < ratio < 1)
            x (array-like): data
            y (array-like): labels
            groups (array-like): group labels

        Returns:
            (x, y, groups): rebalanced data, labels and group labels
        """
        # make sure we start with a balanced dataset
        unique_y, class_counts = np.unique(y, return_counts=True)
        assert (class_counts == class_counts[0]).all(), (
            f"classes have different numbers of "
            f"samples (counts: {list(class_counts)})"
        )

        # compute expected class sizes according to ratio
        n0 = class_counts[1] / ratio - class_counts[1]
        n1 = class_counts[1]

        # make sure the expected class sizes are realizable with the provided data
        alpha = min(class_counts[0] / n0, class_counts[1] / n1)
        n0 = int(n0 * alpha)
        n1 = int(class_counts[1] * alpha)

        # rebalance data
        idxs = np.concatenate([np.where(y == 0)[0][:n0], np.where(y == 1)[0][:n1]])
        return x[idxs], y[idxs], None if groups is None else groups[idxs]

    def limit_dataset_size(
        ratio: float, x: np.ndarray, y: np.ndarray, groups: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Limits the size of a given dataset according to a given ratio.

        Args:
            ratio (float): the ratio of data to keep (0 < ratio <= 1)
            x (array-like): data
            y (array-like): labels
            groups (array-like): group labels

        Returns:
            (x, y, groups): reduced data, labels and group labels
        """
        indices = np.concatenate(
            [
                np.where(y == c)[0][: int(count * ratio)]
                for c, count in zip(*np.unique(y, return_counts=True))
            ]
        )
        return x[indices], y[indices], None if groups is None else groups[indices]

    def shuffle(
        x: np.ndarray, y: np.ndarray, groups: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Shuffles data, labels and groups labels randomly.

        The arrays are shuffled along the first axis.

        Args:
            x (array-like): data
            y (array-like): labels
            groups (array-like): group labels

        Returns:
            (x, y, groups): shuffled data, labels and group labels
        """
        permutation = np.random.permutation(len(x))
        return (
            x[permutation],
            y[permutation],
            None if groups is None else groups[permutation],
        )

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
    from pprint import pprint

    # generate random data
    n = 10000
    x = np.random.normal(size=(n, 3))
    y = np.concatenate((np.zeros(n // 2), np.ones(n // 2))).astype(int)
    groups = np.arange(n // 5, dtype=int).repeat(5)

    # initialize the pipeline
    pl = Pipeline(x, y, groups)
    # fit and evaluate the classifiers with different configurations
    pl.evaluate()

    pprint(pl.scores)
