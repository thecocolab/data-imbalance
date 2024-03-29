import warnings
from tqdm import tqdm
from typing import Sequence, List, Tuple, Union, Optional, Dict, Any
import numpy as np
from sklearn.base import clone, BaseEstimator
from sklearn.model_selection import (
    BaseCrossValidator,
    StratifiedKFold,
    StratifiedGroupKFold,
    permutation_test_score,
)
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from imbalance.utils import classification


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
        cross_validation (CrossValidator): an instantiated scikit-learn cross validation
                                           strategy (if None, uses (Group)KFold with n=5)
        dataset_balance (array-like): sequence of balance ratios where 0 corresponds to
                                      100% of class 0, 0.5 is balanced and 1 is 100% of
                                      class 1
        dataset_size (str, array-like): "full" or sequence of ratios to evaluate different
                                        dataset sizes
        n_permutations (int): number of permutations to run for evaluating statistical significance
                              (set to 0 to disable permutation tests). Note that permutations will
                              only be computed for the first initialization (if n_init > 1).
        n_init (int): number of reinitialisation to run for evaluating variance in datasplit
                              (set to 1 to run only one time)
        single_balanced_split (bool): if True, only do a single cross-validation split with a balanced
                                      hold-out test set
    """

    CLASSIFIERS: Dict[str, BaseEstimator] = {
        "lr": LogisticRegression(),
        "svm": SVC(probability=True),
        "lda": LinearDiscriminantAnalysis(),
        "rf": RandomForestClassifier(n_estimators=25),
    }

    def __init__(
        self,
        x: Sequence[float],
        y: Sequence[int],
        groups: Optional[Sequence[int]] = None,
        classifiers: Union[str, BaseEstimator, List[Any]] = "lr",
        cross_validation: BaseCrossValidator = None,
        dataset_balance: Sequence[float] = np.linspace(0.1, 0.9, 25),
        dataset_size: Union[str, Sequence[float]] = "full",
        n_permutations: int = 100,
        rand_seed: int = 42,
        n_init: int = 10,
        single_balanced_split: bool = False,
    ):
        # check x and y parameters
        x, y = np.asarray(x), np.asarray(y)
        assert x.ndim in [1, 2], f"x must be 1- or 2-dimensional, got {x.ndim}D"
        assert y.ndim == 1 and np.issubdtype(
            y.dtype, np.integer
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
                f"The dataset is unbalanced, in order to generate balanced results, "
                f"undersampled data will be used ({class_counts_str})"
            )

        # store dataset
        self.x = x.reshape(x.shape[0], -1)
        self.y = y

        # check groups parameter
        if groups is not None:
            groups = np.asarray(groups)
            assert groups.ndim == 1 and np.issubdtype(groups.dtype, np.integer), (
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
            cross_validation = StratifiedKFold(n_splits=5)
        elif cross_validation is None:
            cross_validation = StratifiedGroupKFold(n_splits=5)
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
        self.n_init = n_init
        # For now we will use those metrics and we ignore the input metrics parameter
        self.metrics = ["accuracy", "roc_auc", "f1", "balanced_accuracy"]

        if single_balanced_split and groups is None:
            self.holdout_cv = StratifiedKFold(n_splits=10)
        elif single_balanced_split:
            self.holdout_cv = StratifiedGroupKFold(n_splits=10)
        self.single_balanced_split = single_balanced_split

        # set random seed
        self.seed = rand_seed

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
                        metric1: (meanscore-metric1, stdscore-metric1, p-value-metric1, permutation_score_metric1),
                        metric2: (meanscore-metric2, stdscore-metric2, p-value-metric2, permutation_score_metric2),
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
            * self.n_init
        )
        pbar = tqdm(desc="fitting classifiers", total=num_loops)

        # initialize result dictionary
        results = {}
        avg_results = {}  # results averaged over reinitialisation

        if self.seed is not None:
            np.random.seed(self.seed)

        for nr_init in range(self.n_init):
            results[nr_init] = {}
            # shuffle the dataset randomly
            shuffle_x, shuffle_y, shuffle_groups = Pipeline.shuffle(
                self.x, self.y, self.groups
            )

            if self.single_balanced_split:
                holdout_train_idxs, holdout_test_idxs = next(
                    self.holdout_cv.split(shuffle_x, shuffle_y, shuffle_groups)
                )
                # create holdout test set
                holdout_x_test = shuffle_x[holdout_test_idxs]
                holdout_y_test = shuffle_y[holdout_test_idxs]
                holdout_groups_test = (
                    None
                    if shuffle_groups is None
                    else shuffle_groups[holdout_test_idxs]
                )
                # create train set
                shuffle_x = shuffle_x[holdout_train_idxs]
                shuffle_y = shuffle_y[holdout_train_idxs]
                shuffle_groups = (
                    None
                    if shuffle_groups is None
                    else shuffle_groups[holdout_train_idxs]
                )

                # make sure the test set is balanced
                n_per_class_test = np.unique(holdout_y_test, return_counts=True)[1]
                assert (
                    n_per_class_test[0] == n_per_class_test
                ).all(), "the hold-out test set is unbalanced"

            # nested loops over data configurations, classifiers and metrics
            for dset_balance in self.dataset_balance:
                results[nr_init][dset_balance] = {}

                # adjust class balance by dropping as few samples as possible
                unbalanced_x, unbalanced_y, unbalanced_groups = Pipeline.unbalance_data(
                    dset_balance,
                    shuffle_x,
                    shuffle_y,
                    shuffle_groups,
                    seed=self.seed + nr_init,
                )

                for dset_size in self.dataset_size:
                    results[nr_init][dset_balance][dset_size] = {}

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
                        results[nr_init][dset_balance][dset_size][clf_name] = {}

                        # add current configuration to progress bar
                        pbar.set_postfix(
                            dict(
                                size=dset_size,
                                balance=dset_balance,
                                classifier=clf_name,
                            )
                        )

                        if self.single_balanced_split:
                            comb_x = np.concatenate([curr_x, holdout_x_test])
                            comb_y = np.concatenate([curr_y, holdout_y_test])
                            comb_groups = (
                                np.concatenate([curr_groups, holdout_groups_test])
                                if curr_groups is not None
                                else None
                            )
                            cv_splits = [
                                np.arange(len(curr_x)),
                                np.arange(len(holdout_x_test)),
                            ]

                            # TODO: run permutation tests only once and evaluate all metrics
                            accuracy = permutation_test_score(
                                clf,
                                comb_x,
                                comb_y,
                                groups=comb_groups,
                                scoring="accuracy",
                                cv=[cv_splits],
                                n_permutations=self.n_permutations,
                                n_jobs=-1,
                            )
                            balanced_accuracy = permutation_test_score(
                                clf,
                                comb_x,
                                comb_y,
                                groups=comb_groups,
                                scoring="balanced_accuracy",
                                cv=[cv_splits],
                                n_permutations=self.n_permutations,
                                n_jobs=-1,
                            )
                            roc_auc = permutation_test_score(
                                clf,
                                comb_x,
                                comb_y,
                                groups=comb_groups,
                                scoring="roc_auc",
                                cv=[cv_splits],
                                n_permutations=self.n_permutations,
                                n_jobs=-1,
                            )
                            f1 = permutation_test_score(
                                clf,
                                comb_x,
                                comb_y,
                                groups=comb_groups,
                                scoring="f1",
                                cv=[cv_splits],
                                n_permutations=self.n_permutations,
                                n_jobs=-1,
                            )

                            # curr_clf = clone(clf)
                            # curr_clf.fit(curr_x, curr_y)

                            # # predict on the holdout set
                            # y_test = holdout_y_test
                            # y_pred = curr_clf.predict_proba(holdout_x_test)[:, 1]

                            # # compute metrics
                            # try:
                            #     roc_auc = roc_auc_score(y_test, y_pred)
                            # except:
                            #     roc_auc = np.nan
                            # y_pred = (y_pred > 0.5).astype(int)
                            # f1 = f1_score(y_test, y_pred)
                            # balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
                            # accuracy = accuracy_score(y_test, y_pred)

                            # store test scores
                            curr_results_dict = results[nr_init][dset_balance][
                                dset_size
                            ][clf_name]
                            curr_results_dict["accuracy"] = (
                                accuracy[0],
                                accuracy[2],
                                np.nanmean(accuracy[1]),
                            )
                            curr_results_dict["roc_auc"] = (
                                roc_auc[0],
                                roc_auc[2],
                                np.nanmean(roc_auc[1]),
                            )
                            curr_results_dict["balanced_accuracy"] = (
                                balanced_accuracy[0],
                                balanced_accuracy[2],
                                np.nanmean(balanced_accuracy[1]),
                            )
                            curr_results_dict["f1"] = (f1[0], f1[2], np.nanmean(f1[1]))
                        else:
                            # only run the permutation test for the first  interation
                            if nr_init == 0:
                                # run a permutation test for the current combination of
                                # imbalance, sample size, classifier and metric
                                output = classification(
                                    clone(clf),
                                    self.cross_validation,
                                    curr_x,
                                    curr_y,
                                    groups=curr_groups,
                                    perm=self.n_permutations
                                    if self.n_permutations != 0
                                    else None,
                                    n_jobs=-1,
                                )
                            # don't run permutation test for other itertations
                            elif nr_init > 0:
                                # we don't have a p-value if the number of permutations is zero
                                output = classification(
                                    clone(clf),
                                    self.cross_validation,
                                    curr_x,
                                    curr_y,
                                    groups=curr_groups,
                                    perm=None,
                                    n_jobs=-1,
                                )

                            # store current results
                            results[nr_init][dset_balance][dset_size][clf_name][
                                "accuracy"
                            ] = (
                                output["acc_score"],
                                output["acc_pvalue"]
                                if "acc_pvalue" in output.keys()
                                else None,
                                np.nanmean(output["acc_pscores"])
                                if "acc_pscores" in output.keys()
                                else None,
                            )
                            results[nr_init][dset_balance][dset_size][clf_name][
                                "roc_auc"
                            ] = (
                                output["auc_score"],
                                output["auc_pvalue"]
                                if "auc_pvalue" in output.keys()
                                else None,
                                np.nanmean(output["auc_pscores"])
                                if "auc_pscores" in output.keys()
                                else None,
                            )
                            results[nr_init][dset_balance][dset_size][clf_name][
                                "balanced_accuracy"
                            ] = (
                                output["bacc_score"],
                                output["bacc_pvalue"]
                                if "bacc_pvalue" in output.keys()
                                else None,
                                np.nanmean(output["bacc_pscores"])
                                if "bacc_pscores" in output.keys()
                                else None,
                            )
                            results[nr_init][dset_balance][dset_size][clf_name][
                                "f1"
                            ] = (
                                output["f1_score"],
                                output["f1_pvalue"]
                                if "f1_pvalue" in output.keys()
                                else None,
                                np.nanmean(output["f1_pscores"])
                                if "f1_pscores" in output.keys()
                                else None,
                            )

                        # update the progress bar
                        pbar.update()
        pbar.close()

        # collect resutls and average over iterations
        for dset_balance in self.dataset_balance:
            avg_results[dset_balance] = {}

            for dset_size in self.dataset_size:
                avg_results[dset_balance][dset_size] = {}

                for clf in self.classifiers:
                    clf_name = type(clf).__name__
                    avg_results[dset_balance][dset_size][clf_name] = {}

                    for metric in self.metrics:
                        # store results from first iteration only
                        pvalue = results[0][dset_balance][dset_size][clf_name][metric][
                            1
                        ]
                        perm_score = results[0][dset_balance][dset_size][clf_name][
                            metric
                        ][2]

                        scores = []
                        for nr_init in range(self.n_init):
                            scores.append(
                                results[nr_init][dset_balance][dset_size][clf_name][
                                    metric
                                ][0]
                            )
                        # average the score over all iterations
                        score_mean = np.nanmean(scores)
                        score_std = np.std(scores)

                        avg_results[dset_balance][dset_size][clf_name][metric] = (
                            score_mean,
                            score_std,
                            pvalue,
                            perm_score,
                        )

        self.scores = avg_results

    def get(
        self,
        balance: Union[str, float, list] = "all",
        dataset_size: Union[str, float, list] = "all",
        classifier: Union[str, list] = "all",
        metric: Union[str, list] = "all",
        result_type: str = "both",
    ):
        """Collects subsets of the results and returns them as dictionaries. If a specific value
        or "max" is specified for a parameter, the corresponding dimension of the dictionary
        will be removed. For example, if "balance" is set to 0.3 and "dataset_size" is "all",
        the first dimension of the resulting dictionary will be dataset size.

        Args:
            balance (str, float, list): dataset balance ratio as a float, a list of ratios or
                                        "all" for the full range of balances
            dataset_size (str, float, list): dataset size as a float, a list of sizes, "all"
                                             for all sizes or "max" for only the maximum size
            classifier (str, list): name of the classifier to extract, a list of classifier
                                    names or "all" to return results from all classifiers
            metric (str, list): name or list of metric names to extract
            result_type (str): can be "score", "pvalue" or "both" (if "both", returns a tuple
                               of score and p-value)

        Returns:
            result: the dictionary with a subset of the results
        """
        if self.scores is None:
            raise RuntimeError(
                "The pipeline was not evaluated. Try running pipeline.evaluate()"
            )

        # parse balance parameter
        keep_balance = True
        if balance == "all":
            balance = list(self.scores.keys())
        elif isinstance(balance, float):
            balance = [balance]
            keep_balance = False

        # parse dataset_size parameter
        keep_dataset_size = True
        if dataset_size == "all":
            dataset_size = list(self.scores[balance[0]].keys())
        elif dataset_size == "max":
            dataset_size = [max(self.scores[balance[0]].keys())]
            keep_dataset_size = False
        elif isinstance(dataset_size, float):
            dataset_size = [dataset_size]
            keep_dataset_size = False

        # parse classifier parameter
        keep_classifier = True
        if classifier == "all":
            classifier = list(self.scores[balance[0]][dataset_size[0]].keys())
        elif isinstance(classifier, str):
            classifier = [classifier]
            keep_classifier = False

        # parse metric parameter
        keep_metric = True
        if metric == "all":
            metric = list(
                self.scores[balance[0]][dataset_size[0]][classifier[0]].keys()
            )
        elif isinstance(metric, str):
            metric = [metric]
            keep_metric = False

        result = {}
        # collect balances
        for bal in self.scores.keys():
            if bal not in balance:
                continue

            result[bal] = {}
            # collect dataset sizes
            for size in self.scores[bal].keys():
                if size not in dataset_size:
                    continue

                result[bal][size] = {}
                # collect classifiers
                for clf in self.scores[bal][size].keys():
                    if clf not in classifier:
                        continue

                    result[bal][size][clf] = {}
                    # collect metrics
                    for met in self.scores[bal][size][clf].keys():
                        if met not in metric:
                            continue

                        # choose result type to keep
                        if result_type == "both":
                            result[bal][size][clf][met] = self.scores[bal][size][clf][
                                met
                            ]
                        elif result_type == "score":
                            result[bal][size][clf][met] = self.scores[bal][size][clf][
                                met
                            ][0]
                        elif result_type == "score_std":
                            result[bal][size][clf][met] = self.scores[bal][size][clf][
                                met
                            ][1]
                        elif result_type == "pvalue":
                            result[bal][size][clf][met] = self.scores[bal][size][clf][
                                met
                            ][2]
                        elif result_type == "perm_score":
                            result[bal][size][clf][met] = self.scores[bal][size][clf][
                                met
                            ][3]

        # remove empty dimensions from the dictionary but keep dimensions where "all" was specified
        result = Pipeline._squeeze_dict(
            result, keep=[keep_balance, keep_dataset_size, keep_classifier, keep_metric]
        )
        return result

    def unbalance_data(
        ratio: float,
        x: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray = None,
        seed: int = 0,
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
        if groups is not None:
            groups_x, groups_y, groups_groups = [], [], []
            for group in np.unique(groups):
                group_idx = np.where(groups == group)[0]
                group_x, group_y, _ = Pipeline.unbalance_data(
                    ratio, x[group_idx], y[group_idx], seed=seed
                )
                groups_x.append(group_x)
                groups_y.append(group_y)
                groups_groups.append([group] * len(group_y))
            return (
                np.concatenate(groups_x),
                np.concatenate(groups_y),
                np.concatenate(groups_groups),
            )

        else:
            unique_y, class_counts = np.unique(y, return_counts=True)
            data_ratio = 1 - min(class_counts) / sum(class_counts)
            max_class = max(class_counts)

            if ratio >= data_ratio:
                class_counts = [max_class, max_class]
                n0, n1 = Pipeline._get_class_numbers(class_counts, ratio)
            else:
                # balances the data by undersampling. Will change nothing
                # if the data is already balanced
                undersample = RandomUnderSampler(
                    sampling_strategy="majority", random_state=seed
                )
                x, y = undersample.fit_resample(x, y)
                unique_y, class_counts = np.unique(y, return_counts=True)
                n0, n1 = Pipeline._get_class_numbers(class_counts, ratio)

            # rebalance data
            idxs = np.concatenate(
                [np.where(y == unique_y[0])[0][:n0], np.where(y == unique_y[1])[0][:n1]]
            )
            return x[idxs], y[idxs], None if groups is None else groups[idxs]

    def _get_class_numbers(class_counts: list, ratio: float) -> tuple:
        # compute expected class sizes according to ratio
        n0 = class_counts[1] / ratio - class_counts[1]
        n1 = class_counts[1]

        # make sure the expected class sizes are realizable with the provided data
        alpha = min(class_counts[0] / n0, class_counts[1] / n1)
        n0 = int(n0 * alpha)
        n1 = int(class_counts[1] * alpha)

        return n0, n1

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
        return Pipeline.CLASSIFIERS[name]

    def _squeeze_dict(data: dict, keep=[]) -> dict:
        """Recursively removes dimensions from a nested dictionary, which only have a single
        entry.

        Args:
            data (dict): the dictionary to reduce
            keep (list): list of bools, indicating if a certain dimension should be kept
                         regardless of the number of entries

        Returns:
            result: the squeezed dictionary
        """
        if isinstance(data, dict):
            if len(data) == 1:
                if not (len(keep) == 0 or keep[0]):
                    # remove this dimension as we only have one entry
                    data = list(data.values())[0]
                    return Pipeline._squeeze_dict(data, keep[1:])

            # recursively squeeze all entries of the current dimension
            for key in data.keys():
                data[key] = Pipeline._squeeze_dict(data[key], keep[1:])
            return data
        # we reached the end of the nesting
        return data


if __name__ == "__main__":
    from pprint import pprint
    from imbalance.data import gaussian_binary

    # generate random data
    x, y, groups = gaussian_binary()

    # initialize the pipeline
    pl = Pipeline(x, y, groups, n_permutations=10, n_init=1, single_balanced_split=True)
    # fit and evaluate the classifiers with different configurations
    pl.evaluate()
    # print classification results
    pprint(pl.scores)
