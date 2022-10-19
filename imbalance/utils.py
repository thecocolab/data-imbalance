import numpy as np
from numpy.random import permutation
from sklearn.base import clone
from joblib import Parallel, delayed
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    balanced_accuracy_score,
    f1_score,
)


def _cross_val(train_index, test_index, estimator, X, y):
    """fit and predict using the given data."""
    clf = clone(estimator)
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf.fit(x_train, y_train)
    # TODO not all classifiers implement predict_proba !!!
    y_pred = clf.predict_proba(x_test)[:, 1]
    return y_pred, y_test


def cross_val_score(estimator, cv, X, y, groups=None, n_jobs=1):
    """Computes all crossval on the chosen estimator, cross-val and dataset.

    it can be used instead of sklearn.model_selection.cross_val_score if you want both roc_auc and
    acc in one go."""
    clf = clone(estimator)
    crossv = clone(cv, safe=False)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_cross_val)(train_index, test_index, clf, X, y)
        for train_index, test_index in crossv.split(X=X, y=y, groups=groups)
    )

    accuracy, auc_list, f1_scores, balanced_accuracies = [], [], [], []
    for test in results:
        y_pred = test[0]
        y_test = test[1]
        try:
            auc_list.append(roc_auc_score(y_test, y_pred))
        except:
            auc_list.append(np.nan)
        y_pred = (y_pred > 0.5).astype(int)
        f1_scores.append(f1_score(y_test, y_pred))
        balanced_accuracies.append(balanced_accuracy_score(y_test, y_pred))
        accuracy.append(accuracy_score(y_test, y_pred))
    return accuracy, auc_list, f1_scores, balanced_accuracies


def compute_pval(score, perm_scores):
    """computes pvalue of an item in a distribution)"""
    n_perm = len(perm_scores)
    pvalue = (np.sum(perm_scores >= score) + 1.0) / (n_perm + 1)
    return pvalue


def permutation_test(estimator, cv, X, y, groups=None, n_perm=0, n_jobs=1):
    """Will do compute permutations aucs and accs."""
    acc_pscores, auc_pscores = [], []
    f1_pscores, bacc_pscores = [], []
    for _ in range(n_perm):
        perm_index = permutation(len(y))
        clf = clone(estimator)
        y_perm = y[perm_index]
        groups_perm = groups[perm_index] if groups is not None else None
        perm_acc, perm_auc, perm_f1_score, perm_bacc = cross_val_score(
            clf, cv, X, y_perm, groups_perm, n_jobs
        )
        acc_pscores.append(np.mean(perm_acc))
        auc_pscores.append(np.mean(perm_auc))
        bacc_pscores.append(np.mean(perm_bacc))
        f1_pscores.append(np.mean(perm_f1_score))

    return acc_pscores, auc_pscores, f1_pscores, bacc_pscores


def classification(estimator, cv, X, y, groups=None, perm=None, n_jobs=1):
    """Do a classification.

    Parameters
    ----------
    estimator : sklearn Estimator
        The estimator that will fit and be tested.
    cv : sklearn CrossValidator
        The cross-validation method that will be used to test the estimator.
    X : array
        The Data, must be of shape (n_samples x n_features).
    y : list or array
        The labels used for training and testing.
    groups : list or array, optional
        The groups for groups based cross-validations
    perm : int, optional
        The number of permutations that will be done to assert significance of the result.
        None means no permutations will be computed
    n_jobs : int, optional (default=1)
        The number of threads to use for the cross-validations. higher means faster. setting
        to -1 will use all available threads - Warning: may slow down computer. Set to -2 to
        keep a thread available for display and other tasks on the computer.

    Returns
    -------
    save : dictionnary
    The dictionnary contains all the information about the classification and the testing :
        acc_score: the mean score across all cross-validations using the
        accuracy scoring method
        auc_score: the mean score across all cross-validations using the
        roc_auc scoring method
        acc: the list of all cross-validations accuracy scores
        auc: the list of all cross-validations roc_auc scores
        if permutation is not None it also countains:
        auc_pvalue: the pvalue using roc_auc as a scoring method
        acc_pvalue: the pvalue using accuracy as a scoring method
        auc_pscores: a list of all permutation auc scores
        acc_pscores: a list of all permutation accuracy scores

    """
    y = np.asarray(y)
    X = np.asarray(X)
    if len(X) != len(y):
        raise ValueError(
            "Dimension mismatch for X and y : {}, {}".format(len(X), len(y))
        )
    if groups is not None:
        try:
            if len(y) != len(groups):
                raise ValueError("dimension mismatch for groups and y")
        except TypeError:
            print(
                "Error in classification: y or",
                "groups is not a list or similar structure",
            )
            exit()
    clf = clone(estimator)
    accuracies, aucs, f1_scores, balanced_accuracies = cross_val_score(
        clf, cv, X, y, groups, n_jobs
    )
    acc_score = np.mean(accuracies)
    auc_score = np.mean(aucs)
    f1_score = np.mean(f1_scores)
    bacc_score = np.mean(balanced_accuracies)
    save = {
        "acc_score": [acc_score],
        "auc_score": [auc_score],
        "f1_score": [f1_score],
        "bacc_score": [bacc_score],
        "acc": accuracies,
        "auc": aucs,
        "f1": f1_scores,
        "bacc": balanced_accuracies,
        "n_splits": cv.get_n_splits(X, y, groups),
    }
    if perm is not None:
        acc_pscores, auc_pscores, f1_pscores, bacc_pscores = permutation_test(
            clf, cv, X, y, groups, perm, n_jobs
        )
        acc_pvalue = compute_pval(acc_score, acc_pscores)
        auc_pvalue = compute_pval(auc_score, auc_pscores)
        f1_pvalue = compute_pval(f1_score, f1_pscores)
        bacc_pvalue = compute_pval(bacc_score, bacc_pscores)

        save.update(
            {
                "auc_pvalue": auc_pvalue,
                "acc_pvalue": acc_pvalue,
                "f1_pvalue": f1_pvalue,
                "bacc_pvalue": bacc_pvalue,
                "auc_pscores": auc_pscores,
                "acc_pscores": acc_pscores,
                "f1_pscores": f1_pscores,
                "bacc_pscores": bacc_pscores,
            }
        )

    return save
