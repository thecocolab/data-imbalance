import warnings
from copy import copy
from typing import Union, Optional
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from imbalance.pipeline import Pipeline

LINESTYLES = ["solid", "dashed", "dotted", "dashdot"]

CLASSIFIERS = {
    "lr": "LogisticRegression",
    "svm": "SVC",
    "lda": "LinearDiscriminantAnalysis",
    "rf": "RandomForestClassifier",
}

METRIC = {
    "roc_auc": "AUC",
    "accuracy": "Accuracy",
    "f1": "F1 score",
}

def metric_balance(
    pl: Pipeline, classifier: str ,p_threshold: float = 0.05, ax: plt.Axes = None, show: bool = True
):
    """Visualizes classification scores of different metrics and classifiers across a range
    of imbalance ratios. If you want to add something to the plot, set show to False and
    use plt.xyz after calling this function.

    Args:
        pl (Pipeline): a pipeline object, which has been evaluated
        classifier (string): the classifier within the pipeline object to plot
        p_threshold (float): threshold of statistical significance
        ax (Axes): if provided, plot in ax instead of creating a new figure
        show (bool): whether the function calls plt.show() or not
    """
    _check_pipeline(pl)

    # extract relevant results from the pipeline
    scores = pl.get(dataset_size="max", result_type="score")
    scores_std = pl.get(dataset_size="max", result_type="score_std")
    pvalues = pl.get(dataset_size="max", result_type="pvalue")
    perm_score = pl.get(dataset_size="max", result_type="perm_score")

    # start the figure
    if ax is None:
        fig, ax = plt.subplots()

    metric_legend, classifier_legend = {}, {}

    clf = CLASSIFIERS[classifier]


    for idx_met, met in enumerate(scores[list(scores.keys())[0]][clf].keys()):
        # get the current scores and p-values as lists
        balances = np.array(list(scores.keys()))
        curr_scores = np.array([scores[bal][clf][met] for bal in balances])
        curr_scores_std = np.array([scores_std[bal][clf][met] for bal in balances])
        curr_pvals = np.array([pvalues[bal][clf][met] for bal in balances])
        curr_perm_score = np.array([perm_score[bal][clf][met] for bal in balances])


        # plot the scores for the current classifier and metric
        line = ax.plot(
            balances,
            curr_scores,
            linestyle="solid",
            color=f"C{idx_met}",
        )[0]

        # fill the std area
        stds = ax.fill_between(
            balances,
            curr_scores - curr_scores_std,
            curr_scores + curr_scores_std,
            color=f"C{idx_met}",
            alpha=0.5,
        )

        # plot the chance level for the current classifier and metric
        chance = ax.plot(
            balances,
            curr_perm_score,
            linestyle="dashed",
            color="darkgrey",
        )[0]

        # visualize statistical significance
        try:
            mask = curr_pvals < p_threshold
            ax.scatter(
                balances[mask],
                curr_scores[mask],
                marker="*",
                s=70,
                color=f"C{idx_met}",
            )
        except TypeError:
            pass

        # add current metric to the legend
        if met not in metric_legend:
            metric_legend[METRIC[met]] = line


    # add annotations
    ax.set_xlabel("data balance")
    ax.set_ylabel("score")
    ax.set_title(classifier)

    ax.legend(
        list(metric_legend.values()) + list(classifier_legend.values()),
        list(metric_legend.keys()) + list(classifier_legend.keys()),
        ncol=2,
    )
    if show:
        plt.show()


def data_distribution(pl: Pipeline, ax: Optional[plt.Axes] = None, show: bool = True):
    """Plots the data distribution of the input data. If x is single-feature, creates a density plot. If
    x is multi-feature, applies TSNE and creates a scatter plot of the two classes.

    Args:
        pl (Pipeline): a pipeline object containing data and labels
        ax (Axes): matplotlib Axes object in which the figures are created. If None, creates a new
                   figure
        show (bool): if True, call plt.show() after creating the figure
    """
    # start the figure
    if ax is None:
        fig, ax = plt.subplots()

    # retrieve data from the pipeline
    x = pl.x
    y = pl.y

    if x.shape[1] > 1:
        # plot single-feature density-plots for separate classes
        _multi_feature_distribution(x, y, ax)
    else:
        # multi-feature TSNE plots
        _single_feature_distribution(x, y, ax)

    if show:
        plt.show()


def _single_feature_distribution(x: np.ndarray, y: np.ndarray, ax: plt.Axes):
    # create density plots
    sns.kdeplot(x[y == 0, 0], ax=ax, label="class 0")
    sns.kdeplot(x[y == 1, 0], ax=ax, label="class 1")
    # add annotations
    ax.set_xlabel("variable")
    ax.legend()


def _multi_feature_distribution(x: np.ndarray, y: np.ndarray, ax: plt.Axes):
    with warnings.catch_warnings():
        # ignore a TSNE FutureWarning about PCA initialization
        warnings.filterwarnings("ignore", category=FutureWarning)
        x = TSNE(learning_rate="auto", init="pca").fit_transform(x)
    # TSNE scatter plot
    ax.scatter(*x[y == 0].T, label="class 0")
    ax.scatter(*x[y == 1].T, label="class 1")
    # add annotations
    ax.set_xlabel("component 0")
    ax.set_ylabel("component 1")
    ax.legend()


def _check_pipeline(pl: Pipeline):
    if pl.scores is None:
        raise RuntimeError(
            "Received an unevaluated pipeline. Try calling pipeline.evaluate()."
        )


if __name__ == "__main__":
    from sklearn.svm import SVC
    from imbalance.data import gaussian_binary

    # generate random data
    x, y, groups = gaussian_binary()

    # run the pipeline
    pl = Pipeline(
        x,
        y,
        groups,
        dataset_balance=np.linspace(0, 1, 32)[1:-1],
        classifiers=["lr", "lda", SVC(kernel="linear")],
        metrics=["roc_auc", "accuracy", "f1", "balanced_accuracy"],
    )
    pl.evaluate()

    # visualize the results
    metric_balance(pl)
