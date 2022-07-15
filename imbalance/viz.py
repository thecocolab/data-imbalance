import warnings
from copy import copy
from typing import Union, Optional, List
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
    "balanced_accuracy": "Balanced Accuracy",
    "f1": "F1 score",
}

def metric_balance(
    pl: Pipeline,
    classifier: str,
    p_threshold: float = 0.01,
    ax: plt.Axes = None,
    show: bool = True,
    show_leg: bool = True,
    ignore_metrics : Union[list,str] = [],
    enforce_ylim: bool = True,
    color_offset: int = 0
) -> List[plt.Axes]:
    """Visualizes classification scores of different metrics and classifiers across a range
    of imbalance ratios. If you want to add something to the plot, set show to False and
    use plt.xyz after calling this function.

    Args:
        pl (Pipeline): a pipeline object, which has been evaluated
        classifier (string): the classifier within the pipeline object to plot
        p_threshold (float): threshold of statistical significance
        ax (Axes): if provided, plot in ax instead of creating a new figure
        show (bool): whether the function calls plt.show() or not
        show_leg (bool): whether to add a legend or not
        ignore_metrics (list, str): name(s) of metrics to ignore in the figure
        enforce_ylim (bool): if True, set the y-limits to (0, 1)
        color_offset (int): offset added to the color index for each classifier
    """
    _check_pipeline(pl)

    if not isinstance(ignore_metrics, list):
        ignore_metrics = [ignore_metrics]

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

    line_plots = []
    for idx_met, met in enumerate(scores[list(scores.keys())[0]][clf].keys()):
        # potentially ignore some metrics
        if met in ignore_metrics:
            continue

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
            color=f"C{idx_met + color_offset}",
        )[0]
        line_plots.append(line)

        # fill the std area
        stds = ax.fill_between(
            balances,
            curr_scores - curr_scores_std,
            curr_scores + curr_scores_std,
            color=f"C{idx_met + color_offset}",
            alpha=0.5,
        )

        # plot the chance level for the current classifier and metric
        chance = ax.plot(
            balances,
            curr_perm_score,
            linestyle="dotted",
            color=f"C{idx_met + color_offset}",
        )[0]

        # visualize statistical significance
        try:
            mask = curr_pvals < p_threshold
            ax.scatter(
                balances[mask],
                curr_scores[mask],
                marker="*",
                s=70,
                color=f"C{idx_met + color_offset}",
            )
        except TypeError:
            pass

        # add current metric to the legend
        if met not in metric_legend:
            metric_legend[METRIC[met]] = line
    # trick to get the chance level dotted line in the legend
    metric_legend["Chance level"] = ax.plot([],[],linestyle="dotted",color="black")[0]

    # add annotations
    ax.set_xlabel("data balance")
    ax.set_ylabel("score")
    ax.set_title(classifier)
    if enforce_ylim:
        ax.set_ylim(0, 1)

    if show_leg is True:
        ax.legend(
            list(metric_legend.values()) + list(classifier_legend.values()),
            list(metric_legend.keys()) + list(classifier_legend.keys()),
            ncol=1,
        )
    if show:
        plt.show()
    return line_plots

def plot_different_n(
    pl: Pipeline, classifier: str, metric: str , ax: plt.Axes = None, show: bool = True, show_leg: bool = True,
):
    """Visualizes classification scores of different sizes of datasets.

    Args:
        pl (Pipeline): a pipeline object, which has been evaluated
        classifier (string): the classifier within the pipeline object to plot
        ax (Axes): if provided, plot in ax instead of creating a new figure
        show (bool): whether the function calls plt.show() or not
    """
    _check_pipeline(pl)

    # extract relevant results from the pipeline
    scores = pl.get(dataset_size="all", result_type="score")
    scores_std = pl.get(dataset_size="all", result_type="score_std")
    dataset_size = pl.dataset_size
    # start the figure
    if ax is None:
        fig, ax = plt.subplots()

    metric_legend = {}

    clf = CLASSIFIERS[classifier]

    for idx_size, size in enumerate(dataset_size):
        # get the current scores and p-values as lists
        balances = np.array(list(scores.keys()))
        curr_scores = np.array([scores[bal][size][clf][metric] for bal in balances])
        curr_scores_std = np.array([scores_std[bal][size][clf][metric] for bal in balances])

        # plot the scores for the current classifier and metric
        line = ax.plot(
            balances,
            curr_scores,
            linestyle="solid",
            color=f"C{idx_size}",
            alpha=0.5,
        )[0]

        # fill the std area
        stds = ax.fill_between(
            balances,
            curr_scores - curr_scores_std,
            curr_scores + curr_scores_std,
            color=f"C{idx_size}",
            alpha=0.2,
        )

        # add current metric to the legend
        if size not in metric_legend:
            metric_legend[size] = line

    # add annotations
    ax.set_xlabel("data balance")
    ax.set_ylabel("score")
    ax.set_title(classifier + '__' + metric)

    if show_leg is True:
        ax.legend(
            list(metric_legend.values()),
            list(metric_legend.keys()),
            ncol=1,
        )
    if show:
        plt.show()

def plot_different_cvs(
    pls: dict, classifier: str, metric: str , ax: plt.Axes = None, show: bool = True, show_leg: bool = True,
):
    """Visualizes classification scores of different cross-validation schemes.

    Args:
        pls (dict of Pipeline objects): a dict of Pipeline objects called with
        different cross-validators, which have all been evaluated.
        classifier (string): the classifier within the pipeline object to plot.
        ax (Axes): if provided, plot in ax instead of creating a new figure.
        show (bool): whether the function calls plt.show() or not.
    """
    for cv_name in pls:
        _check_pipeline(pls[cv_name])

    # extract relevant results from the pipeline
    scores_list = []
    scores_std_list = []
    cross_val_names = []
    for cv_name in pls.keys():
        scores_list.append(pls[cv_name].get(dataset_size="all", result_type="score"))
        scores_std_list.append(pls[cv_name].get(dataset_size="all", result_type="score_std"))
        cross_val_names.append(cv_name)
    # start the figure
    if ax is None:
        fig, ax = plt.subplots()

    metric_legend = {}

    clf = CLASSIFIERS[classifier]

    for idx_cv, cv_name in enumerate(pls):
        # get the current scores and p-values as lists
        # TODO : find a better way to remove the size level
        balances = np.array(list(scores_list[idx_cv].keys()))
        curr_scores = np.array([scores_list[idx_cv][bal][1.0][clf][metric] for bal in balances])
        curr_scores_std = np.array([scores_std_list[idx_cv][bal][1.0][clf][metric] for bal in balances])

        # plot the scores for the current classifier and metric
        line = ax.plot(
            balances,
            curr_scores,
            linestyle="solid",
            color=f"C{idx_cv}",
            alpha=0.5,
        )[0]

        # fill the std area
        stds = ax.fill_between(
            balances,
            curr_scores - curr_scores_std,
            curr_scores + curr_scores_std,
            color=f"C{idx_cv}",
            alpha=0.2,
        )

        # add current metric to the legend
        if cv_name not in metric_legend:
            metric_legend[cv_name] = line

    # add annotations
    ax.set_xlabel("data balance")
    ax.set_ylabel("score")
    ax.set_title(classifier + '__' + metric)

    if show_leg is True:
        ax.legend(
            list(metric_legend.values()),
            list(metric_legend.keys()),
            ncol=1,
        )
    if show:
        plt.show()


def data_distribution(pl: Pipeline, ax: Optional[plt.Axes] = None, show: bool = True, show_leg: bool = True, class_names: list = None):
    """Plots the data distribution of the input data. If x is single-feature, creates a density plot. If
    x is multi-feature, applies TSNE and creates a scatter plot of the two classes.

    Args:
        pl (Pipeline): a pipeline object containing data and labels
        ax (Axes): matplotlib Axes object in which the figures are created. If None, creates a new
                   figure
        show (bool): if True, call plt.show() after creating the figure
        class_names (list): list of names of the classes (indexed by the integers corresponding to a
                            class)
    """
    # start the figure
    if ax is None:
        fig, ax = plt.subplots()

    # retrieve data from the pipeline
    x = pl.x
    y = pl.y

    if x.shape[1] > 1:
        # plot single-feature density-plots for separate classes
        _multi_feature_distribution(x, y, ax, show_leg, class_names=class_names)
    else:
        # multi-feature TSNE plots
        _single_feature_distribution(x, y, ax, show_leg, class_names=class_names)

    if show:
        plt.show()


def _single_feature_distribution(x: np.ndarray, y: np.ndarray, ax: plt.Axes, show_leg: bool = True, class_names: list = None):
    if class_names is None:
        class_names = ["class 0", "class 1"]
    # create density plots
    sns.kdeplot(x[y == 0, 0], shade=True, color="C0", ax=ax, label=class_names[0])
    sns.kdeplot(x[y == 1, 0], shade=True, color="C1", ax=ax, label=class_names[1])
    # add annotations
    ax.set_xlabel("variable")
    if show_leg:
        ax.legend()


def _multi_feature_distribution(x: np.ndarray, y: np.ndarray, ax: plt.Axes, show_leg: bool = True, class_names: list = None):
    if class_names is None:
        class_names = ["class 0", "class 1"]
    with warnings.catch_warnings():
        # ignore a TSNE FutureWarning about PCA initialization
        warnings.filterwarnings("ignore", category=FutureWarning)
        x = TSNE(learning_rate="auto", init="pca").fit_transform(x)
    # TSNE scatter plot
    ax.scatter(*x[y == 0].T, label=class_names[0])
    ax.scatter(*x[y == 1].T, label=class_names[1])
    # add annotations
    ax.set_xlabel("component 0")
    ax.set_ylabel("component 1")
    if show_leg:
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
        dataset_balance=np.linspace(0.1, 0.9, 25),
    )
    pl.evaluate()

    # visualize the results
    metric_balance(pl, "lr")
