from copy import copy
from typing import Union
import numpy as np
from matplotlib import pyplot as plt
from imbalance.pipeline import Pipeline

LINESTYLES = ["solid", "dashed", "dotted", "dashdot"]


def metric_balance(
    pl: Pipeline, p_threshold: float = 0.05, ax: plt.Axes = None, show: bool = True
):
    """Visualizes classification scores of different metrics and classifiers across a range
    of imbalance ratios. If you want to add something to the plot, set show to False and
    use plt.xyz after calling this function.

    Args:
        pl (Pipeline): a pipeline object, which has been evaluated
        p_threshold (float): threshold of statistical significance
        ax (Axes): if provided, plot in ax instead of creating a new figure
        show (bool): whether the function calls plt.show() or not
    """
    _check_pipeline(pl)

    # extract relevant results from the pipeline
    scores = pl.get(dataset_size="max", result_type="score")
    pvalues = pl.get(dataset_size="max", result_type="pvalue")

    # start the figure
    if ax is None:
        fig, ax = plt.subplots()

    metric_legend, classifier_legend = {}, {}
    for idx_clf, clf in enumerate(scores[list(scores.keys())[0]].keys()):
        for idx_met, met in enumerate(scores[list(scores.keys())[0]][clf].keys()):
            # get the current scores and p-values as lists
            balances = np.array(list(scores.keys()))
            curr_scores = np.array([scores[bal][clf][met] for bal in balances])
            curr_pvals = np.array([pvalues[bal][clf][met] for bal in balances])

            # plot the scores for the current classifier and metric
            line = ax.plot(
                balances,
                curr_scores,
                linestyle=LINESTYLES[idx_clf],
                color=f"C{idx_met}",
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
                metric_legend[met] = line
            # add current classifier to the legend
            if clf not in classifier_legend:
                l = copy(line)
                l.set_color("black")
                classifier_legend[clf] = l

    # add annotations
    ax.set_xlabel("data balance")
    ax.set_ylabel("score")

    ax.legend(
        list(metric_legend.values()) + list(classifier_legend.values()),
        list(metric_legend.keys()) + list(classifier_legend.keys()),
        ncol=2,
    )
    if show:
        plt.show()


def _check_pipeline(pl: Pipeline):
    if pl.scores is None:
        raise RuntimeError(
            "Received an unevaluated pipeline. Try calling pipeline.evaluate()."
        )


if __name__ == "__main__":
    from sklearn.svm import SVC

    # generate random data
    n = 1000
    x = np.concatenate(
        [np.random.normal(0, size=n // 2), np.random.normal(2, size=n // 2)]
    ).reshape(-1, 1)
    y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(int)
    groups = np.concatenate([np.arange(n // 2), np.arange(n // 2)]).astype(int)

    # run the pipeline
    pl = Pipeline(
        x,
        y,
        groups,
        dataset_balance=np.linspace(0, 1, 100)[1:-1],
        classifiers=["lr", "lda", SVC(kernel="linear")],
        metrics=["roc_auc", "accuracy", "f1", "balanced_accuracy"],
    )
    pl.evaluate()

    # visualize the results
    metric_balance(pl)
