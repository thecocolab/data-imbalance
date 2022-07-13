# Exploring the robustness of classification metrics on imbalanced datasets
`pipeline.py` contains a generalizable pipeline for comparing classification metrics for different combinations of classifiers, sample size and class balance. It will be extended with helper-functions to access and visualize the results in the future.

## Usage
The `Pipeline` class is the central piece of code for running experiments. It works with on binary classification datasets using scikit-learn classifiers, automatically testing different dataset sizes, class distributions and evaluates classification performance using a range of metrics. By default, 5-fold cross-validation will be used to increase robustness of the results but it is possible to adjust the cross-validation strategy when configuring the pipeline.

The general workflow of using the `Pipeline` class looks as follows:
```python
from pipeline import Pipeline

# load or generate dataset
x, y, groups = get_dataset(...)

# initialize pipeline
pl = Pipeline(x, y, groups,
              classifiers=["lda", "svm"],
              metrics=["acc", "auc", "f1"],
              dataset_balance=[0.1, 0.3, 0.5, 0.7, 0.9])
# fit and evaluate classifiers on dataset configurations
pl.evaluate()

# visualize classification scores
visualize_results(pl)
```
Note that `x` and `y` are the only required arguments to instantiate a `Pipeline`.

## The `Pipeline` API
- **`x` (required):** The dataset used for training and evaluating the classifiers. It can be a 1D array for single-feature classification or a 2D matrix with size `num-samples x num-features`. The array should contain floats suitable for classification, meaning that any kind of preprocessing or normalization should be applied before passing it to the pipeline.
- **`y` (required):** The classification labels corresponding to the samples given by `x`. This should be a one-dimensional integer-array where each unique integer represents a different class. Currently, only binary classification is implemented so `y` should be made up of exactly two unique integers (e.g. 0 and 1).
- **`groups` (optional, default=`None`):** An array with the same shape as `y`, containing group labels for each sample where each group corresponds to a unique integer (e.g. one group per subject). This will be used by the cross-validation strategy to avoid bias due to group effects. If left as `None`, the cross-validation will not take groups into account.
- **`classifiers` (optional, default=`"lr"`):** Can be a string with the name of a classifier, an instantiated scikit-learn classifier or a list, potentially mixing strings and classifier objects. To instantiate the classifier from a string, its name must be present in `Pipeline.CLASSIFIERS.keys()`.
- **`metrics` (optional, default=`["accuracy", "roc_auc"]`):** A list of metric names that should be evaluated. By default, accuracy (`accuracy`) and area-under-curve (`roc_auc`) will be computed for each classifier. See `sklearn.metrics.SCORERS.keys()` for a list of options.
- **`cross_validation` (optional, default=`KFold(n_splits=5)`):** An instantiated scikit-learn cross-validation strategy. By default, 5-fold cross-validation will be used. If `groups` is not `None`, the group-aware variant of k-fold cross-validation is used.
- **`dataset_balance` (optional, default=`[0.1, 0.3, 0.5, 0.7, 0.9]`):** A list of imbalance ratios that should be tested. Ratios should be larger than zero and smaller than one. A value of zero corresponds to a dataset made up fully of class 0, a balance value of 0.5 corresponds to a perfectly balanced dataset (50% of class 0 and 50% of class 1) and a balance of one corresponds to a dataset made up fully of class 1.
- **`dataset_size` (optional, default=`"full"`):** Can be either `"full"` or a list of ratios (`0 < x <= 1`). The value `"full"` is equivalent to `dataset_size=[1.0]`, which evaluates the complete dataset. Values between zero and one artificially limit the number of samples, allowing to check effects related to dataset size.
- **`n_permutations` (optional, default=`0`):** The number of permutations to run for evaluating statistical significance using permutation tests. A value of zero does not run permutation tests.

After calling `pl.evaluate()` on a `Pipeline` object, the nested dictionary `pl.scores` contains all results. The dictionary is structued in the following way:
```python
pl.scores = {
    # dataset balance ratio
    0.3: {
        # dataset size ratio
        0.5: {
            # classifier name
            "clf1": {
                # metric name
                "metric1": (<score-metric1>, <p-value-metric1>),
                "metric2": (<score-metric2>, <p-value-metric2>),
                ...
            },
            "clf2": ...
        },
        0.7: ...
    },
    0.5: ...
}
```

## TODO
- implement helper-functions for easier access to results
    - e.g. something that allows `get_scores(balance="all", size=1.0, classifier="svm", metric="acc")`, which should return a NumPy array with only the selected results
    - accessing subsets of the `scores` dictionary
- a visualization framework which uses the `Pipeline` class
    - line-graph with data balance on the x-axis and different metrics on the y-axis
    - same as above but comparing different classifiers
    - 3D surface plot showing different metrics on the data-balance x data-size plane
- implement a dataset factory for synthetic data
    - random gaussians with binary classification labels
    - variable distance between the means of the two distributions (in other words how much the two classes overlap)
    - single- or multi-feature dataset (potentially correlated features?)
