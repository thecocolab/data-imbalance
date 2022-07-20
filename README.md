# Class imbalance should not throw you off balance: Choosing classifiers and performance metrics for brain decoding with imbalanced data

Machine learning (ML) is becoming a standard tool in neuroscience and neuroimaging research. Yet, because it is such a powerful tool, the appropriate application of ML requires a sound understanding of its subtleties and limitations. In particular, applying ML to datasets with imbalanced classes, which are very common in neuroscience, can have severe consequences if not adequately addressed. With the neuroscience machine-learning user in mind, this technical note provides a didactic overview of the class imbalance problem and illustrates its impact through systematic manipulation of class imbalance ratios in both simulated data, and real electroencephalography (EEG) and magnetoencephalography (MEG) brain data. Our results illustrate how in highly imbalanced data, the commonly used Accuracy (Acc) metric yields misleadingly high performances by preferentially predicting the majority class, while other evaluations metrics (e.g. Balanced Accuracy (BAcc) and the Area Under the Curve (AUC) of the Receiver Operating Characteristic (ROC)) may still provide reliable performance evaluations. In terms of classifiers and cross-validation schemes, our data highlights the higher robustness of Random Forest (RF) and Stratified K-Fold cross-validation, compared to the other approaches tested. Critically, for neuroscience ML applications that seek to minimize overall classification error (not preferentially that of a single class), we recommend the routine use of BAcc, rather than the simple and more commonly used Acc metric. Importantly, we provide a best practices list of recommendations for dealing with imbalanced data, and open-source code to allow the neuroscience community to replicate our observations and further explore the best practices in handling imbalanced data.

---

This repository contains the code to the analysis performed in the accompanying paper.

`pipeline.py` contains a generalizable pipeline for comparing classification metrics for different combinations of classifiers, sample size and class balance. It will be extended with helper-functions to access and visualize the results in the future.

## Installation
First, create and activate a new Python environment:
```bash
conda create -n imbalance python==3.8 -y
conda activate imbalance
```
Afterwards, clone the repository and install its dependencies as follows:
```bash
git clone git@github.com:thecocolab/data-imbalance.git
cd data-imbalance
pip install -e .
```

### Reduced Installation
The `-e` flag allows you to make changes to the repository without the need to reinstall. If you are only planning to use the package and don't want to make changes you can skip cloning and install via
```bash
pip install git+https://github.com/thecocolab/data-imbalance.git
```

## Usage
The `Pipeline` class is the central piece of code for running experiments. It works with on binary classification datasets using scikit-learn classifiers, automatically testing different dataset sizes, class distributions and evaluates classification performance using a range of metrics. By default, 5-fold cross-validation will be used to increase robustness of the results but it is possible to adjust the cross-validation strategy when configuring the pipeline.

The general workflow of using the `Pipeline` class looks as follows:
```python
from imbalance.pipeline import Pipeline
from imbalance import viz

# load or generate dataset
x, y, groups = get_dataset(...)

# initialize pipeline
pl = Pipeline(x, y, groups,
              classifiers=["lda", "svm"],
              dataset_balance=[0.1, 0.3, 0.5, 0.7, 0.9])
# fit and evaluate classifiers on dataset configurations
pl.evaluate()

# visualize classification scores
viz.metric_balance(pl)
```
Note that `x` and `y` are the only required arguments to instantiate a `Pipeline`.

## The `Pipeline` API
- **`x` (required):** The dataset used for training and evaluating the classifiers. It can be a 1D array for single-feature classification or a 2D matrix with size `num-samples x num-features`. The array should contain floats suitable for classification, meaning that any kind of preprocessing or normalization should be applied before passing it to the pipeline.
- **`y` (required):** The classification labels corresponding to the samples given by `x`. This should be a one-dimensional integer-array where each unique integer represents a different class. Currently, only binary classification is implemented so `y` should be made up of exactly two unique integers (e.g. 0 and 1).
- **`groups` (optional, default=`None`):** An array with the same shape as `y`, containing group labels for each sample where each group corresponds to a unique integer (e.g. one group per subject). This will be used by the cross-validation strategy to avoid bias due to group effects. If left as `None`, the cross-validation will not take groups into account.
- **`classifiers` (optional, default=`"lr"`):** Can be a string with the name of a classifier, an instantiated scikit-learn classifier or a list, potentially mixing strings and classifier objects. To instantiate the classifier from a string, its name must be present in `Pipeline.CLASSIFIERS.keys()`.
- **`cross_validation` (optional, default=`StratifiedKFold(n_splits=5)`):** An instantiated scikit-learn cross-validation strategy. By default, 5-fold cross-validation will be used. If `groups` is not `None`, the group-aware variant of k-fold cross-validation is used.
- **`dataset_balance` (optional, default=`np.linspace(0.1,0.9,25)`):** A list of imbalance ratios that should be tested. Ratios should be larger than zero and smaller than one. A value of zero corresponds to a dataset made up fully of class 0, a balance value of 0.5 corresponds to a perfectly balanced dataset (50% of class 0 and 50% of class 1) and a balance of one corresponds to a dataset made up fully of class 1.
- **`dataset_size` (optional, default=`"full"`):** Can be either `"full"` or a list of ratios (`0 < x <= 1`). The value `"full"` is equivalent to `dataset_size=[1.0]`, which evaluates the complete dataset. Values between zero and one artificially limit the number of samples, allowing to check effects related to dataset size.
- **`n_permutations` (optional, default=`100`):** The number of permutations to run for evaluating statistical significance using permutation tests. A value of zero does not run permutation tests. Note that permutations will only be computed for the first initialization (if n_init > 1).
- **`random_seed` (optional, default=`42`):** The random seed to use.
- **`n_init` (optional, default=`10`):** Number of re-initializations of the whole pipeline.

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
                "metric1": (mean_score-metric1, std_score-metric1, p-value-metric1, permutation_score_metric1),
                "metric2": (mean_score-metric2, std_score-metric2, p-value-metric2, permutation_score_metric2),
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
    - single- or multi-feature dataset (potentially correlated features?)
