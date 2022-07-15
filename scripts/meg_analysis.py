import warnings
import pickle
import copy
import os
from itertools import product
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
from camcan.dataloaders import create_datasets
from torch.utils.data import ConcatDataset, DataLoader
from scipy.io import savemat
from joblib import Parallel, delayed
from imbalance.pipeline import Pipeline


def compute_and_save_results(x, y, groups, clf, data_path, elec):
    pl = Pipeline(x, y, groups, classifiers=[clf])
    pl.evaluate()
    save_path = os.path.join(data_path, "results")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(
        os.path.join(save_path, f"MEG_PSD_imbalance_{clf}_{elec}.pckl"),
        "wb",
    ) as fp:
        pickle.dump(copy.deepcopy(pl), fp, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        "Compute the scores for imbalanced datasets and saves them."
    )
    parser.add_argument(
        "--elec-index", type=int, default=1, help="The elec index, between 0 and 102."
    )
    args = parser.parse_args()
    elec_index = args.elec_index

    data_path = "/home/arthur/data/camcan"
    train_size = 0.8
    seed = 42
    dattype = "passive"
    eventclf = True
    epoched = True
    s_freq = 500

    datasets = create_datasets(
        data_path,
        train_size,
        max_subj=200,
        seed=seed,
        dattype=dattype,
        eventclf=eventclf,
        epoched=epoched,
        psd=True,
    )
    dataset = ConcatDataset(datasets)
    dataloader = DataLoader(dataset, batch_size=len(dataset))
    x, y, groups = next(iter(dataloader))
    x = x[:, [1, 2]].mean(axis=1)  # averages the psd values for both gradiometers
    x = x[..., 1]  # Only using the alpha frequency band

    classifiers = ["lr", "svm", "lda", "rf"]

    Parallel(n_jobs=-1)(
        delayed(compute_and_save_results)(
            x[:, elec_index], y, groups, clf, data_path, elec_index
        )
        for clf in classifiers
    )
