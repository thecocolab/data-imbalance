import pickle
import copy
import os
from itertools import product
import numpy as np
from camcan.dataloaders import create_datasets
from torch.utils.data import ConcatDataset, DataLoader
from joblib import Parallel, delayed
from sklearn.preprocessing import scale
from imbalance.pipeline import Pipeline


def compute_and_save_results(x, y, groups, data_path, sensor, band_name):
    classifiers = ["lr", "svm", "rf", "lda"]
    pl = Pipeline(x, y, groups, classifiers=classifiers, n_permutations=100, n_init=10)
    pl.evaluate()
    save_path = os.path.join(data_path, "results")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    band_name = "multifeature" if band_name is None else band_name
    with open(
        os.path.join(save_path, f"MEG_{band_name}_imbalance_{sensor}.pckl"),
        "wb",
    ) as fp:
        pickle.dump(copy.deepcopy(pl), fp, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    import argparse

    BANDS = ["delta", "theta", "alpha", "beta", "gamma1", "gamma2", "gamma3", None]

    parser = argparse.ArgumentParser(
        "Compute the scores for imbalanced datasets and saves them."
    )
    parser.add_argument(
        "--n-sub",
        type=int,
        default=50,
        help="The number of subjects to use for the analysis",
    )
    parser.add_argument(
        "--sensor",
        type=int,
        default=None,
        help="A specific sensor to run analysis on if left to None, will compute all sensors",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="/home/arthur/data/camcan",
        help="The path where the data can be found",
    )
    parser.add_argument(
        "--band",
        type=str,
        default=None,
        choices=BANDS,
        help="The frequency band to use for single feature classification",
    )
    args = parser.parse_args()
    band_name = args.band
    data_path = args.data_path
    sensor = args.sensor
    n_sub = args.n_sub

    train_size = 0.8
    seed = 42
    dattype = "passive"
    eventclf = True
    epoched = True
    band = band_name.index(band_name) if band_name is not None else None

    datasets = create_datasets(
        data_path,
        train_size,
        max_subj=n_sub,
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
    x = x.numpy()
    if band is not None:
        x = x[..., band]
        x = x[..., np.newaxis]

    for i, feature in enumerate(range(x.shape[-1])):
        for e in range(102):
            for g in np.unique(groups):
                group_idx = np.where(groups == g)[0]
                scaled = scale(x[group_idx, e, i])
                x[group_idx, e, i] = scaled

    # running all classifiers and 102 sensor locations gives us
    sensor_list = range(102) if sensor is None else [sensor]

    Parallel(n_jobs=-1)(
        delayed(compute_and_save_results)(
            x[:, sensor_index], y, groups, data_path, sensor_index, band_name
        )
        for sensor_index in sensor_list
    )
