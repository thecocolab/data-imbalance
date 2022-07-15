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


def compute_and_save_results(x, y, groups, data_path, elec, band_name):
    classifiers = ['lr', 'svm', 'rf', 'lda']
    pl = Pipeline(x, y, groups, classifiers=classifiers, n_permutations=0, n_init=1)
    pl.evaluate()
    save_path = os.path.join(data_path, "results")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(
        os.path.join(save_path, f"MEG_{band_name}_imbalance_{elec}.pckl"),
        "wb",
    ) as fp:
        pickle.dump(copy.deepcopy(pl), fp, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    import argparse

    BANDS = ["delta", "theta", "alpha", "beta", "gamma1", "gamma2", "gamma3"]

    parser = argparse.ArgumentParser(
        "Compute the scores for imbalanced datasets and saves them."
    )
    parser.add_argument(
        "--data-path", type=str, default="/home/arthur/data/camcan", help="The path where the data can be found"
    )
    parser.add_argument(
        "--band", type=str, default="alpha", choices=BANDS, help="The frequency band to use for single feature classification"
    )
    args = parser.parse_args()
    band_name = args.band
    data_path = args.data_path

    train_size = 0.8
    seed = 42
    dattype = "passive"
    eventclf = True
    epoched = True
    s_freq = 500
    band = band_name.index(band_name)

    datasets = create_datasets(
        data_path,
        train_size,
        max_subj=50,
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
    x = x[..., band]  # Only using the alpha frequency band. comment this line for multifeature.
    x = x.numpy()

    for g in np.unique(groups):
        group_idx = np.where(groups == g)[0]
        x[group_idx] = scale(x[group_idx])

    # running all classifiers and 102 sensor locations gives us
    Parallel(n_jobs=-1)(
        delayed(compute_and_save_results)(
            x[:, elec_index], y, groups, data_path, elec_index, band_name
        )
        for elec_index in range(102)
    )
