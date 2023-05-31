import os
from typing import Tuple

import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.image import index_img
from nilearn.input_data import NiftiMasker


def fmri_haxby(
    datapath: str = "data", subject: int = 1, return_nilearn: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the voxel data from the Haxby dataset (subject 2) for face vs house.

    Args:
        datapath (str): path where the eeg data files will be downloaded
        subject (int): subject number to load
        return_nilearn (bool): return the nilearn masker and dataset objects
    Returns:
        X : power features array of shape (samples, channel)
        Y : label array of shape (samples,)
    """

    # Make sure path exist
    os.makedirs(datapath, exist_ok=True)

    # Load the data
    haxby_dataset = datasets.fetch_haxby(subjects=[subject], data_dir=datapath)
    func_img = haxby_dataset.func[0]

    # load behavioral data
    behavioral = pd.read_csv(haxby_dataset.session_target[0], sep=" ")
    conditions = behavioral["labels"]

    condition_mask = behavioral["labels"].isin(["face", "house"])
    conditions = conditions[condition_mask]
    func_img = index_img(func_img, condition_mask)

    session_label = behavioral["chunks"][condition_mask]

    mask_filename = haxby_dataset.mask
    masker = NiftiMasker(mask_img=mask_filename, standardize=True)
    masked_data = masker.fit_transform(func_img)

    if return_nilearn:
        return masked_data, np.unique(conditions.values, return_inverse=True)[1], masker, haxby_dataset
    else:
        return masked_data, np.unique(conditions.values, return_inverse=True)[1]


if __name__ == "__main__":
    x, y = fmri_haxby("data")
