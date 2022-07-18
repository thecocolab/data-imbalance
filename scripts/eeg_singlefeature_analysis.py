"""
EEG Test 2
Single-feature experiment across all locations with Stratified Group K-Fold
"""
import os
import numpy as np
import pickle
import string

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib.pyplot as plt
import mne


from imbalance.data import eegbci
from imbalance.data.eeg import get_info
from imbalance.pipeline import Pipeline
from imbalance import viz

###############################################################################
# Configuration of the test
###############################################################################

dataset_balance = [0.1, 0.5, 0.9]
classifiers = ["svm"]
n_permutations = 0
n_init = 10
cross_validation = StratifiedGroupKFold(n_splits=5)

###############################################################################
# Region of Interest
###############################################################################

# We will test each channel as a different region-of-interest
chans = get_info().ch_names
rois = [[x] for x in chans]
rois_names = chans

###############################################################################
#%% Path for results
###############################################################################
# We will replace the %% with each channel later
pipeline_path = "data/single_feature_eeg/eeg_%%.pickle"
features_path = "data/eeg_features_unscaled.npy"

###############################################################################
#%% Generate Features from the EEG signals
###############################################################################

if not os.path.isfile(features_path):
    # We will produce the data for all channels and then slice to save time
    # We will skip normalization and do it later in a per-channel basis
    x, y, groups = eegbci("data", roi=None, scale=False, n_features="multi")
    np.save(features_path, dict(x=x, y=y, groups=groups))
else:
    # Reuse pre-computed features
    print("Loading features from a previous run")
    features = np.load(features_path, allow_pickle=True).item()
    x, y, groups = features["x"], features["y"], features["groups"]

###############################################################################
#%% Run Pipeline on the features for all channels
###############################################################################

for roi, roi_name in zip(rois, rois_names):

    this_pipe = pipeline_path.replace("%%", roi_name)

    # Slice to the current channel
    idxs = [chans.index(c) for c in roi]
    curr_x = x.copy()[:, idxs]

    # Normalize within-channel
    scaler = StandardScaler()
    curr_x = scaler.fit_transform(curr_x)

    print(roi_name, idxs, curr_x.shape)

    if not os.path.isfile(this_pipe):
        pl = Pipeline(
            curr_x,
            y,
            groups,
            dataset_balance=dataset_balance,
            classifiers=classifiers,
            n_permutations=n_permutations,
            n_init=n_init,
            cross_validation=cross_validation,
        )

        pl.evaluate()

        # Store pipeline for later re-use
        with open(this_pipe, "wb") as handle:
            pickle.dump(pl, handle, protocol=pickle.HIGHEST_PROTOCOL)

###############################################################################
#%% Plot Results
###############################################################################

# Some matplolib styling
plt.style.use("seaborn-dark")
plt.rcParams.update({"axes.edgecolor": "black"})
x_offsets = [1, 0, 0, 1]
y_offsets = [0, 0, 1, 1]

# We will do a plot for each (classifier,metric) pair
metrics = [key for key, val in viz.METRIC.items()]

# We will need MNE's info object to know the location of the sensors
info = get_info()


for clf in classifiers:
    fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(10, 6))

    for ax_idx, ax in enumerate(axes[:, ::3].T.flat):
        ax.text(
            -0.1,
            1.3,
            string.ascii_lowercase[ax_idx + 6] + ")",
            transform=ax.transAxes,
            size=13,
            weight="bold",
        )

    clf_key = viz.CLASSIFIERS[clf]
    outpath = pipeline_path.replace("%%.pickle", f"topomap_clf-{clf_key}.pdf")

    for met_idx, met in enumerate(metrics):
        met_ = viz.METRIC[met]

        # Topobalances will be a channels x balances array of scores values
        # Which we will plot into a topomap
        topobalances = []

        for chan in chans:
            # Load pipeline of given channel
            # and extract scores
            pl = pickle.load(open(pipeline_path.replace("%%", chan), "rb"))
            scores = pl.get(dataset_size="max", result_type="score")
            balances = np.array(list(scores.keys()))
            curr_scores = np.array([scores[bal][clf_key][met] for bal in balances])
            topobalances.append(curr_scores)

        topobalances = np.array(topobalances)

        # Plot into topomaps
        vmin = 0
        vmax = 1

        # Do a topomap for each balance tested
        for i in range(topobalances.shape[-1]):
            title = f"IR = {balances[i]}"
            if i == 1:
                title = f"{met_}\n" + title
            axes[y_offsets[met_idx], x_offsets[met_idx] * 3 + i].set_title(title)
            im, cn = mne.viz.plot_topomap(
                topobalances[:, i],
                info,
                sensors=False,
                contours=False,
                axes=axes[y_offsets[met_idx], x_offsets[met_idx] * 3 + i],
                cmap="jet",
                vmin=vmin,
                vmax=vmax,
                show=False,
            )

        # Adjust colorbar
        cbar_ticks = np.linspace(vmin, vmax, 5)
        cbar = plt.colorbar(
            mappable=im,
            ax=axes[
                y_offsets[met_idx], x_offsets[met_idx] * 3 : x_offsets[met_idx] * 3 + 3
            ],
            ticks=cbar_ticks,
            format="%.1f",
            location="bottom",
        )
        cbar.set_label("Score")

        box = cbar.ax.get_position()
        box.x0 = box.x0 + 0.02
        box.x1 = box.x1 - 0.02
        box.y0 = box.y0 - 0.1
        box.y1 = box.y1 - 0.1
        cbar.ax.set_position(box)

    plt.subplots_adjust()
    plt.savefig(outpath, bbox_inches="tight")

print("EEG single-feature channel-wise test done!")
