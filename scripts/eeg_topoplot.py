import mne
import pickle
from imbalance.data.eeg import get_info
import numpy as np
from imbalance import viz
import matplotlib.pyplot as plt

info = get_info()
chans = get_info().ch_names


clfs = ["svm"]  # [key for key,val in viz.CLASSIFIERS.items()]
metrics = [key for key, val in viz.METRIC.items()]

clf_ = "svm"
met = "accuracy"

for clf_ in clfs:
    for met in metrics:
        clf = viz.CLASSIFIERS[clf_]
        outpath = f"data/topos_3balances_allCLFS/eeg_topoplot_met-{met}_clf-{clf}.png"
        met_ = viz.METRIC[met]

        topobalances = []  # channels x balances
        for chan in chans:

            pipeline_path = f"data/topos_3balances_allCLFS/eeg_{chan}.pickle"
            pl = pickle.load(open(pipeline_path, "rb"))
            scores = pl.get(dataset_size="max", result_type="score")
            balances = np.array(list(scores.keys()))
            curr_scores = np.array([scores[bal][clf][met] for bal in balances])
            topobalances.append(curr_scores)

        topobalances = np.array(topobalances)
        vmin = 0  # topobalances.min()
        vmax = 1  # topobalances.max()
        fig, axes = plt.subplots(nrows=1, ncols=balances.shape[0], sharey=True)
        for i in range(topobalances.shape[-1]):
            axes[i].set_title(f"R = {balances[i]}")
            im, cn = mne.viz.plot_topomap(
                topobalances[:, i],
                info,
                contours=0,
                extrapolate="head",
                axes=axes[i],
                cmap="jet",
                vmin=vmin,
                vmax=vmax,
                show=False,
            )
        cbar_ticks = np.linspace(vmin, vmax, 5)
        cbar = plt.colorbar(
            mappable=im, ax=axes, ticks=cbar_ticks, format="%.1f", location="bottom"
        )
        cbar.set_label("Score")
        if clf == "SVC":
            plt.suptitle(f"{met_} for SVM")
        else:
            plt.suptitle(f"{met_} for {clf}")

        fig.savefig(outpath)
