import string
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

x_offsets = [1, 0, 0, 1]
y_offsets = [0, 0, 1, 1]

for clf_ in clfs:
    fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(10, 6))

    for ax_idx, ax in enumerate(axes[:, ::3].T.flat):
        ax.text(
            -0.1,
            1.3,
            string.ascii_lowercase[ax_idx] + ")",
            transform=ax.transAxes,
            size=13,
            weight="bold",
        )

    for met_idx, met in enumerate(metrics):
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
        vmin = 0
        vmax = 1
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
    plt.savefig("notebooks/Figure5.pdf", bbox_inches="tight")
