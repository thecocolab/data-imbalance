import mne
import pickle
from imbalance.data.eeg import get_info
import numpy as np
from imbalance import viz
import matplotlib.pyplot as plt
info = get_info()
chans = get_info().ch_names


clfs = [key for key,val in viz.CLASSIFIERS.items()]
metrics = [key for key,val in viz.METRIC.items()]

clf_ = "svm"
met = "accuracy"

for clf_ in clfs:
    for met in metrics:
        clf = viz.CLASSIFIERS[clf_]
        outpath = f"data/eeg_topoplot_met-{met}_clf-{clf}.png"

        topobalances = [] # channels x balances
        for chan in chans:

            pipeline_path = f'data\eeg_{chan}.pickle'
            pl = pickle.load(open(pipeline_path,"rb"))
            scores = pl.get(dataset_size="max", result_type="score")
            balances = np.array(list(scores.keys()))
            curr_scores = np.array([scores[bal][clf][met] for bal in balances])
            topobalances.append(curr_scores)



        topobalances=np.array(topobalances)
        vmin=0#topobalances.min()
        vmax=1#topobalances.max()
        fig,axes = plt.subplots(nrows=2,ncols=balances.shape[0],sharey=True)
        for i in range(topobalances.shape[-1]):
            im,cn = mne.viz.plot_topomap(topobalances[:,i],info,axes=axes[0][i],cmap='seismic',vmin=vmin,vmax=vmax,show=False)

        cbar=plt.colorbar(mappable=im,ax=axes[1][1],format='%.1f',location='top')

        plt.tight_layout()
        plt.suptitle(outpath)

        fig.savefig(outpath)