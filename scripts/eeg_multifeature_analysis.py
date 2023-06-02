"""
EEG Test 1
Multi-feature experiment with Stratified Group K-Fold
"""

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold

from imbalance.data import eegbci
from imbalance.data.eeg import get_info
from imbalance.pipeline import Pipeline
from imbalance import viz
import umap

###############################################################################
# Configuration of the test
###############################################################################

cross_validation = StratifiedGroupKFold(n_splits=5)
cross_validation_label = "Stratified-Group"
n_features = "multi"
dataset_balance = np.linspace(0.1, 0.9, 25)
classifiers = [
    "lda",
    "svm",
    LogisticRegression(max_iter=1000),
    RandomForestClassifier(n_estimators=25),
]
n_permutations = 100
n_init = 10

###############################################################################
# Region of Interest
###############################################################################
# Any electrode that starts with
#   P -> Parietal
#   O -> Occipital
roi = lambda x: x[0] in ["P", "O"]

###############################################################################
#%% Path for results
###############################################################################

pipeline_path = f"data/eeg_roi_{n_features}_{cross_validation_label}.pickle"
features_path = f"data/eeg_features_{n_features}.npy"

###############################################################################
#%% Generate Features from the EEG signals
###############################################################################

if not os.path.isfile(features_path):
    print("Generating features!")
    x, y, groups = eegbci("data", roi=roi, n_features=n_features)
    np.save(features_path, dict(x=x, y=y, groups=groups))
else:
    # Reuse pre-computed features
    print("Loading features from a previous run")
    features = np.load(features_path, allow_pickle=True).item()
    x, y, groups = features["x"], features["y"], features["groups"]

###############################################################################
#%% Run Pipeline on the features
###############################################################################

if not os.path.isfile(pipeline_path):
    pl = Pipeline(
        x,
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
    with open(pipeline_path, "wb") as handle:
        pickle.dump(pl, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    # Reuse a pre-computed pipeline
    print("Loading pipeline from a previous run")
    pl = pickle.load(open(pipeline_path, "rb"))

###############################################################################
#%% Plot Results
###############################################################################

# Feature Distribution

print("Plotting feature distribution")
viz.data_distribution(pl, show=False)

fig = plt.gcf()
fig.savefig(pipeline_path.replace(".pickle", f"_dist.png"))

import umap.plot
mapper = umap.UMAP().fit(pl.x)#,pl.y)

foo=lambda x: 'Eyes Closed' if x==0 else 'Eyes Open'

color_key = {'Eyes Closed':'b','Eyes Open':(241/255, 90/255, 34/255)}
umap.plot.points(mapper,labels=np.array([foo(x) for x in pl.y],dtype=object),color_key=color_key)
fig = plt.gcf()
fig.savefig(pipeline_path.replace(".pickle", f"_umap.png"))
fig.savefig(pipeline_path.replace(".pickle", f"_umap.pdf"))

# Balance vs Score Curves for each Classifier
print("Plotting Balance vs Score Curves")
clfs = [key for key, val in viz.CLASSIFIERS.items()]
for clf in clfs:
    viz.metric_balance(pl, clf, show=False)
    fig = plt.gcf()
    fig.savefig(pipeline_path.replace(".pickle", f"_{clf}.png"))

print("EEG multi-feature test done!")


###

import copy
from imbalance.pipeline import Pipeline
from imbalance.viz import metric_balance, data_distribution, plot_different_cvs, plot_different_n
from imbalance.data import eegbci, gaussian_binary
from joblib import Parallel, delayed
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    StratifiedGroupKFold,
)
from sklearn.svm import SVC
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import string
from copy import deepcopy
import pickle
import warnings
warnings.simplefilter("ignore", RuntimeWarning)
plt.style.use('seaborn-dark')

###

# fpath = "../imbalance/data/eeg_roi_multi_Stratified-Group.pickle"
# with open(fpath, "rb") as f:
#     pl = pickle.load(f)

def task_panel(pl: Pipeline, figtitle: str=[], class_names=[]):
    # visualize the result
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    #fig.suptitle(figtitle, fontsize=25)
    classifiers=["lr", "lda", "svm", "rf"]
    show_leg_metric=True
    for ax_idx,ax in enumerate(axes.flat):
        if ax_idx != 5:
            ax.text(-0.1, 1.05, string.ascii_lowercase[ax_idx]+")", transform=ax.transAxes, 
                    size=22, weight='bold')
        if ax_idx == 0:
            #data_distribution(pl, ax=ax, show=False, class_names=class_names)
            foo=lambda x: class_names[0] if x==0 else class_names[1]
            mapper = umap.UMAP().fit(pl.x)#,pl.y)
            color_key = {class_names[0]:'b',class_names[1]:(241/255, 90/255, 34/255)}
            umap.plot.points(mapper,labels=np.array([foo(x) for x in pl.y],dtype=object),color_key=color_key,ax=ax)

        elif ax_idx < 5:
            metric_balance(pl, ax=ax, show=False, classifier=classifiers[ax_idx-1], p_threshold=0, show_leg=show_leg_metric)
            show_leg_metric=False
        elif ax_idx == 5:
            ax.axis('off')
            #plot_different_n(pl_nsamples, ax=ax, show=False, classifier="svm", show_leg=True, metric="accuracy")
    plt.show()

figtitle = "EEG data multi"
task_panel(pl, figtitle, ['Eyes closed', 'Eyes open'])

plt.close('all')

from imbalance.data.eeg import get_info
import mne

# fpath = "../imbalance/data/eeg_roi_multi_Stratified-Group.pickle"
# with open(fpath, "rb") as f:
#     pl = pickle.load(f)
#tval = np.load("../imbalance/data/gamma1_tvalues.npy")    

figtitle = "EEG : Eyes Closed VS Open"
class_names = ['Eyes Closed', 'Eyes Open']
# visualize the result
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle(figtitle, fontsize=28, y=1.05, fontweight="bold")
classifiers=["lr", "lda", "svm", "rf"]

#data_distribution(pl, ax=axes[0,0], show=False, class_names=class_names)
foo=lambda x: class_names[0] if x==0 else class_names[1]
mapper = umap.UMAP().fit(pl.x)#,pl.y)
color_key = {class_names[0]:'b',class_names[1]:(241/255, 90/255, 34/255)}
umap.plot.points(mapper,labels=np.array([foo(x) for x in pl.y],dtype=object),color_key=color_key,ax=axes[0,0])
print(axes[0,0].texts[0])
del axes[0,0].texts[0]
#axes[0,0].set_xlim(-2.5,5)
axes[1,0].axis('off')

info = get_info()
roi = [x for x in info['ch_names'] if x[0] in ['P','O']]
sphere=(0, 0.015, 0.01, 0.115)

metric_balance(pl, ax=axes[0,1], show=False, classifier="lr", p_threshold=0, show_leg=True)
metric_balance(pl, ax=axes[0,2], show=False, classifier="lda", p_threshold=0, show_leg=False)
metric_balance(pl, ax=axes[1,1], show=False, classifier="svm", p_threshold=0, show_leg=False)
metric_balance(pl, ax=axes[1,2], show=False, classifier="rf", p_threshold=0, show_leg=False)

for ax_idx,ax in enumerate(axes.T.flat):
    ax.text(-0.1, 1.075, string.ascii_lowercase[ax_idx]+")", transform=ax.transAxes, 
            size=26, weight='bold')

# plot topomap and highlight selected sensors
crange = np.linspace(1, 0, 100)
cm = ListedColormap(np.stack((np.ones(len(crange)), crange, crange, np.ones(len(crange))), axis=1))
mask = np.array(list(map(lambda x: x.startswith(("O", "P")), info.ch_names)))
vals = np.zeros(len(info.ch_names))
vals[mask] = 1
with sns.axes_style("white"):
    #mne.viz.plot_sensors(info, axes=axes[1,0], show=False, sphere=sphere)
    mne.viz.plot_topomap(vals, info, axes=axes[1,0], image_interp="sinc", res=62, show=False,
                         sphere=(0, 0.018, 0.01, 0.1), show_names=False, contours=False, extrapolate="local",
                         vmin=0, vmax=4, cmap=cm, mask=np.ones(vals.shape),
                         mask_params=dict(marker="o", markersize=6, markerfacecolor="0"))
    
plt.savefig("Figure4.pdf",bbox_inches="tight")#,pad_inches=1)
#plt.show()
