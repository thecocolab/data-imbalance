from imbalance.data import eegbci
from imbalance.data.eeg import get_info
from imbalance.pipeline import Pipeline
from imbalance import viz
import pickle
from sklearn.linear_model import LogisticRegression
import os
from sklearn.preprocessing import StandardScaler
import numpy as np

pipeline_path='data/eeg_%%.pickle'
features_path ="data/eeg_features_unscaled.npy"
chans = get_info().ch_names

rois = [[x] for x in chans]
rois_names = chans

if not os.path.isfile(features_path):
    # load or generate dataset
    x, y, groups = eegbci('data',roi=None,scale=False,n_features='multi')
    np.save(features_path,dict(x=x, y=y, groups=groups))
else:
    features = np.load(features_path,allow_pickle=True).item()
    x, y, groups = features["x"] , features["y"] , features["groups"]


# Tests : all chans, roi, chan by chan

for roi,roi_name in zip(rois,rois_names):

    this_pipe = pipeline_path.replace('%%',roi_name)
    idxs = [chans.index(c) for c in roi]
    curr_x =  x.copy()[:,idxs]
    scaler = StandardScaler()
    curr_x = scaler.fit_transform(curr_x)
    print(roi_name,idxs,curr_x.shape)

    if not os.path.isfile(this_pipe):
        pl = Pipeline(
            curr_x,
            y,
            groups,
            dataset_balance = [0.1,0.3,0.5,0.7,0.9],
            classifiers = ["svm"],#['rf',"lda","svm",LogisticRegression(max_iter=1000)],
            n_permutations = 0,
            n_init = 10,
        )
        # fit and evaluate classifiers on dataset configurations
        pl.evaluate()

        # Store data (serialize)
        with open(this_pipe, 'wb') as handle:
            pickle.dump(pl, handle, protocol=pickle.HIGHEST_PROTOCOL)
