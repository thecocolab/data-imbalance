import os
import mne
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Sequence, List, Tuple, Union, Optional, Dict, Any
from collections.abc import Callable
from copy import deepcopy

def eegbci(
    datapath : str = 'data',
    epoch_duration : Union[float,int] = 5,
    band : Tuple[float,float]= (8.,12.),
    scale : bool=True,
    roi : Union[List[str],Callable]=None,
    n_features : str = 'single',
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns the eyes-close (1) and eyes-open (0) EEG dataset.

    Args:
        datapath (str): path where the eeg data files will be downloaded
        epoch_duration (float,int): duration in seconds of the epochs
        band (tuple[float,float]): frequency range desired
        scale (bool) : Whether to apply (True) a zero-mean unit-variance scaler or not (False)
        roi (list[str],Callable) : List of channel names or a function that selects (returns True) based on the channel name.
        n_features (str) : 'single' to apply mean along channels or 'multi' to keep the channels.
    Returns:
        X : power features array of shape (samples, channel)
        Y : label array of shape (samples,)
        G : group/subject array of shape (samples,)
    """

    # Make sure path exist
    os.makedirs(datapath,exist_ok=True)

    # Download the dataset
    filenames = {}
    sub_list = list(range(1,110))
    task_list = (0,1) #('close','open')
    for sub in sub_list:
        filenames[sub] = {}
        filenames[sub][task_list[0]]=mne.datasets.eegbci.load_data(sub, [2],path=datapath,update_path=False, verbose=False)[0] #2 Baseline, eyes closed
        filenames[sub][task_list[1]]=mne.datasets.eegbci.load_data(sub, [1],path=datapath,update_path=False, verbose=False)[0] #1 Baseline, eyes open

    X = [] # Data Vector
    Y = [] # Labels
    G = [] # Groups (Subjects)
    chan_names = []

    for sub in sub_list:
        for task in task_list:
            raw = mne.io.read_raw_edf(filenames[sub][task], preload=False, verbose=False)

            # Segment into epochs
            epochs = mne.make_fixed_length_epochs(raw, duration=epoch_duration, preload=True, verbose=False)
            chan_names.append(deepcopy(epochs.info["ch_names"]))
            # Calculate Spectrum
            psds,freqs=mne.time_frequency.psd_array_multitaper(epochs.get_data(),epochs.info['sfreq'], verbose=False)

            # Slice Spectrum to the selected band
            band_idxs = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]

            # Obtain the average power inside the selected band
            power = np.mean(psds[:,:,band_idxs],axis=-1)

            # Repeat the label for each epoch in the current data --> shape = (epochs,)
            Y.append(np.array([task]*epochs.get_data().shape[0]))

            # power is of shape (epochs,channels)
            X.append(power)

            # Repeat the subject for each epochs in the current data --> shape = (epochs,)
            G.append(np.array([sub]*epochs.get_data().shape[0]))

    # For each of the collected combinations of (subject,task)
    # concatenate along the epochs
    X = np.concatenate(X,axis=0) # axis = 0 --> epochs, 1 would be the channels
    Y = np.concatenate(Y)
    G = np.concatenate(G)

    # Assert channels and their order are the same
    chan_names = np.array(chan_names)
    for chan in range(chan_names.shape[1]):
        assert len(set(chan_names[:,chan]))==1
    chans = list(chan_names[0,:])

    wrong_subs=[]
    for sub in set(G):
        if not set(Y[np.where(G==sub)[0]]) == {0,1}:
            print(f"sub-{sub} is missing one class")
            wrong_subs.append(sub)
    assert wrong_subs == []

    # assert X.shape[0]==109*12*2 # subjects x epochs x tasks
    # assert np.sum(Y)==109*12*2/2 # half of them must be class 1, the other class 0


    # Single-Feature Classification Based on ROI
    if roi is not None:
        if isinstance(roi,list):
            idxs = [chans.index(c) for c in roi]
        else:
            idxs = [chans.index(c) for c in chans if roi(c)]
        X = X[:,idxs]
    if n_features == 'single':
        X = np.mean(X,axis=-1,keepdims=True)

    # Scale if desired
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    return X,Y,G

def get_info(datapath = 'data'):
    os.makedirs(datapath,exist_ok=True)
    sub = 1
    filepath = mne.datasets.eegbci.load_data(sub, [2],path=datapath,update_path=False)[0]
    raw = mne.io.read_raw_edf(filepath, preload=False)
    mapping = {x:x.replace('.',' ').rstrip().upper() for x in raw.ch_names}
    mne.rename_channels(raw.info,mapping)
    montage = mne.channels.make_standard_montage('standard_1005')
    montage.rename_channels({x:x.upper() for x in montage.ch_names})
    raw.set_montage(montage)
    return raw.info.copy()

if __name__ == '__main__':
    info = get_info()
    roi = [x for x in info['ch_names'] if x[0] in ['P','O']]
    sphere=(0, 0, 0.01, 0.12)
    mne.viz.plot_sensors(info,sphere=sphere,pointsize=25,linewidth=0,block=True,show_names=roi)


