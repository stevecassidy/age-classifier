import os
import numpy as np
import mne
import hcp

def spectral_features(raw: mne.io.Raw, max_freq: int = 100, n_fft: int = 48, filter=True) -> np.array: 
    """Compute flattened spectral features given a cropped raw recording.

    If the `filter` argument is True data is low-pass filtered to max_freq (default 100Hz)
    and downsampled to 2*max_freq before features are
    computed. 

    Computes a n_fft (default 48) point FFT. Takes the log to compute the final features which
    are then concatenated into a one dimensional feature vector.

    Returns a N x 24 np.array where N is the number of channels"""

    if filter:
        raw.filter(None, max_freq, h_trans_bandwidth=0.5, filter_length='10s', phase='zero-double', fir_design='firwin2')
        raw.resample(2*max_freq, npad="auto")

    psds, _freqs = mne.time_frequency.psd_welch(raw, fmin=1, n_fft=n_fft)

    #return np.ravel(np.log(psds))
    return np.log(psds)

def spectral_epochs(label: str, raw: mne.io.Raw, epoch_size: int, max_freq: int = 100, n_fft: int = 48, filter=True) -> np.array: 
    """Read raw data and split into epochs of a given size (s), compute features
    over each one
    label: subject identifier
    epoch_size: duration of epochs in seconds
    data_folder: location of source data
    max_freq: max frequency for FFT (default 100)
    n_fft: FFT size (default 48)

    Returns: labels, features
    labels: a list of labels with the format <subject>-<run_index>-<N>
    features: a list of np.arrays one per epoch containing the features
    """

    features = []
    labels = []

    events = mne.make_fixed_length_events(raw, id=1, duration=epoch_size)
    epochs = mne.Epochs(raw, events, tmin=0., tmax=epoch_size, baseline=None,
                        detrend=1, decim=8, preload=True)

    
    for N in range(len(epochs)):
        features.append(spectral_features(epochs[N], max_freq=max_freq, n_fft=n_fft, filter=filter))
        labels.append("{}-{}".format(label, N))
        print('.', end='', flush=True)
    print('|', end='', flush=True)        
    return labels, features


def read_hcp(subject: str, data_folder: str, run_index: int) -> mne.io.Raw:
    """
    Read a data file from the HCP dataset, return a Raw instance
    """

    raw = hcp.read_raw(subject=subject, data_type='rest', hcp_path=data_folder, run_index=run_index)

    return raw


def read_mous(subject: str, data_folder: str) -> mne.io.Raw:
    """
    Read a data file from the MOUS dataset, return a Raw instance
    """

    try:
        raw_path = os.path.join(data_folder, 'Resting/{}/sub-{}_task-rest_meg.ds'.format(subject, subject))

        raw = mne.io.read_raw_ctf(raw_path, preload=True)

        picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=True, exclude='bads') 
        raw.pick(picks)
        return raw

    except:
        return None

def read_camcan(subject: str, data_folder: str) -> mne.io.Raw:
    """
    Read a data file from the MOUS dataset, return a Raw instance
    """

    try:
        raw_path = os.path.join(data_folder, '{}/rest/rest_raw.fif'.format(subject))

        print("READING", raw_path)

        raw = mne.io.read_raw_fif(raw_path, preload=True)

        picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=True, exclude='bads') 
        raw.pick(picks)
        return raw

    except:
        return None



if __name__=='__main__': 

    mne.set_log_level('ERROR')

    hcp_folder = os.path.join(os.getcwd(), 'data/raw')

    raw = read_hcp('107473', hcp_folder, 0)
    labels, features = spectral_epochs('107473', raw, 60)

    for i in range(len(labels)):
        print(labels[i], features[i].shape)

    mous_folder = os.path.join(os.getcwd(), 'data/Donders_MEG/')

    raw = read_mous('A2002', mous_folder)
    labels, features = spectral_epochs('107473', raw, 60, max_freq=74)

    for i in range(len(labels)):
        print(labels[i], features[i].shape)

    