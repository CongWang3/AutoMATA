import shutup  

shutup.please()

from sklearn.preprocessing import LabelEncoder

import torch.optim

import warnings

import pandas as pd

import torch

warnings.simplefilter(action='ignore', category=RuntimeWarning)

_PATH_TRAIN = "../../data/train_example/20240808232043_OtJF37SH_train.txt"

_PATH_TEST = "../../data/train_example/20240808232043_OtJF37SH_test.txt"

_PATH_VAL = "../../data/train_example/20240808232043_OtJF37SH_val.txt"

def load_data(state="train", feature_indices=None):
    if state == "train":
        data = pd.read_csv(_PATH_TRAIN, sep="\t")
    elif state == "test":
        data = pd.read_csv(_PATH_TEST, sep="\t")
    else:
        data = pd.read_csv(_PATH_VAL, sep="\t")
    name = data.iloc[:, 0].values.astype(str)
    feature = data.iloc[:, 1:-1].values.astype(float)
    if feature_indices is not None:
        import numpy as _np
        idx = _np.array(feature_indices, dtype=int).ravel()
        feature = feature[:, idx]
    label = data.iloc[:, -1].values
    encoder = LabelEncoder()
    label = encoder.fit_transform(label.ravel())
    feature, label, name = torch.FloatTensor(feature), torch.LongTensor(label), name
    return feature, label, name
