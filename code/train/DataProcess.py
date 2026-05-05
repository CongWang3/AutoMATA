import shutup  

shutup.please()

from sklearn.preprocessing import LabelEncoder

import torch.optim

import warnings

import pandas as pd

import numpy as np

import torch

from sklearn.model_selection import train_test_split

from scipy import stats

from sklearn.feature_selection import SelectFpr, chi2

from sklearn.ensemble import RandomForestClassifier

torch.manual_seed(2022)

warnings.simplefilter(action='ignore', category=RuntimeWarning)

PATH_DATA = "../../data/train_example/20240808232043_OtJF37SH_data.txt"

PATH_TEST = "../../data/train_example/20240808232043_OtJF37SH_test.txt"

PATH_VAL = "../../data/train_example/20240808232043_OtJF37SH_val.txt"

def select_pearson(X, y, threshold=1e-5):
    """Feature selection based on the specified statistic."""
    if not isinstance(X, np.ndarray):
        X = np.asarray(X, dtype=float)
    if not isinstance(y, np.ndarray):
        y = np.asarray(y).ravel()
    X = np.nan_to_num(X, nan=np.nanmean(X))
    n_features = X.shape[1]
    indices = []
    for j in range(n_features):
        r, _ = stats.pearsonr(X[:, j], y.astype(float))
        if np.abs(r) > threshold:
            indices.append(j)
    return np.array(indices) if indices else np.arange(n_features)

def select_spearman(X, y, threshold=1e-5):
    if not isinstance(X, np.ndarray):
        X = np.asarray(X, dtype=float)
    if not isinstance(y, np.ndarray):
        y = np.asarray(y).ravel()
    X = np.nan_to_num(X, nan=np.nanmean(X))
    n_features = X.shape[1]
    indices = []
    for j in range(n_features):
        r, _ = stats.spearmanr(X[:, j], y)
        if np.isfinite(r) and np.abs(r) > threshold:
            indices.append(j)
    return np.array(indices) if indices else np.arange(n_features)

def select_chi2(X, y, p_value=0.05, n_bins=10):
    if not isinstance(X, np.ndarray):
        X = np.asarray(X, dtype=float)
    if not isinstance(y, np.ndarray):
        y = np.asarray(y).ravel()
    X_bin = np.zeros_like(X, dtype=int)
    for j in range(X.shape[1]):
        col = X[:, j]
        try:
            bins = np.percentile(col, np.linspace(0, 100, n_bins + 1)[1:-1])
            X_bin[:, j] = np.digitize(col, bins)
        except Exception:
            X_bin[:, j] = 0
    X_bin = np.maximum(X_bin, 0)
    try:
        selector = SelectFpr(score_func=chi2, alpha=p_value)
        selector.fit(X_bin, y)
        idx = np.where(selector.get_support())[0]
        if idx.size == 0:
            chi2_vals, _ = chi2(X_bin, y)
            idx = np.argsort(chi2_vals)[-1:]
        return idx
    except Exception:
        return np.arange(X.shape[1])

def select_rf_importance(X, y, threshold=0.0, n_estimators=100, random_state=42):
    if not isinstance(X, np.ndarray):
        X = np.asarray(X, dtype=float)
    if not isinstance(y, np.ndarray):
        y = np.asarray(y).ravel()
    col_means = np.nanmean(X, axis=0)
    X = np.where(np.isnan(X), col_means, X)
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X, y)
    importances = clf.feature_importances_
    keep = importances > threshold
    if not np.any(keep):
        return np.arange(X.shape[1])
    return np.where(keep)[0]

def apply_feature_selection(X, y, feature_method):
    method_upper = (feature_method or "").strip().upper()
    if method_upper in ("PCC", "PEARSON"):
        return select_pearson(X, y)
    if method_upper in ("SPEARMAN",):
        return select_spearman(X, y)
    if method_upper in ("CHI2", "CHI_SQUARED", "CHISQUARE"):
        return select_chi2(X, y)
    if method_upper in ("RF", "RANDOM_FOREST", "RANDOMFOREST"):
        return select_rf_importance(X, y)
    raise ValueError("Unknown feature selection method: {}".format(feature_method))

def load_data(state="train", feature_method=None, feature_indices=None):
    """Load train/validate/test data; support feature selection and return tensors."""
    if state == "train":
        data = pd.read_csv(PATH_DATA, sep="\t")
    elif state == "test":
        data = pd.read_csv(PATH_TEST, sep="\t")
    else:
        data = pd.read_csv(PATH_VAL, sep="\t")
    data = data.iloc[:, 1:]
    data = data.dropna()
    label_col = data.iloc[:, -1]
    numeric_labels = pd.to_numeric(label_col, errors="coerce")
    valid_mask = numeric_labels.notna()
    if not valid_mask.all():
        invalid_count = (~valid_mask).sum()
        print(f"警告: 过滤掉 {invalid_count} 行无效标签数据")
        data = data[valid_mask]
    feature = data.iloc[:, :-1].values.astype(float)
    label = data.iloc[:, -1].values.astype(int)
    encoder = LabelEncoder()
    label = encoder.fit_transform(label.ravel())
    indices_out = None
    if state == "train" and feature_method:
        indices_out = apply_feature_selection(feature, label, feature_method)
        feature = feature[:, indices_out]
    elif feature_indices is not None:
        feature = feature[:, feature_indices]
    feature = torch.FloatTensor(feature)
    label = torch.LongTensor(label)
    if state == "train" and feature_method:
        return feature, label, indices_out
    return feature, label, None

def process(ratio="8:1:1"):
    data = pd.read_csv(PATH_DATA, sep="\t")
    ratio_str = ratio.split(":")
    ratio_num = list(map(int, ratio_str))  # [8, 1, 1]
    train_ratio = ratio_num[0] / sum(ratio_num)
    test_ratio = ratio_num[2] / sum(ratio_num[1:])
    train_data, res_data = train_test_split(
        data, test_size=1 - train_ratio, stratify=data[["Label"]], shuffle=True, random_state=42
    )
    val_data, test_data = train_test_split(
        res_data, test_size=test_ratio, stratify=res_data[["Label"]], shuffle=True, random_state=42
    )
    train_data.to_csv(PATH_DATA, sep="\t", index=False)
    test_data.to_csv(PATH_TEST, sep="\t", index=False)
    val_data.to_csv(PATH_VAL, sep="\t", index=False)
