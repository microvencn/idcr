import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit, KFold


def kfold(df: pd.DataFrame, k, stratified = False, sensitive=None):
    if stratified:
        assert sensitive is not None
        kf = StratifiedShuffleSplit(n_splits=k, random_state=42)
        folds = kf.split(df, df[sensitive])
    else:
        kf = KFold(n_splits=k, random_state=1, shuffle=True)
        folds = kf.split(df)
    train_result = []
    test_result = []
    for train_index, test_index in folds:
        train_data = df.iloc[train_index, :].copy()
        train_data.index = np.arange(len(train_data))
        test_data = df.iloc[test_index, :].copy()
        test_data.index = np.arange(len(test_data))
        train_result.append(train_data)
        test_result.append(test_data)
    return train_result, test_result