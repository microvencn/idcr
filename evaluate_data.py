import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
from sklearn.preprocessing import QuantileTransformer, LabelEncoder


def eval_law_school(train_set, test_set):
    scaler = QuantileTransformer()
    y_scaler = QuantileTransformer()
    model = MLPRegressor()
    y_train = train_set[["ZFYA"]].copy()
    y_test = test_set[["ZFYA"]].copy()
    y_train = y_scaler.fit_transform(y_train)
    train = train_set.drop("ZFYA", axis=1)
    test = test_set.drop("ZFYA", axis=1)
    train = scaler.fit_transform(train)
    test = scaler.transform(test)
    model.fit(train, y_train.reshape(-1))
    y_pred = model.predict(test).reshape(-1, 1)
    y_pred = y_scaler.inverse_transform(y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    print("MSE: %.3f \n" % metrics.mean_squared_error(y_test, y_pred))
    print("RMSE: %.3f \n" % rmse)
    print("MAE: %.3f \n" % metrics.mean_absolute_error(y_test, y_pred))
    blackX = pd.read_csv("data/real_world/law_school/Black.csv")
    asianX = pd.read_csv("data/real_world/law_school/Asian.csv")
    whiteX = pd.read_csv("data/real_world/law_school/White.csv")
    blackX.drop("ZFYA", axis=1, inplace=True)
    whiteX.drop("ZFYA", axis=1, inplace=True)
    asianX.drop("ZFYA", axis=1, inplace=True)
    blackX = scaler.transform(blackX)
    whiteX = scaler.transform(whiteX)
    asianX = scaler.transform(asianX)
    black_pred = model.predict(blackX).reshape(-1, 1)
    black_pred = y_scaler.inverse_transform(black_pred)
    asian_pred = model.predict(asianX).reshape(-1, 1)
    asian_pred = y_scaler.inverse_transform(asian_pred)
    white_pred = model.predict(whiteX).reshape(-1, 1)
    white_pred = y_scaler.inverse_transform(white_pred)
    te1 = np.abs(white_pred - black_pred).mean()
    print("TE_BLACK_WHITE: %.3f \n" % te1)
    te2 = np.abs(white_pred - asian_pred).mean()
    print("TE_ASIAN_WHITE: %.3f \n" % te2)

    test_set = test_set.copy()
    ZYFA_median = test_set["ZFYA"].median()
    test_set['sensitive'] = (test_set['race'] == 7).astype(int)
    test_set['label_bin'] = (test_set['ZFYA'] >= ZYFA_median).astype(int)
    test_set['pred_bin'] = (y_pred >= ZYFA_median).astype(int)
    test_set['pred'] = y_pred
    grouped_pred = test_set.groupby('sensitive')['pred_bin'].mean()
    print("[Debug] Group Pred: ", grouped_pred.values.tolist())
    print("[Debug] Pred Median: ", np.median(y_pred))
    print("[Debug] Pred Max: ", y_pred.max())
    print("[Debug] Pred Min: ", y_pred.min())
    dp_diff = abs(grouped_pred.loc[1] - grouped_pred.loc[0])
    grouped_pred = test_set.groupby('sensitive')['pred'].mean()
    print("Pred difference", abs(grouped_pred.loc[1] - grouped_pred.loc[0]))
    fp_mask = test_set['label_bin'] == 0
    fp_grouped = test_set[fp_mask].groupby('sensitive')['pred_bin'].mean()
    print("[Debug] Group Pred: ", fp_grouped.values.tolist())
    fpr_1 = fp_grouped.get(1, 0.0)
    fpr_0 = fp_grouped.get(0, 0.0)
    fpr_diff = abs(fpr_1 - fpr_0)

    print(f"Demographic Parity Difference: {dp_diff:.4f}")
    print(f"False Positive Rate Difference: {fpr_diff:.4f}")
    return rmse, te1, te2, dp_diff, fpr_diff


def eval_synthetic_dataset(train_dataset, test_dataset):

    categorical_trans = LabelEncoder()
    continuous_trans = QuantileTransformer()
    y_trans = QuantileTransformer()
    label_col = ["Y"]
    continuous = ["X1", "X2"]
    categorical = ["A"]
    train = train_dataset.copy()
    # Preprocessing training set
    train[categorical] = categorical_trans.fit_transform(train[categorical])
    train[continuous] = continuous_trans.fit_transform(train[continuous])
    train[label_col] = y_trans.fit_transform(train[label_col])
    def preprocessing(dataset):
        dataset[categorical] = categorical_trans.transform(dataset[categorical])
        dataset[continuous] = continuous_trans.transform(dataset[continuous])
    y_train = train[label_col].copy()
    train.drop(label_col, axis=1, inplace=True)
    # Preprocessing testing set
    test = test_dataset.copy()
    preprocessing(test)
    y_test = test[label_col].copy()
    test.drop(label_col, axis=1, inplace=True)
    model = MLPRegressor()
    model.fit(train, y_train)
    y_pred = model.predict(test)
    y_pred = y_trans.inverse_transform(y_pred.reshape(-1, 1))

    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    print("MSE: %.3f \n" % metrics.mean_squared_error(y_test, y_pred))
    print("RMSE: %.3f \n" % rmse)
    print("MAE: %.3f \n" % metrics.mean_absolute_error(y_test, y_pred))

    data_0 = pd.read_csv("data/real_world/synthetic/0.csv")
    data_49 = pd.read_csv("data/real_world/synthetic/49.csv")
    data_99 = pd.read_csv("data/real_world/synthetic/99.csv")
    preprocessing(data_0)
    preprocessing(data_49)
    preprocessing(data_99)
    data_0.drop(label_col, axis=1, inplace=True)
    data_99.drop(label_col, axis=1, inplace=True)
    data_49.drop(label_col, axis=1, inplace=True)
    pred_0 = model.predict(data_0).reshape(-1, 1)
    pred_0 = y_trans.inverse_transform(pred_0)
    pred_49 = model.predict(data_49).reshape(-1, 1)
    pred_49 = y_trans.inverse_transform(pred_49)
    pred_99 = model.predict(data_99).reshape(-1, 1)
    pred_99 = y_trans.inverse_transform(pred_99)
    te1 = np.abs(pred_49 - pred_0).mean()
    te2 = np.abs(pred_99 - pred_49).mean()
    print("TE_0_49: %.3f \n" % te1)
    print("TE_49_99: %.3f \n" % te2)

    test_set = test_dataset.copy()
    test_set[categorical] = categorical_trans.transform(test_set[categorical])
    Y_median = test_set["Y"].median()
    test_set['sensitive'] = (test_set['A'] >= 50).astype(int)
    test_set['label_bin'] = (test_set['Y'] >= Y_median).astype(int)
    test_set['pred_bin'] = (y_pred >= Y_median).astype(int)
    grouped_pred = test_set.groupby('sensitive')['pred_bin'].mean()
    dp_diff = abs(grouped_pred.loc[1] - grouped_pred.loc[0])
    fp_mask = test_set['label_bin'] == 0
    fp_grouped = test_set[fp_mask].groupby('sensitive')['pred_bin'].mean()
    fpr_1 = fp_grouped.get(1, 0.0)
    fpr_0 = fp_grouped.get(0, 0.0)
    fpr_diff = abs(fpr_1 - fpr_0)

    print(f"Demographic Parity Difference: {dp_diff:.4f}")
    print(f"False Positive Rate Difference: {fpr_diff:.4f}")
    return rmse, te1, te2, dp_diff, fpr_diff