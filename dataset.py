from __future__ import print_function, division
import torch
from torch.utils.data import Dataset
from collections import namedtuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder


class CSVDataset(Dataset):
    """CSV Dataset"""

    def __init__(self, csv_file, reorder=None):
        self.data = pd.read_csv(csv_file)
        keep_col = self.data.keys().tolist()
        keep_col = ['Unnamed' not in item for item in keep_col]
        self.data = self.data.iloc[:, keep_col]

        if reorder is not None:
            cols = [reorder[i] for i in range(len(reorder))]
            self.data = self.data[cols]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.data.iloc[idx, :]
        data = np.array(data)
        data = torch.Tensor(data)

        return data


class NumpyDataset(Dataset):
    """CSV Dataset"""

    def __init__(self, data_matrix):
        self.data = data_matrix

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = torch.FloatTensor(self.data[idx, :])

        return data


SpanInfo = namedtuple("SpanInfo", ["dim", "activation_fn"])
ColumnTransformInfo = namedtuple(
    "ColumnTransformInfo", ["column_name", "column_type",
                            "transform", "transform_aux",
                            "output_info", "output_dimensions"])

CATEGORICAL = "categorical"
CONTINUOUS = "continuous"
ORDINAL = "ordinal"

class QuantileTrans(object):
    """Data Transformer.
    Model continuous columns with a BayesianGMM and normalized to a scalar [0, 1] and a vector.
    Discrete columns are encoded using a scikit-learn OneHotEncoder.
    """

    def __init__(self):
        """Create a data transformer.
            Continuous columns: standradize as: (value - mean) / Use Tanh activation
            Discrete columns: encode the category into integer. Use Relu activation

        """

    def _fit_continuous(self, column_name, raw_column_data):
        """Train Bayesian GMM for continuous column."""
        ss = QuantileTransformer(n_quantiles=2000)
        ss.fit(raw_column_data.reshape(-1, 1))

        return ColumnTransformInfo(
            column_name=column_name, column_type="continuous", transform=ss,
            transform_aux=None,
            output_info=[],
            output_dimensions=1)

    def _fit_discrete(self, column_name, raw_column_data):
        """Fit one hot encoder for discrete column."""
        ohe = OneHotEncoder()
        ohe.fit(raw_column_data.reshape((-1, 1)))
        num_categories = np.unique(raw_column_data).shape[0]

        return ColumnTransformInfo(
            column_name=column_name, column_type="discrete", transform=ohe,
            transform_aux=None,
            output_info=[SpanInfo(num_categories, 'softmax')],
            output_dimensions=num_categories)

    def fit(self, raw_data, discrete_columns=tuple()):
        """Fit GMM for continuous columns and One hot encoder for discrete columns.
        This step also counts the #columns in matrix data, and span information.
        """
        self.output_info_list = []
        self.output_dimensions = 0

        if not isinstance(raw_data, pd.DataFrame):
            self.dataframe = False
            raw_data = pd.DataFrame(raw_data)
        else:
            self.dataframe = True

        self._column_raw_dtypes = raw_data.infer_objects().dtypes  # infer_object: better way to examine data dtype

        self._column_transform_info_list = []
        for column_name in raw_data.columns:
            raw_column_data = raw_data[column_name].values
            if column_name in discrete_columns:
                column_transform_info = self._fit_discrete(
                    column_name, raw_column_data)
            else:
                column_transform_info = self._fit_continuous(
                    column_name, raw_column_data)

            self.output_info_list.append(column_transform_info.output_info)
            self.output_dimensions += column_transform_info.output_dimensions
            self._column_transform_info_list.append(column_transform_info)

    def _transform_continuous(self, column_transform_info, raw_column_data):
        ss = column_transform_info.transform
        normalized_value = ss.transform(raw_column_data.reshape(-1, 1))
        return [normalized_value]

    def _transform_discrete(self, column_transform_info, raw_column_data):
        ohe = column_transform_info.transform
        return [ohe.transform(raw_column_data).A] # 这里是9 但 categories 是 4

    def transform(self, raw_data):
        """Take raw data and output a matrix data."""
        if not isinstance(raw_data, pd.DataFrame):
            raw_data = pd.DataFrame(raw_data)

        column_data_list = []
        out_dimensions = []
        for column_transform_info in self._column_transform_info_list:
            column_data = raw_data[[column_transform_info.column_name]].values
            if column_transform_info.column_type == "continuous":
                column_data_list += self._transform_continuous(
                    column_transform_info, column_data)
            else:
                assert column_transform_info.column_type == "discrete"
                column_data_list += self._transform_discrete(
                    column_transform_info, column_data)
            out_dimensions.append(column_transform_info.output_dimensions)

        return np.concatenate(column_data_list, axis=1).astype(float), out_dimensions

    def _inverse_transform_continuous(self, column_transform_info, column_data):
        ss = column_transform_info.transform
        column = ss.inverse_transform(column_data.reshape(-1, 1))

        return column

    def _inverse_transform_discrete(self, column_transform_info, column_data):
        ohe: OneHotEncoder = column_transform_info.transform
        return ohe.inverse_transform(column_data)

    def inverse_transform(self, data):
        st = 0
        recovered_column_data_list = []
        column_names = []
        for column_transform_info in self._column_transform_info_list:
            dim = column_transform_info.output_dimensions
            column_data = data[:, st:st + dim]

            if column_transform_info.column_type == 'continuous':
                recovered_column_data = self._inverse_transform_continuous(
                    column_transform_info, column_data)
            else:
                assert column_transform_info.column_type == 'discrete'
                recovered_column_data = self._inverse_transform_discrete(
                    column_transform_info, column_data)

            recovered_column_data_list.append(recovered_column_data)
            column_names.append(column_transform_info.column_name)
            st += dim

        recovered_data = np.column_stack(recovered_column_data_list)
        recovered_data = (pd.DataFrame(recovered_data, columns=column_names)
                          .astype(self._column_raw_dtypes))
        if not self.dataframe:
            recovered_data = recovered_data.values

        return recovered_data

    def convert_column_name_value_to_id(self, column_name, value):
        discrete_counter = 0
        column_id = 0
        for column_transform_info in self._column_transform_info_list:
            if column_transform_info.column_name == column_name:
                break
            if column_transform_info.column_type == "discrete":
                discrete_counter += 1
            column_id += 1

        return {
            "discrete_column_id": discrete_counter,
            "column_id": column_id,
            "value_id": np.argmax(column_transform_info.transform.transform(np.array([value]))[0])
        }