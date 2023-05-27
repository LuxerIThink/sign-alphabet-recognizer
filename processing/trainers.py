import os
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class ModelTrainer:
    def __init__(self):
        self.train_csv_path = f'{os.path.dirname(__file__)}/test/dataset/train_data.csv'
        self.rm_pattern = r"^Unnamed: 0|world_landmark_\d+\.[xyz]$"
        self.y_col_name = 'letter'

    def run(self, test_csv_path: str):
        x_train, y_train, rev_mapping, label_enc = self.prepare_data(self.train_csv_path)
        x_test, _, _, _ = self.prepare_data(test_csv_path)

    def prepare_data(self, path: str) -> tuple[pd.DataFrame, np.ndarray, dict, LabelEncoder]:
        data = pd.read_csv(path)
        data = self.rm_cols(data)
        x, y = self.split_xy(data)
        x_enc, y_enc, rev_mapping, label_enc = self.enc_labels(x, y)
        return x_enc, y_enc, rev_mapping, label_enc

    def rm_cols(self, data: pd.DataFrame) -> pd.DataFrame:
        cols_to_rm = [col for col in data.columns if re.search(self.rm_pattern, col)]
        data = data.drop(columns=cols_to_rm)
        return data

    def split_xy(self, data) -> tuple[pd.DataFrame, pd.Series]:
        x = data.drop(self.y_col_name, axis=1)
        y = data[self.y_col_name]
        return x, y

    @staticmethod
    def enc_labels(x: pd.DataFrame, y: pd.Series) \
            -> tuple[pd.DataFrame, np.ndarray, dict, LabelEncoder]:
        x = pd.get_dummies(x)
        label_enc = LabelEncoder()
        y_enc = label_enc.fit_transform(y)
        rev_mapping = {i: label for i, label in enumerate(label_enc.classes_)}
        return x, y_enc, rev_mapping, label_enc