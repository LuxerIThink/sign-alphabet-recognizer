import os
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


class ModelTrainer:
    def __init__(self):
        self.train_csv_path = f'{os.path.dirname(__file__)}/test/dataset/train_data.csv'
        self.rm_pattern = r"^Unnamed: 0|world_landmark_\d+\.[xyz]$"
        self.y_col_name = 'letter'
        self.rand_state = 42

    def run(self, test_csv_path: str):
        x_train, y_train, rev_mapping_train, label_enc_train = self.prepare_data(self.train_csv_path)
        x_test, y_test, rev_mapping_test, label_enc_test = self.prepare_data(test_csv_path)
        clf = self.train_clf(x_train, y_train)
        y_pred_labels, y_test_decoded = self.predict(
            clf,
            x_test,
            y_test,
            rev_mapping_test,
            label_enc_test,
        )
        print(y_test_decoded)
        return y_pred_labels, y_test_decoded

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

    def train_clf(self, x_train: pd.DataFrame, y_train: np.ndarray) -> DecisionTreeClassifier:
        clf = DecisionTreeClassifier(criterion="entropy", random_state=self.rand_state)
        clf.fit(x_train, y_train)
        return clf

    @staticmethod
    def predict(model, x_test, y_test, rev_mapping, label_enc):
        y_pred = model.predict(x_test)
        y_pred_labels = [
            rev_mapping[int(round(label))]
            if int(round(label)) in rev_mapping
            else "Unknown"
            for label in y_pred
        ]
        y_test_dec = label_enc.inverse_transform(y_test)
        return y_pred_labels, y_test_dec