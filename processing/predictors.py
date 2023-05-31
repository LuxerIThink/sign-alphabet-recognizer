import os
import re
import pandas as pd
import numpy as np
import joblib


class SignAlphabetPredictor:
    def __init__(self):
        self.rm_pattern = r"^Unnamed: 0|letter|world_landmark_\d+\.[xyz]$"
        self.y_col_name = 'letter'
        self.model_path = f'{os.path.dirname(__file__)}/model.pkl'
        self.clf = self.load_model()

    def run(self, test_csv_path: str):
        x_test = self.prepare_data(test_csv_path)
        y_pred = self.predict(x_test)
        return y_pred

    def load_model(self):
        clf = joblib.load(self.model_path)
        return clf

    def predict(self, x_test: pd.DataFrame) -> np.ndarray:
        y_pred = self.clf.predict(x_test)
        return y_pred

    def prepare_data(self, path: str) -> pd.DataFrame:
        data = pd.read_csv(path)
        data = self.rm_cols(data)
        data = pd.get_dummies(data)
        return data

    def rm_cols(self, data: pd.DataFrame) -> pd.DataFrame:
        cols_to_rm = [col for col in data.columns if re.search(self.rm_pattern, col)]
        return data.drop(columns=cols_to_rm)
