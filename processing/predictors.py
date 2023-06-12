import os
import re
import pandas as pd
import numpy as np
import joblib


class SignAlphabetPredictor:
    def __init__(self, model_path: str = None, rm_pattern: str = None, y_name: str = None):
        self.model_path = model_path or f'{os.path.dirname(__file__)}/model.pkl'
        self.rm_pattern = rm_pattern or r"^Unnamed: 0|hand*|letter|world_landmark_\d+\.[xyz]$"
        self.y_name = y_name or 'letter'
        self.clf = self.load_model(self.model_path)

    @staticmethod
    def load_model(model_path: str):
        clf = joblib.load(model_path)
        return clf

    def run(self, data: pd.DataFrame) -> np.ndarray:
        x_test = self.prepare_data(data)
        y_pred = self.predict(x_test)
        return y_pred

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data = self.rm_cols(data)
        data = pd.get_dummies(data)
        return data

    def rm_cols(self, data: pd.DataFrame) -> pd.DataFrame:
        cols_to_rm = [col for col in data.columns if re.search(self.rm_pattern, col)]
        return data.drop(columns=cols_to_rm)

    def predict(self, x_test: pd.DataFrame) -> pd.Series:
        y_pred = self.clf.predict(x_test)
        y_pred = pd.Series(y_pred, name=self.y_name)
        return y_pred
