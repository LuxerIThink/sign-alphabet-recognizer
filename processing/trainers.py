import os
import re
import pandas as pd


class ModelTrainer:
    def __init__(self):
        self.train_csv_path = f'{os.path.dirname(__file__)}/test/dataset/test_data.csv'
        self.rm_pattern = r"^Unnamed: 0|world_landmark_\d+\.[xyz]$"

    def run(self, test_csv_path: str):
        train_data = pd.read_csv(self.train_csv_path)
        test_data = pd.read_csv(test_csv_path)

        train_data = self.rm_cols(train_data)
        test_data = self.rm_cols(test_data)

        pass

    def rm_cols(self, data: pd.DataFrame) -> pd.DataFrame:
        cols_to_rm = [col for col in data.columns if re.search(self.rm_pattern, col)]
        data = data.drop(columns=cols_to_rm)
        return data
