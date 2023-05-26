import os
import pandas as pd


class ModelTrainer:
    def __init__(self):
        self.train_csv_path = f'{os.path.dirname(__file__)}/test/dataset/test_data.csv'

    def run(self, test_csv_path: str):
        train_data = pd.read_csv(self.train_csv_path)
        test_data = pd.read_csv(test_csv_path)
        pass
