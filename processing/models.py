import re
import pandas as pd
from sklearn.svm import SVC
import joblib


class SignAlphabetModel:
    def __init__(self, dataset_path: str, model_path: str):
        self.train_csv_path = dataset_path
        self.model_path = model_path
        self.regex_columns_to_remove = r"^Unnamed: 0|world_landmark_\d+\.[xyz]$"
        self.y_column_name = 'letter'
        self.rand_state = 42

    def prepare_data(self, path: str) -> tuple[pd.DataFrame, pd.Series]:
        data = pd.read_csv(path)
        data = self.rm_cols(data)
        x, y = self.split_xy(data)
        return x, y

    def rm_cols(self, data: pd.DataFrame) -> pd.DataFrame:
        cols_to_rm = [col for col in data.columns if re.search(self.regex_columns_to_remove, col)]
        return data.drop(columns=cols_to_rm)

    def split_xy(self, data) -> tuple[pd.DataFrame, pd.Series]:
        y = data.pop(self.y_column_name) if self.y_column_name in data.columns else None
        x = pd.get_dummies(data)
        return x, y

    def train_clf(self, x_train: pd.DataFrame, y_train: pd.Series) -> SVC:
        clf = SVC(kernel="linear", C=250, gamma='auto', random_state=self.rand_state)
        clf.fit(x_train, y_train)
        return clf

    def create_model(self):
        x_train, y_train = self.prepare_data(self.train_csv_path)
        clf = self.train_clf(x_train, y_train)
        joblib.dump(clf, self.model_path)


if __name__ == '__main__':
    model = SignAlphabetModel('test/datasets/train_data.csv', 'model.pkl')
    model.create_model()
