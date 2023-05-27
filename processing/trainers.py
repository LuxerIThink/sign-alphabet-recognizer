import os
import re
import itertools
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


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

    @staticmethod
    def check_score(y_test_dec: pd.DataFrame, y_pred_labels: list[str]):
        compare = pd.DataFrame({"Real": y_test_dec, "Pred": y_pred_labels})
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        similarity = (accuracy_score(compare["Real"], compare["Pred"]) * 100)
        print(f"Similarity: {similarity:.2f}%")

    @staticmethod
    def plot_conf_matrix(y_pred_labels, y_test_decoded):
        conf_matrix = confusion_matrix(y_test_decoded, y_pred_labels)
        plt.figure(figsize=(8, 6))
        classes = np.unique(np.concatenate((y_test_decoded, y_pred_labels)))
        tick_marks = np.arange(len(classes))
        thresh = conf_matrix.max() / 2
        for i, j in itertools.product(
                range(conf_matrix.shape[0]), range(conf_matrix.shape[1])
        ):
            plt.text(j, i, format(conf_matrix[i, j], "d"),
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > thresh else "black",
                     )
        plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.colormaps['GnBu'])
        plt.title("Confusion Matrix")
        plt.colorbar()
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.show()