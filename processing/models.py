import re
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import joblib


class SignAlphabetModel:

    def __init__(self, rm_cols_pattern: str = None, y_name: str = None, state: int = None):
        self.rm_cols_pattern = rm_cols_pattern or r"^Unnamed: 0|hand*|world_landmark_\d+\.[xyz]$"
        self.y_name = y_name or 'letter'
        self.state = state or 42

    def create_model(self, dataset_path: str) -> SVC:
        x_train, y_train = self.prepare_data(dataset_path)
        classifier = self.train_clf(x_train, y_train)
        return classifier

    def prepare_data(self, path: str) -> tuple[pd.DataFrame, pd.Series]:
        data = pd.read_csv(path)
        data = self.remove_columns(data)
        data = self.filtrate_with_iqr(data)
        x, y = self.split_xy(data)
        return x, y

    def remove_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        cols_to_rm = [col for col in data.columns if re.search(self.rm_cols_pattern, col)]
        return data.drop(columns=cols_to_rm)

    def filtrate_with_iqr(self, data: pd.DataFrame) -> pd.DataFrame:
        # Identify unique values in the 'letter' column
        unique_letters = data[self.y_name].unique()

        # Create an empty DataFrame to store the filtered results
        filtered_df = pd.DataFrame()

        # Iterate over each unique letter value
        for letter in unique_letters:
            # Filter the DataFrame based on the current letter
            df_letter = data[data[self.y_name] == letter]

            # Filter numeric columns using IQR
            numeric_columns = df_letter.select_dtypes(include=[np.number]).columns
            for column in numeric_columns:
                # Calculate the IQR range
                q1 = df_letter[column].quantile(0.25)
                q3 = df_letter[column].quantile(0.75)
                iqr = q3 - q1

                # Define the lower and upper bounds
                lower_bound = q1 - 5 * iqr
                upper_bound = q3 + 5 * iqr

                # Filter the column's values within the IQR range
                df_letter = df_letter[(df_letter[column] >= lower_bound) & (df_letter[column] <= upper_bound)]

            # Concatenate the filtered rows for the current letter to the result DataFrame
            filtered_df = pd.concat([filtered_df, df_letter])

        return filtered_df

    def split_xy(self, data) -> tuple[pd.DataFrame, pd.Series]:
        y = data.pop(self.y_name) if self.y_name in data.columns else None
        x = pd.get_dummies(data)
        return x, y

    def train_clf(self, x_train: pd.DataFrame, y_train: pd.Series) -> SVC:
        classifier = SVC(kernel="linear", C=250, gamma='auto', random_state=self.state)
        classifier.fit(x_train, y_train)
        return classifier


if __name__ == '__main__':
    model = SignAlphabetModel()
    clf = model.create_model('test/datasets/all_data.csv')
    joblib.dump(clf, 'model.pkl')
