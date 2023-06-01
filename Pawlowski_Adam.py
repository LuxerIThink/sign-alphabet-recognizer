import sys
from processing.predictors import SignAlphabetPredictor
import numpy as np


def save_file(data: np.ndarray, output_csv_path: str):
    with open(output_csv_path, 'w') as file:
        file.write('\n'.join(data))


if __name__ == '__main__':
    test_csv_path = sys.argv[1]
    output_csv_path = sys.argv[2]
    trainer = SignAlphabetPredictor()
    y_pred_labels = trainer.run(test_csv_path)
    save_file(y_pred_labels, output_csv_path)
