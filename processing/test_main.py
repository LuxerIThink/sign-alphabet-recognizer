import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import itertools
import pytest
from subprocess import check_output
from pathlib import Path


@pytest.fixture
def run_main():
    def run_main_with_args(main_path, data_path, output_file):
        command = ["python3", main_path, data_path, output_file]
        result = check_output(command)
        return result.decode("utf-8")
    return run_main_with_args


def test_reproduction_percentage(run_main):
    file_path = Path(__file__).resolve().parent
    main_path = file_path.parent / "Pawlowski_Adam.py"
    data_file = file_path / 'test' / 'datasets' / 'test1_no_y_data.csv'
    data_file_with_y = file_path / 'test' / 'datasets' / 'test1_data.csv'
    output_file = file_path.parent / "output.txt"
    y_column = 'letter'

    _ = run_main(main_path, data_file, output_file)

    data = pd.read_csv(data_file_with_y)

    y_test_decoded = data[y_column]

    with open(output_file, 'r') as f:
        y_pred_labels = f.read().splitlines()

    similarity = accuracy_score(y_test_decoded, y_pred_labels) * 100
    print(f"Reproduction Percentage: {similarity:.2f}%")

    plot_conf_matrix(y_pred_labels, y_test_decoded)

    assert similarity >= 50


def plot_conf_matrix(y_pred_labels, y_test_decoded):
    conf_matrix = confusion_matrix(y_test_decoded, y_pred_labels)
    classes = np.unique(np.concatenate((y_test_decoded, y_pred_labels)))
    tick_marks = np.arange(len(classes))
    thresh = conf_matrix.max() / 2
    plt.figure(figsize=(8, 6))
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
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


if __name__ == '__main__':
    pytest.main([__file__])
