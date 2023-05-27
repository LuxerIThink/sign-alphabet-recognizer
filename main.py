import sys
from processing.trainers import ModelTrainer

if __name__ == '__main__':
    test_csv_path = sys.argv[1]
    output_csv_path = sys.argv[2]
    trainer = ModelTrainer()
    y_pred_labels, y_test_decoded = trainer.run(test_csv_path)
    # trainer.check_score(y_test_decoded, y_pred_labels)
    # trainer.plot_conf_matrix(y_pred_labels, y_test_decoded)