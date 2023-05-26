import sys
from processing.trainers import ModelTrainer

if __name__ == '__main__':
    test_csv_path = sys.argv[1]
    output_csv_path = sys.argv[2]
    trainer = ModelTrainer()
    pred_solution = trainer.run(test_csv_path)
