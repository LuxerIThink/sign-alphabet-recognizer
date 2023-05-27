import pandas as pd
import csv


def split_data(input_file: str, output_file: str):
    df = pd.read_csv(input_file)
    extruded_data = df.iloc[::5]
    extruded_data.to_csv(output_file, index=False)


def re_last_column(input_file: str, output_file: str):
    with open(input_file, 'r') as file:
        reader = csv.reader(file)
        rows = [row[:-1] for row in reader]
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)