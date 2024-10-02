import evaluate
import pandas as pd

import os
import logging
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)

def read_csvs(folder_name):
    dataframes = []
    for root, _, files in os.walk(folder_name):
        for file in files:
            file_path = os.path.join(root, file)
            dataframes.append((root, file, pd.read_csv(file_path)))
    return dataframes

def addRougeToDf(rouge, dataframe, reference_column_name, prediction_column_name):
    results = rouge.compute(predictions=dataframe[prediction_column_name], references=dataframe[reference_column_name], use_aggregator=False)
    for metric, values in results.items():
        dataframe[metric] = values
    return dataframe

def main():
    REFERENCE_COLUMN_NAME = "risposta_gold"
    PREDICTION_COLUMN_NAME = "risposta"
    CSV_FOLDER_NAME = "output"

    rouge = evaluate.load('rouge')
    dataframes_list = read_csvs(CSV_FOLDER_NAME)

    for folder, file_name, df in dataframes_list:
        logger.info(os.path.join(folder, file_name))
        df = addRougeToDf(rouge, df, REFERENCE_COLUMN_NAME, PREDICTION_COLUMN_NAME)

        file_name = file_name.replace(".csv", "_rouge.csv")
        df.to_csv(os.path.join(folder, file_name), index=False)
        

if __name__ == "__main__":
    main()