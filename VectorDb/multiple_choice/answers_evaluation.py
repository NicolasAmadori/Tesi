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


def main():
    CSV_FOLDER_NAME = "answers"
    dataframes_list = read_csvs(CSV_FOLDER_NAME)
    for root, file, df in dataframes_list:
        if file == "UniboSviCoop.csv":
            model_name = root.split("/")[-1]
            logger.info(f"Evaluating {model_name} results")

            #Stampando la statistica delle risposte corrette
            booleani = df["se_corretta"].tolist()
            totale = len(booleani)
            true_count = sum(booleani)
            percentuale_true = (true_count / totale) * 100 if totale > 0 else 0
            logger.info(f"Quantità di risposte corrette: {true_count}/{totale} ({percentuale_true:.2f}%)")

            # for index, row in df.iterrows():
            #     risposta_gold = row['risposta_gold']
            #     risposta = row['risposta']
            #     se_corretta = row['se_corretta']
            #     if not se_corretta:
            #         logger.info(f"{risposta_gold[0]} {risposta[0]}")

if __name__ == "__main__":
    main()
