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

def sort_key(key):
    if key.startswith("k"):
        prefix, num = key.split('_')
        return (prefix, int(num))
    else:
        return (key, 0)

def print_tree(tree, level=0):
    for key in sorted(tree.keys(), key=sort_key):
        print("    " * level + f"└── {key}")
        print_tree(tree[key], level + 1)

def main():
    CSV_FOLDER_NAME = "multiple_choice/answers"
    dataframes_list = read_csvs(CSV_FOLDER_NAME)

    tree = {}
    for root, file, df in dataframes_list:
        parts = root.split("/")
        current = tree
        for part in parts:
            current = current.setdefault(part, {})

        booleani = df["se_corretta"].tolist()
        totale = len(booleani)
        true_count = sum(booleani)
        percentuale_true = (true_count / totale) * 100 if totale > 0 else 0
        current = current.setdefault(f"{file}: {true_count}/{totale} ({percentuale_true:.2f}%)", {})
    
    print_tree(tree)

if __name__ == "__main__":
    main()
