from pathlib import Path
import os
import logging
import kagglehub
import pandas as pd


DATASET_ID = "blastchar/telco-customer-churn"
DATA_DIR = Path('data')
RAW_DATA_DIR = DATA_DIR / "raw"


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Downloading dataset ---------------------------------
def download_dataset():
    logger.info("Downloading dataset from kaggle...")
    path = kagglehub.dataset_download(DATASET_ID, output_dir=RAW_DATA_DIR, force_download=True)
    logger.info("Downloaded to: %s", path)

    file = next(Path(RAW_DATA_DIR).glob("*.csv"))

    return file

# Basic checks ---------------------------------
def basic_checks(filepath):
    df = pd.read_csv(filepath)

    # print(df.head())

    logger.info("\nColumns: %s", ", ".join(df.columns.to_list()))
    na_cols = []
    for col in df.columns:
        if df[col].isna().sum() > 0:
            na_cols.append(col)

    logger.info("\nColumns with NA values: %s", na_cols)
    for col in na_cols:
        logger.info("%s\t%s", col, df[col].isna().sum())

    logger.info("\n%s", df.dtypes)

    logger.info("\nSummary")
    logger.info("Rows: %s", df.shape[0])
    logger.info("Columns: %s", df.shape[1])
    if "Churn" in df.columns:
        class_balance = df["Churn"].value_counts(dropna=False)
        logger.info("Class balance (Churn):")
        for label, count in class_balance.items():
            pct = (count / len(df)) * 100 if len(df) else 0
            logger.info("  %s: %s (%.1f%%)", label, count, pct)
    
    return df
    
    

def ingest():
    filepath = download_dataset()
    basic_checks(filepath)

if __name__ == "__main__":
    ingest()