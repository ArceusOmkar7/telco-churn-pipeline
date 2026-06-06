import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

MODELS_DIR = Path("model")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
FILEPATH = next(Path(RAW_DATA_DIR).glob("*.csv"))

PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_DIR = DATA_DIR / "reference"
REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 420

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess(df):
    # drop as its of no use in training
    df = df.drop(columns=["customerID"])
    logger.info("Dropped `customerID` column")

    # converting TotalCharges column to float
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    logger.info("Converted `TotalCharges` column to float")
    logger.info("Resulted in %d NaN records", df["TotalCharges"].isna().sum())
    logger.info("Dropped NaN Records")
    df = df.dropna()

    # print(df.head())

    # Encoding and scaling
    categorical_bin_columns = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
    ]
    categorical_multi_columns = [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod",
    ]
    numerical_columns = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    preprocessing_pipeline = ColumnTransformer(
        [
            ("categorical_bin_enc", OrdinalEncoder(), categorical_bin_columns),
            ("categorical_multi_enc", OneHotEncoder(), categorical_multi_columns),
            ("scaling", StandardScaler(), numerical_columns),
        ],
        remainder="drop",
    )

    le = LabelEncoder()

    y_processed = le.fit_transform(y)
    X_processed = preprocessing_pipeline.fit_transform(X)

    logger.info("Binary-categorical encoded columns: %s", categorical_bin_columns)
    logger.info("Multi-categorical encoded columns: %s", categorical_multi_columns)
    logger.info("Scaled numerical columns: %s", numerical_columns)
    logger.info("Label encoded target Churn column")

    # Train test split
    logger.info("Train test split into 80/20 ratio")
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed, test_size=0.2, stratify=y_processed, random_state=420
    )
    # Saving the splits into csv
    np.savetxt(PROCESSED_DIR / "X_train.csv", X_train, delimiter=",")
    np.savetxt(PROCESSED_DIR / "y_train.csv", y_train, delimiter=",")
    np.savetxt(PROCESSED_DIR / "X_test.csv", X_test, delimiter=",")
    np.savetxt(PROCESSED_DIR / "y_test.csv", y_test, delimiter=",")
    logger.info("Saved train test split files into %s", PROCESSED_DIR)

    ref_file = REFERENCE_DIR / "reference.csv"
    df.iloc[:5000, :].to_csv(ref_file, index=False)
    logger.info("Saved first 5000 raw records to %s", ref_file)

    # saving preprocessor
    preprocessor_file = MODELS_DIR / "preprocessor.pkl"
    with open(preprocessor_file, "wb") as f:
        pickle.dump(preprocessing_pipeline, f)
        logger.info("Saved preprocessor to %s", preprocessor_file)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = pd.read_csv(FILEPATH)
    logger.info("Loaded dataset: %s", FILEPATH)

    preprocess(df)
