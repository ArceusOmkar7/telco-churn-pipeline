import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from pathlib import Path
import logging

DATA_DIR = Path('data')
PROCESSED_DIR = DATA_DIR / 'processed'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ML FLow setup
mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("Telco Churn Pipeline")
mlflow.sklearn.autolog(log_models=False)
logger.info("Ml flow setup done")


def train(X_train, y_train, X_test, y_test):
    rf_params = {
        "criterion": "gini",    # preferred for binary classification
    }
    rf_clf = RandomForestClassifier(**rf_params)

    rf_clf.fit(X_train, y_train)

    # y_pred = rf_clf.predict(X_test)
    # y_pred_proba = rf_clf.predict_proba(X_test)[:, 1]

    # mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    # mlflow.log_metric("f1 score", f1_score(y_test, y_pred))
    # mlflow.log_metric("auc", roc_auc_score(y_test, y_pred_proba))
    # mlflow.sklearn.log_model(rf_clf, artifact_path="model")

if __name__ == "__main__":
    # Putting header none, so pandas doesn't make the first row header
    X_train = pd.read_csv(PROCESSED_DIR / 'X_train.csv', header=None)
    X_test = pd.read_csv(PROCESSED_DIR / 'X_test.csv', header=None)

    # df with single row/col (here col) will be squeezed into a Series
    y_train = pd.read_csv(PROCESSED_DIR / 'y_train.csv', header=None).squeeze()
    y_test = pd.read_csv(PROCESSED_DIR / 'y_test.csv', header=None).squeeze()

    train(X_train, y_train, X_test, y_test)