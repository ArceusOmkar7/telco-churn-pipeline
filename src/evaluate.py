import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from pathlib import Path
import logging
import tempfile
import os

DATA_DIR = Path('data')
PROCESSED_DIR = DATA_DIR / 'processed'

ACCURACY_SCORE_THREHOLD = 0.70
F1_SCORE_THREHOLD = 0.65
AUC_THREHOLD = 0.75


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri("http://localhost:5001")

class LatestRunNotFound(Exception):
    """Raised when no latest run found"""
    pass

def load_latest_run():
    logger.info("Retrieving latest run from MLFlow")
    # Retrieve the latest run for a specific experiment ID
    latest_runs = mlflow.search_runs(
        experiment_names=['Telco Churn Pipeline'],
        max_results=1,
        order_by=["attributes.start_time DESC"]
    )
    
    if not latest_runs.empty:
        latest_run_id = latest_runs.iloc[0]["run_id"]
        logger.info("Found latest Run ID: %s", latest_run_id)
    else:
        logger.error("No run found in the experiment")
        raise LatestRunNotFound

    model = mlflow.sklearn.load_model(f"runs:/{latest_run_id}/model")
    logger.info("Loaded model from run [%s]", latest_run_id)

    return model, latest_run_id

def load_test_data():
    X_test = pd.read_csv(PROCESSED_DIR / 'X_test.csv', header=None)
    y_test = pd.read_csv(PROCESSED_DIR / 'y_test.csv', header=None).squeeze()
    logger.info("Loaded test data")

    return X_test, y_test

def compute_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')       # weighted due to class imbalance
    auc = roc_auc_score(y_test, y_pred_proba)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true=y_test, 
        y_pred=y_pred, 
        ax=ax, 
        cmap="Blues"
    )

    metrics = {
        "accuracy": accuracy,
        "f1_score": f1,
        "auc": auc,
        "confusion_matrix": fig
    }
    
    return metrics
     

def quality_gate(metrics):
    pass_run = True
    reasons = []
    if metrics['accuracy'] < ACCURACY_SCORE_THREHOLD:
        pass_run = False
        reasons.append(f'Accuracy below threshold ({ACCURACY_SCORE_THREHOLD})')
    if metrics['f1_score'] < F1_SCORE_THREHOLD:
        pass_run = False
        reasons.append(f'F1 score below threshold ({F1_SCORE_THREHOLD})')
    if metrics['auc'] < AUC_THREHOLD:
        pass_run = False
        reasons.append(f'AUC below threshold ({AUC_THREHOLD})')
    
    logger.info("Run evaluation: %s | Reasons: %s", pass_run, ", ".join(reasons) if reasons else "[]")
    return pass_run, reasons

def evaluate():
    model, latest_run_id = load_latest_run()
    X_test, y_test = load_test_data()
    metrics = compute_metrics(model, X_test, y_test)
    quality_check, reasons = quality_gate(metrics)

    with mlflow.start_run(run_id=latest_run_id, nested=True):
        mlflow.log_metric("accuracy", metrics['accuracy'])
        mlflow.log_metric("f1 score", metrics['f1_score'])
        mlflow.log_metric("auc", metrics['auc'])

        with tempfile.TemporaryDirectory() as tmpdir:
            conf_matrix_path = os.path.join(tmpdir, "confusion_matrix.png")
            metrics['confusion_matrix'].savefig(conf_matrix_path)
            mlflow.log_artifact(conf_matrix_path)
            plt.close(metrics['confusion_matrix'])

        if quality_check:
            mlflow.set_tag("quality_gate", "passed")
        else:
            mlflow.set_tag("quality_gate", "failed")
            mlflow.set_tag("quality_gate_failed_reasons", ", ".join(reasons))

    logger.info("Logged metrics to MLFlow for run [%s]", latest_run_id)
    
    return quality_check, latest_run_id if quality_check else None


if __name__ == "__main__":
    evaluate()