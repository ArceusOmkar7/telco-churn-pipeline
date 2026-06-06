import logging
from pathlib import Path

import pandas as pd
from prefect import flow, task

from src.evaluate import evaluate
from src.ingest import ingest
from src.monitor import load_records, report
from src.preprocess import preprocess
from src.register import register
from src.train import train

DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@task(name="Ingest", retries=2, retry_delay_seconds=5)
def ingest_task():
    ingest()


@task(name="Preprocess", retries=1)
def preprocess_task():
    filepath = next(RAW_DATA_DIR.glob("*.csv"))
    df = pd.read_csv(filepath)
    return preprocess(df)


@task(name="Train")
def train_task(X_train, X_test, y_train, y_test):
    train(X_train, y_train, X_test, y_test)


@task(name="Evaluate")
def evaluate_task():
    return evaluate()


@task(name="Register")
def register_task(latest_run_id):
    register(latest_run_id)


@task(name="Monitor")
def monitor_task():
    ref_df, sim_prod_df = load_records()
    return report(ref_df, sim_prod_df)


@flow(name="Telco Churn Pipeline")
def pipeline(run_monitor: bool = False):
    ingest_task()

    X_train, X_test, y_train, y_test = preprocess_task()

    train_task(X_train, X_test, y_train, y_test)

    quality_check, latest_run_id = evaluate_task()

    if quality_check:
        register_task(latest_run_id)
        logger.info("Model registered successfully.")
    else:
        logger.warning("Quality gate failed. Skipping registration.")

    if run_monitor:
        drift_detected = monitor_task()
        if drift_detected:
            logger.warning("Drift detected — triggering retrain.")
            X_train, X_test, y_train, y_test = preprocess_task()
            train_task(X_train, X_test, y_train, y_test)
            quality_check, latest_run_id = evaluate_task()
            if quality_check:
                register_task(latest_run_id)


if __name__ == "__main__":
    pipeline(run_monitor=True)
