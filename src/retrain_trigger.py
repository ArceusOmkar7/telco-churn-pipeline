import logging
from pathlib import Path

import pandas as pd

from src.evaluate import evaluate
from src.monitor import load_records, report
from src.preprocess import preprocess
from src.register import register
from src.train import train

DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
FILEPATH = next(Path(RAW_DATA_DIR).glob("*.csv"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_report():
    ref_df, sim_prod_df = load_records()
    drift_detected = report(ref_df, sim_prod_df)

    return drift_detected


def retrain():
    df = pd.read_csv(FILEPATH)
    X_train, X_test, y_train, y_test = preprocess(df)
    train(X_train, y_train, X_test, y_test)
    quality_check, latest_run_id = evaluate()

    if not quality_check:
        logger.warning(
            "Model didn't passed the quality checks on retrain. Skipping model registry."
        )
        return
    else:
        register(latest_run_id)


if __name__ == "__main__":
    retrain()
