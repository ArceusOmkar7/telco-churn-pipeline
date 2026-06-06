import logging
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"

REFERENCE_DIR = DATA_DIR / "reference"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_records():
    filepath = next(Path(RAW_DATA_DIR).glob("*.csv"))
    ref_df = pd.read_csv(REFERENCE_DIR / "reference.csv")
    sim_prod_df = pd.read_csv(filepath)

    ref_df = ref_df.drop(columns=["Churn"])
    sim_prod_df = sim_prod_df.drop(columns=["customerID", "Churn"])

    # TotalCharges is string in raw CSV
    ref_df["TotalCharges"] = pd.to_numeric(ref_df["TotalCharges"], errors="coerce")
    sim_prod_df["TotalCharges"] = pd.to_numeric(
        sim_prod_df["TotalCharges"], errors="coerce"
    )

    ref_df = ref_df.dropna(subset=["TotalCharges"])
    sim_prod_df = sim_prod_df.dropna(subset=["TotalCharges"])

    return ref_df, sim_prod_df.iloc[5001:]


def report(ref_df, sim_prod_df):

    definition = DataDefinition(
        numerical_columns=["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"],
        categorical_columns=[
            "gender",
            "Partner",
            "Dependents",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
        ],
    )

    ref_dataset = Dataset.from_pandas(ref_df, data_definition=definition)
    cur_dataset = Dataset.from_pandas(sim_prod_df, data_definition=definition)

    report = Report([DataDriftPreset()])
    result = report.run(cur_dataset, ref_dataset)
    result.save_html("report.html")

    result_dict = result.dict()

    # Find the specific metric results from the list
    # DataDriftPreset populates multiple underlying metrics like 'DatasetDriftMetric'
    drift_metric = next(
        metric
        for metric in result_dict["metrics"]
        if metric["metric_name"].startswith("DriftedColumnsCount")
    )

    drifted_count = drift_metric["value"]["count"]
    drifted_share = drift_metric["value"]["share"]

    drift_detected = drifted_count > 0

    if drift_detected:
        logger.warning(
            "Drift detected! %d columns drifted (%.2f%%)",
            int(drifted_count),
            drifted_share * 100,
        )
    else:
        logger.info("No data drift detected.")

    return drift_detected


if __name__ == "__main__":
    ref_df, sim_prod_df = load_records()
    report(ref_df, sim_prod_df)
