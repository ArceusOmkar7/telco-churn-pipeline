import logging
import time

import mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


mlflow.set_tracking_uri("http://localhost:5001")
client = mlflow.MlflowClient()

MODEL_NAME = "churn-model"


class LatestRunNotFound(Exception):
    """Raised when no latest run found"""

    pass


class TagNotFound(Exception):
    """Raised when the run doesn't have the required tag associated with it."""


def load_latest_run():
    logger.info("Retrieving latest run from MLFlow")
    # Retrieve the latest run for a specific experiment ID
    latest_runs = mlflow.search_runs(
        experiment_names=["Telco Churn Pipeline"],
        max_results=1,
        order_by=["attributes.start_time DESC"],
    )

    if not latest_runs.empty:
        latest_run_id = latest_runs.iloc[0]["run_id"]
        logger.info("Found latest Run ID: %s", latest_run_id)
    else:
        logger.error("No run found in the experiment")
        raise LatestRunNotFound

    # model = mlflow.sklearn.load_model(f"runs:/{latest_run_id}/model")
    # logger.info("Loaded model from run [%s]", latest_run_id)

    # return model, latest_run_id
    return latest_run_id


def register(latest_run_id):
    run = client.get_run(latest_run_id)

    tag_value = run.data.tags.get("quality_gate")

    if not tag_value:
        logger.error("Quality check tag not found in this run [%s]", latest_run_id)
        raise TagNotFound

    if tag_value == "failed":
        logger.info("Run didn't pass the quality check, skipping model registry.")
    else:
        model_uri = f"runs:/{run.info.run_id}/model"
        result = mlflow.register_model(model_uri, MODEL_NAME)

        while result.status != "READY":
            time.sleep(1)
            result = client.get_model_version(MODEL_NAME, result.version)
        logger.info(
            "Model registered\nversion: %s\nname: %s", result.version, MODEL_NAME
        )

        client.set_registered_model_alias(MODEL_NAME, "production", result.version)
        logger.info(
            "Model transitioned to production stage. Model version: %s", result.version
        )


if __name__ == "__main__":
    latest_run_id = load_latest_run()
    register(latest_run_id)
