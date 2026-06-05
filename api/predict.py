import pickle
import pandas as pd
import mlflow
from api.schemas import CustomerFeatures

class ModelService:
    def __init__(self, model_name: str = "churn-model", stage: str = "Production"):
        self.model_uri = f"models:/{model_name}/{stage}"
        self.model = None
        self.preprocessor = None
        mlflow.set_tracking_uri("http://localhost:5001")

    def load_model(self):
        print(f"Loading model from {self.model_uri}...")
        
        client = mlflow.MlflowClient()
        versions = client.search_model_versions("name='churn-model'")
        production_versions = [v for v in versions if v.current_stage == "Production"]
        run_id = production_versions[0].run_id

        self.model = mlflow.sklearn.load_model(self.model_uri)
        print(f"Model loaded. Run ID: {run_id}. Downloading preprocessor...")
        
        preprocessor_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="preprocessor.pkl"
        )
        with open(preprocessor_path, "rb") as f:
            self.preprocessor = pickle.load(f)
        print("Preprocessor loaded successfully.")

    def predict(self, features: CustomerFeatures):
        df = pd.DataFrame([features.model_dump()])
        X_processed = self.preprocessor.transform(df)
        prediction = self.model.predict(X_processed)
        probability = self.model.predict_proba(X_processed)
        return {
            "prediction": bool(prediction[0]),
            "probability": float(probability[0][1])
        }

model_service = ModelService()
