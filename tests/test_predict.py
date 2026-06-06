from fastapi.testclient import TestClient
from api.main import app
from unittest.mock import patch, MagicMock

client = TestClient(app)

def test_predict_endpoint():
    sample_payload = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85
    }
    
    # Mock the model_service to avoid mlflow connection
    with patch("api.main.model_service") as mock_service:
        mock_service.predict.return_value = {"prediction": False, "probability": 0.3}
        
        # Bypass the lifespan load_model
        with patch("api.predict.ModelService.load_model"):
            response = client.post("/predict", json=sample_payload)
        
    assert response.status_code == 200
    assert response.json() == {"prediction": False, "probability": 0.3}
