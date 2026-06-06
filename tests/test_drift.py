import pandas as pd
from src.monitor import report
from unittest.mock import patch

def test_drift_detection_returns_boolean():
    df = pd.DataFrame({
        "gender": ["Female", "Male"],
        "SeniorCitizen": [0, 0],
        "Partner": ["Yes", "No"],
        "Dependents": ["No", "No"],
        "tenure": [1, 34],
        "PhoneService": ["No", "Yes"],
        "MultipleLines": ["No phone service", "No"],
        "InternetService": ["DSL", "DSL"],
        "OnlineSecurity": ["No", "Yes"],
        "OnlineBackup": ["Yes", "No"],
        "DeviceProtection": ["No", "Yes"],
        "TechSupport": ["No", "No"],
        "StreamingTV": ["No", "No"],
        "StreamingMovies": ["No", "No"],
        "Contract": ["Month-to-month", "One year"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod": ["Electronic check", "Mailed check"],
        "MonthlyCharges": [29.85, 56.95],
        "TotalCharges": [29.85, 1889.5]
    })
    
    # Mock Report class to avoid writing files and evidently dependencies
    with patch("src.monitor.Report") as mock_report:
        mock_instance = mock_report.return_value
        mock_instance.run.return_value = mock_instance
        mock_instance.dict.return_value = {
            "metrics": [
                {
                    "metric_name": "DriftedColumnsCount",
                    "value": {"count": 0, "share": 0.0}
                }
            ]
        }
        drift_detected = report(df, df)
    
    assert isinstance(drift_detected, bool)
