import pandas as pd
import numpy as np
import pytest
from src.preprocess import preprocess
from unittest.mock import patch

def test_preprocess_shapes_and_nulls(tmp_path):
    # Create sample data
    data = {
        "customerID": [str(i) for i in range(10)],
        "gender": ["Female", "Male"] * 5,
        "SeniorCitizen": [0, 1] * 5,
        "Partner": ["Yes", "No"] * 5,
        "Dependents": ["No", "Yes"] * 5,
        "tenure": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "PhoneService": ["No", "Yes"] * 5,
        "MultipleLines": ["No phone service", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"],
        "InternetService": ["DSL", "Fiber optic"] * 5,
        "OnlineSecurity": ["No", "Yes"] * 5,
        "OnlineBackup": ["Yes", "No"] * 5,
        "DeviceProtection": ["No", "Yes"] * 5,
        "TechSupport": ["No", "Yes"] * 5,
        "StreamingTV": ["No", "Yes"] * 5,
        "StreamingMovies": ["No", "Yes"] * 5,
        "Contract": ["Month-to-month", "One year"] * 5,
        "PaperlessBilling": ["Yes", "No"] * 5,
        "PaymentMethod": ["Electronic check", "Mailed check"] * 5,
        "MonthlyCharges": [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0],
        "TotalCharges": ["20.0", "30.0", "40.0", "50.0", "60.0", "70.0", "80.0", "90.0", "100.0", "110.0"],
        "Churn": ["No", "Yes"] * 5
    }
    df = pd.DataFrame(data)
    
    # Mock the directories to avoid side effects
    with patch("src.preprocess.PROCESSED_DIR", tmp_path / "processed"), \
         patch("src.preprocess.REFERENCE_DIR", tmp_path / "reference"), \
         patch("src.preprocess.MODELS_DIR", tmp_path / "model"):
        
        (tmp_path / "processed").mkdir()
        (tmp_path / "reference").mkdir()
        (tmp_path / "model").mkdir()
        
        X_train, X_test, y_train, y_test = preprocess(df)
    
    # Assert output shapes
    # 80/20 split of 10 rows is 8 and 2
    assert X_train.shape[0] == 8
    assert X_test.shape[0] == 2
    assert y_train.shape[0] == 8
    assert y_test.shape[0] == 2
    
    # Assert no nulls
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()
    assert not np.isnan(y_train).any()
    assert not np.isnan(y_test).any()
