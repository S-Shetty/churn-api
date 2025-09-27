import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

sample_payload = {
    "data": [
        {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 5,
            "PhoneService": "Yes",
            "MultipleLines": "No",
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
            "MonthlyCharges": 70.35,
            "TotalCharges": 350.5
        }
    ]
}


def test_predict():
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert isinstance(data["predictions"], list)
    assert isinstance(data["predictions"][0], float)


def test_explain():
    response = client.post("/explain", json=sample_payload)
    assert response.status_code == 200
    data = response.json()
    assert "probability" in data
    assert "explanation" in data
    assert isinstance(data["probability"], float)
    assert isinstance(data["explanation"], dict)
