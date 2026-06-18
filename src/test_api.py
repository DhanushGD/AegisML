# src/test_api.py
import pytest
from fastapi.testclient import TestClient
from src.app import app

def test_health_check():
    """Ensure the API root endpoint is responsive and healthy."""
    # Using 'with' forces FastAPI to trigger the lifespan setup blocks
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

def test_prediction_schema():
    """Verify that the /predict endpoint returns the correct multi-label targets."""
    payload = {"text": "This is a clean test sentence for testing automation schema structure."}
    
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        
        # If this assert crashes, it will print the error detail from the server
        assert response.status_code == 200, f"Server returned 500 with error: {response.text}"
        
        data = response.json()
        assert "text" in data
        assert "predictions" in data
        
        expected_labels = ['label_hate_speech', 'label_insult', 'label_humiliate', 'label_dehumanize', 'label_violence']
        for label in expected_labels:
            assert label in data["predictions"]
            assert 0.0 <= data["predictions"][label] <= 1.0

def test_empty_payload_handling():
    """Ensure the API throws a 400 bad request error for empty string payloads."""
    payload = {"text": "   "}
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 400