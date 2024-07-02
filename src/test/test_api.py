"""
Test file to test the APIs
Author: Nayeem Ahsan
Date: 06/28/2024
"""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_get_welcome_message():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to FastAPI"}


def test_post_prediction_valid_data(config):

    valid_data = config["sample_data"]["valid_data"]
    
    response = client.post("/predict", json=valid_data)
    assert response.status_code == 200

    prediction = response.json()["predictions"][0]
    assert prediction in [0, 1]


def test_post_prediction_missing_optional_fields(config):

    incomplete_data = config["sample_data"]["incomplete_data"]
    
    response = client.post("/predict", json=incomplete_data)
    # Expecting a 422 status code for validation error
    assert response.status_code == 422


def test_predict_above_50k(config):
    
    input_data = config["sample_data"]["data_above_50k"]
    
    response = client.post("/predict", json=input_data)
    assert response.status_code == 200

    prediction = response.json()["predictions"][0]
    # Map prediction to label
    label = ">50K" if prediction == 1 else "<=50K"
    assert label == ">50K"


def test_predict_below_50k(config):

    input_data = config["sample_data"]["data_below_50k"]
    
    response = client.post("/predict", json=input_data)
    assert response.status_code == 200

    prediction = response.json()["predictions"][0]
    # Map prediction to label
    label = ">50K" if prediction == 1 else "<=50K"
    assert label == "<=50K"
