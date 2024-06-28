"""
Test file to test the APIs
Author: Nayeem Ahsan
Date: 06/28/2024
"""
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_get_welcome_message():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to FastAPI"}

def test_post_prediction_valid_data():
    valid_data = {
        "age": 25,
        "workclass": "Private",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }
    response = client.post("/predict", json=valid_data)
    assert response.status_code == 200
    
    prediction = response.json()["predictions"][0]
    assert prediction in [0, 1]

def test_post_prediction_missing_optional_fields():
    incomplete_data = {
        "age": 30,
        "fnlgt": 12345,
        "education_num": 9,
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40
    }
    response = client.post("/predict", json=incomplete_data)
    assert response.status_code == 422  # Expecting a 422 status code for validation error

def test_predict_above_50k() -> None:
    input_data = {
        "age": 40,
        "workclass": "Private",
        "fnlgt": 121772,
        "education": "Assoc-voc",
        "education_num": 11,
        "marital_status": "Married-civ-spouse",
        "occupation": "Craft-repair",
        "relationship": "Husband",
        "race": "Asian-Pac-Islander",
        "sex": "Male",
        "capital_gain": 7298,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }
    response = client.post("/predict", json=input_data)
    assert response.status_code == 200
    
    prediction = response.json()["predictions"][0]
    # Map prediction to label
    label = ">50K" if prediction == 1 else "<=50K"
    assert label == ">50K"

def test_predict_below_50k() -> None:
    input_data = {
        "age": 25,
        "workclass": "Private",
        "fnlgt": 226802,
        "education": "11th",
        "education_num": 7,
        "marital_status": "Never-married",
        "occupation": "Machine-op-inspct",
        "relationship": "Own-child",
        "race": "Black",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }
    response = client.post("/predict", json=input_data)
    assert response.status_code == 200
    
    prediction = response.json()["predictions"][0]
    # Map prediction to label
    label = ">50K" if prediction == 1 else "<=50K"
    assert label == "<=50K"
