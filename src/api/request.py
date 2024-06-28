import requests

url = "http://127.0.0.1:8000/predict"
data = {
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

response = requests.post(url, json=data)
print(response.json())
