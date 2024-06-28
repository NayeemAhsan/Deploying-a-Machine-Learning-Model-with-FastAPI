'''
Main file to create API using FastAPI
Author: Nayeem Ahsan
Date: 06/28/2024
'''
import yaml
import joblib # helps to load pre-defined model
import pandas as pd
import logging
from fastapi import FastAPI
from pydantic import BaseModel # Provides data validation and settings management using Python type annotations

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Load configuration
logger.info("Getting the config file")
with open('config.yaml') as yaml_file:
    config = yaml.safe_load(yaml_file)

# Load the pre-trained model
model_path = config["data"]["model_path"]
model = joblib.load(model_path)

# Initialize FastAPI app and router
app = FastAPI(title="Model Inference API", description="Deploying a ML Model with FastAPI", version="0.1")

# Define Pydantic model for input data
class InputData(BaseModel):
    age: int
    workclass: str|None 
    fnlgt: int
    education: str|None 
    education_num: int
    marital_status: str|None # this "|" indicates that the data type of the 'marital_status' feature is optional.
    occupation: str|None # so this feature can be a string or it doesn't contain any data type which means it can be an empty field
    relationship: str|None # so, this is an Union (OR function)
    race: str|None # this '|' is not available in the previous version of python, where we need to import Optional modeule from the typing package
    sex: str|None # in that case, the code will be like this: sex: Optional[str] = None
    capital_gain: int # in this version of python (v10), the optional module is included and we can utlize just typing this '|'
    capital_loss: int
    hours_per_week: int
    native_country: str|None

    class ConfigDict: # A nested class to provide extra configurations, such as an example of the input data.
        json_schema_extra = {
            "example": {
                'age': 50,
                'workclass': "Private",
                'fnlgt': 234721,
                'education': "Doctorate",
                'education_num': 16,
                'marital_status': "Separated",
                'occupation': "Exec-managerial",
                'relationship': "Not-in-family",
                'race': "Black",
                'sex': "Female",
                'capital_gain': 0,
                'capital_loss': 0,
                'hours_per_week': 50,
                'native_country': "United-States"
            }
        }

# Define routes
@app.get("/")
async def greetings():
    return {"message": "Welcome to FastAPI"}

@app.post("/predict")
async def predict(data: InputData):
    input_df = pd.DataFrame([data.model_dump()])
    predictions = model.predict(input_df)

    # Apply thresholding for binary classification
    if isinstance(predictions[0], float):
        predictions = (predictions >= 0.5).astype(int)

    return {"predictions": predictions.tolist()}

