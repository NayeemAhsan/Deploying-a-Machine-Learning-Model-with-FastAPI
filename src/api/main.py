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
from pydantic import BaseModel, Field # Pydantic provides data validation and settings management using Python type annotations

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

# Info: The sign "|" indicates that the data type of the 'marital_status' feature is optional.
# So this feature can be a string or it doesn't contain any data type which means it can be an empty field
# so, this is an Union ('OR' function).
# this '|' is not available in the previous version of python, where we need to import the moddule, 'Optional' from the package 'typing' 
# in that case, the code will be like this: sex: Optional[str] = None
# in this version of python (v10), the optional module is included and we can utlize just typing this '|'
# The Field function is used to provide additional metadata and validation rules, which are especially useful for generating documentation and setting default values.
# The ellipsis ('...') indicates that this field is required. It means that the field must be provided when creating an instance of the InputData model.
# The 'None' sets the default value of the workclass field to None. It means that if this field is not provided when creating an instance of the InputData model, it will default to None.

class InputData(BaseModel):
    age: int = Field(..., example=50)
    workclass: str | None = Field(None, example="Private")
    fnlgt: int = Field(..., example=234721)
    education: str | None = Field(None, example="Doctorate")
    education_num: int = Field(..., example=16)
    marital_status: str | None = Field(None, example="Separated")
    occupation: str | None = Field(None, example="Exec-managerial")
    relationship: str | None = Field(None, example="Not-in-family")
    race: str | None = Field(None, example="Black")
    sex: str | None = Field(None, example="Female")
    capital_gain: int = Field(..., example=0)
    capital_loss: int = Field(..., example=0)
    hours_per_week: int = Field(..., example=50)
    native_country: str | None = Field(None, example="United-States")

    # A nested class to provide extra configurations, such as an example of the input data.
    class Config: 
        schema_extra = {
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

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)