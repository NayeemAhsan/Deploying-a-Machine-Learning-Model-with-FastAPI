#!/usr/bin/env python
"""
This script trains a Random Forest
Auther: Nayeem Ahsan
Date: 6/19/2024
"""
import argparse
import logging
import os
import shutil
import matplotlib.pyplot as plt
import yaml
import tempfile
import itertools
import joblib

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline, make_pipeline


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def model(args):
    '''
    This function uses args as parameters and builds regression model, evulates it, and then saves the model. 
    '''
    logger.info("get the config file")
    with open(args.config, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    # load the dataset and split to train and val
    logger.info("reading train artifact")
    train_data_path = config["data"]["train_path"]
    df = pd.read_csv(train_data_path, low_memory=False)

    # Extract the target from the features
    logger.info("Extracting target from dataframe")
    X = df.copy()
    y = X.pop("salary")  # this removes the column "price" from X and puts it into y

    logger.info("Splitting train/val")
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=config["data"]['val_size'],
        random_state=config["data"]['random_seed'],
        stratify=df[config["data"]['stratify_by']] if config["data"]['stratify_by'] != 'none' else None,
    )

    logger.info("Preparing sklearn pipeline")

    sk_pipe, processed_features = get_inference_pipeline(config)

    # Then fit it to the X_train, y_train data
    logger.info("Fitting")

    # Fit the pipeline sk_pipe by calling the .fit method on X_train and y_train
    logger.info("Training random forest model")
    sk_pipe.fit(X_train[processed_features], y_train)

    # Evaluate
    logger.info("Predicting validation data")
    pred = sk_pipe.predict(X_val[processed_features])
    #pred_proba = sk_pipe.predict_proba(X_val[processed_features])

    # Compute r2 and MAE
    logger.info("Scoring")
    r_squared = sk_pipe.score(X_val, y_val)

    y_pred = sk_pipe.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)

    logger.info(f"Score: {r_squared}")
    logger.info(f"MAE: {mae}")

    logger.info("Exporting model")

    # Save the model 
    save_model(config, sk_pipe)


def save_model(config, model):
    '''
    saving the model to the model path
    '''
    # saving the model
    logger.info('save the model')
    model_pth = config['data']['model_path']
    joblib.dump(model, model_pth)


def get_inference_pipeline(config):

    # categorical prerprocessing pipelne
    categorical_features = sorted(config['features']["non_ordinal_categ"])
    categorical_preproc = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder()
    )

    # Numerical preprocessing pipeline
    # Impute the numerical columns to make sure we can handle missing values
    numeric_features = sorted(config['features']["numerical"])
    numeric_transformer = SimpleImputer(strategy="constant", fill_value=0) # we do not scale because the RF algorithm does not need that

    # create the complete pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("transform_cat", categorical_preproc, categorical_features),
            ("transform_num", numeric_transformer, numeric_features),
        ],
        remainder="drop",  # This drops the columns that we do not transform
    )

    # Get a list of the columns we used
    processed_features = list(itertools.chain.from_iterable([x[2] for x in preprocessor.transformers]))

    # List of supported parameters for RandomForestRegressor
    supported_params = {
        'n_estimators', 'criterion', 'max_depth', 'min_samples_split', 'min_samples_leaf',
        'min_weight_fraction_leaf', 'max_features', 'max_leaf_nodes', 'min_impurity_decrease',
        'bootstrap', 'oob_score', 'n_jobs', 'random_state', 'verbose', 'warm_start',
        'ccp_alpha', 'max_samples'
    }

    # Filter rf_config to only include supported parameters
    filtered_rf_config = {k: v for k, v in config.items() if k in supported_params}

    random_forest = RandomForestRegressor(**filtered_rf_config)

    # Create random forest
    #random_Forest = RandomForestRegressor(**rf_config['random_forest'])

    # Create the inference pipeline. 
    sk_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('rf_model', random_forest)
    ])

    return sk_pipe, processed_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="load the data"
    )

    parser.add_argument(
        "--config", help="YAML file", required=True, default= 'config.yaml'
    )

    args = parser.parse_args()

    model(args)