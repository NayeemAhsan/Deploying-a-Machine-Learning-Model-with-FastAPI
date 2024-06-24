#!/usr/bin/env python
"""
This step takes the model, and tests it against the test dataset
Auther: Nayeem Ahsan
Date: 6/19/2024
"""
import argparse
import logging
import pandas as pd
import yaml
import joblib
from sklearn.metrics import fbeta_score, precision_score, recall_score
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

def read_config(args):
    '''
    read parameters from the config.yaml file
    input: config.yaml location
    output: parameters as dictionary
    '''
    logger.info("get the config file")
    with open(args.config) as yaml_file:
        config = yaml.safe_load(yaml_file)

    return config

def compute_metrics(y_true, y_pred):
    """
    Computes precision, recall and f1 scores

    Args:
        y_true (array): array of true labels
        y_pred (array): array of predicted labels

    Returns:
        f1 (float): f1 score
        precision (float): precision score
        recall (float): recall score
    """
    f1 = fbeta_score(y_true, y_pred, beta=1, zero_division=1)
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)

    return f1, precision, recall

def main():

    parser = argparse.ArgumentParser(description="Load configuration and read data")
    parser.add_argument(
        "--config", help="Path to the configuration file", required=True, default= 'config.yaml'
    )
    args = parser.parse_args()

    # Read the configuration file
    config = read_config(args)

    logger.info("Downloading model artifacts")
    model = config['data']['model_path']

    # Download test dataset
    test_data = config["data"]["test_path"]

    # Read test dataset
    X_test = pd.read_csv(test_data)
    y_test = X_test.pop("salary")

    #y_test = y_test.astype(int)  # Convert to binary if not already

    logger.info("Loading model and performing inference on test set")
    sk_pipe = joblib.load(model)
    y_pred = sk_pipe.predict(X_test)
    y_pred = y_pred.astype(int) # Convert to binary if not already

    logger.info("Evaluating metrics")
    precision, recall, f1_beta = compute_metrics(y_pred, y_test)

    logger.info(f"Precision: {precision}")
    logger.info(f"Recall: {recall}")
    logger.info(f"f1_beta: {f1_beta}")


if __name__ == "__main__":
    main()