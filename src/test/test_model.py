import pytest
import pandas as pd
import os
import logging
import pickle
from src.pipeline.model import get_inference_pipeline
from src.pipeline.evaluate import compute_metrics


def test_import_data(config):
    '''
    Test dataset files for presence and shape
    '''
    data_path = config["data"]["data_path"]

    if data_path is None:
        pytest.fail("Please provide the data_path to the config file")

    if not os.path.exists(data_path):
        pytest.fail(f"Data not found at path: {data_path}")

    print(f"Loading data from: {data_path}")

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError as err:
        logging.error("Failed to read data")
        raise err

    # Check the df shape
    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error("The file doesn't appear to have rows and columns")
        raise err


def test_columns_exist(config, df):
    """
    Test if all expected columns exist
    """
    features_config = config["features"]

    # Flatten the list of features
    columns = []
    for feature_list in features_config.values():
        columns.extend(feature_list)

    try:
        assert sorted(set(df.columns).intersection(columns)) == sorted(columns)
    except AssertionError as err:
        missing_features = set(columns) - set(df.columns)
        logging.error(
            f"Features are missing in the data set: {missing_features}")
        raise err


def test_model_presence(config):
    """
    Test saved model is present or not
    """
    model_path = config["data"]["model_path"]

    if not os.path.isfile(model_path):
        pytest.fail(f"Model not found at path: {model_path}")

    try:
        _ = pickle.load(open(model_path, 'rb'))
    except Exception as err:
        logging.error(
            "Testing saved model: Saved model does not appear to be valid")
        raise err


def test_model_training(config, train_data):
    """
    Test model training
    """
    X_train, X_test, y_train, y_test = train_data
    model, processed_features = get_inference_pipeline(config)

    try:
        # Fit the pipeline sk_pipe by calling the .fit method on X_train and
        # y_train
        logging.info("Training random forest model")
        model = model.fit(X_train[processed_features], y_train)
    except Exception as err:
        logging.error("Model training failed")
        raise err


def test_inference(config, train_data):
    '''
    Test model inference
    '''
    X_train, X_test, y_train, y_test = train_data
    model, processed_features = get_inference_pipeline(config)

    # Fit the pipeline sk_pipe by calling the .fit method on X_train and
    # y_train
    model = model.fit(X_train[processed_features], y_train)

    try:
        logging.info("Predicting test data")
        pred = model.predict(X_test[processed_features])
    except Exception as err:
        logging.error("Model inference failed")
        raise err

    y_pred = pred.astype(int)  # Convert to binary

    return y_pred, y_test


def test_metrics(config, train_data):
    '''
    Test metrics like Precision, Recall, and f-beta score
    '''
    y_pred, y_test = test_inference(config, train_data)

    try:
        logging.info("Evaluating metrics")
        precision, recall, f1_beta = compute_metrics(y_test, y_pred)
        logging.info(f"Precision: {precision}")
        logging.info(f"Recall: {recall}")
        logging.info(f"f1_beta: {f1_beta}")
    except Exception as err:
        logging.error("Model evaluation failed")
        raise err

    # assert precision > 0.7
    # assert recall > 0.7
    # assert f1_beta > 0.7
