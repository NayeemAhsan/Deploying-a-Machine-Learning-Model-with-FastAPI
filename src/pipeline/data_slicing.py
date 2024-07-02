#!/usr/bin/env python
"""
This script provides function for validation on a slice of dataset
Auther: Nayeem Ahsan
Date: 6/19/2024
"""
import logging
import argparse
import yaml
import pandas as pd
import joblib
from evaluate import compute_metrics

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

def compute_slices(df, feature, y, preds):
    """
    Compute the performance on slices for a given categorical feature
    ------
    df:
        test dataframe pre-processed with features as column used for slices
    feature:
        feature on which to perform the slices
    y : np.array
        corresponding known labels, binarized.
    preds : np.array
        Predicted labels, binarized

    Returns
    ------
    Dataframe with
        precision : float
        recall : float
        fbeta : float
    row corresponding to each of the unique values taken by the feature (slice)
    """
    slice_options = df[feature].unique().tolist()
    perf_df = pd.DataFrame(index=slice_options,
                           columns=['feature', 'precision', 'recall', 'fbeta'])
    for option in slice_options:
        slice_mask = df[feature] == option

        slice_y = y[slice_mask]
        slice_preds = preds[slice_mask]
        precision, recall, fbeta = compute_metrics(slice_y, slice_preds)

        perf_df.at[option, 'feature'] = feature
        perf_df.at[option, 'precision'] = precision
        perf_df.at[option, 'recall'] = recall
        perf_df.at[option, 'fbeta'] = fbeta

    # reorder columns in performance dataframe
    perf_df.reset_index()
    colList = list(perf_df.columns)
    colList[0], colList[1] = colList[1], colList[0]
    perf_df = perf_df[colList]

    return perf_df


def main():
    """
    Evaluting model on a slice of data for a specific column
    and data split and saving the results to a file
    """
    parser = argparse.ArgumentParser(description="Load configuration and read data")
    parser.add_argument(
        "--config", help="Path to the configuration file", required=True, default= 'config.yaml'
    )
    args = parser.parse_args()

    # Read the configuration file
    config = read_config(args)
    # Compute performance on slices for categorical features
    # output path to the slicing results in a txt file
    slice_output_path = config["data"]["slicing_output"]
    cat_features = config["features"]["non_ordinal_categ"]

    logger.info("Downloading model artifacts")
    model = config['data']['model_path']

    # Download test dataset
    test_data_path = config["data"]["test_path"]
    test_data = pd.read_csv(test_data_path, low_memory=False)

    # Read test dataset
    X_test = test_data.copy()
    y_test = X_test.pop("salary")

    logger.info("Loading model and performing inference on test set")
    sk_pipe = joblib.load(model)
    y_pred = sk_pipe.predict(X_test)
    y_pred = y_pred.astype(int) # Convert to binary if not already


    # iterate through the categorical features and save results to log and txt file
    for feature in cat_features:
        slicing_df = compute_slices(test_data, feature, y_test, y_pred)
        slicing_df.to_csv(slice_output_path,  mode='a', index=False)
        logger.info(f"Performance on slice {feature}")
        logger.info(slicing_df)

if __name__ == "__main__":
    main()