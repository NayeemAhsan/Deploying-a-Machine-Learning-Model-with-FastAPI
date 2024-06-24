#!/usr/bin/env python
"""
This script splits the provided dataframe in test and remainder
Auther: Nayeem Ahsan
Date: 6/19/2024
"""
import argparse
import logging
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    logger.info('get the config file')
    with open(args.config, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    # read processed data
    logger.info('read processed data')
    data_path = config["data"]["processed_data_path"]

    df = pd.read_csv(data_path)

    logger.info("Splitting the dataset into train and test")
    train, test = train_test_split(
        df,
        test_size=config["data"]['test_size'],
        random_state=config["data"]['random_seed'],
        stratify=df[config["data"]['stratify_by']] if config["data"]['stratify_by'] != 'none' else None,
    )

    # Save to output files
    #train.to_csv(config["data"]["train_path"], index=False)
    #test.to_csv(config["data"]["test_path"], index=False)

    for df, k in zip([train, test], ['train', 'test']):
        logger.info(f"saving {k}_data.csv dataset")

        df.to_csv(config["data"][f"{k}_path"], index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="load the data"
    )

    parser.add_argument(
        "--config", help="YAML file", required=True, default= 'config.yaml'
    )

    args = parser.parse_args()

    go(args)