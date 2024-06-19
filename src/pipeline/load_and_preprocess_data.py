#!/usr/bin/env python
"""
This script loads and preprocess the data

Auther: Nayeem Ahsan
Date: 6/13/2024
"""
import argparse
import logging
import yaml
import pandas as pd 


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    logger.info('get the config file')
    with open(args.config) as yaml_file:
        config = yaml.safe_load(yaml_file)

    #read the data
    logger.info('read data')
    data = config['data']['data_path']
    df = pd.read_csv(data, sep=",", encoding='utf-8')
    
    # chaning column names to use _ instead of -
    columns = df.columns
    columns = [col.replace('-', '_') for col in columns]
    df.columns = columns
    
    # remove duplicates
    logger.info('remove duplicates')
    df = df[~df.duplicated()]

    #Removing leading/trailing whitespaces
    df.columns = [c.strip() for c in df.columns]

    # make all characters to be lowercase in string columns
    df = df.map(
        lambda s: s.lower() if isinstance(s, str) else s)

    # Strip any leading/trailing whitespace and convert to lower case
    df['salary'] = df['salary'].str.strip().str.lower()

    # Convert 'salary' to binary values
    logger.info("mapping 'salary' col to binary integers")
    df['salary'] = df['salary'].map({'<=50k': 0, '>50k': 1})

    # Separate features and target
    logger.info("Separate features and target columns")
    X_df = df.drop('salary', axis=1)
    y_df = df['salary']

    return X_df, y_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="load the data"
    )

    parser.add_argument(
        "--config", help="YAML file", required=True, default= 'config.yaml'
    )

    args = parser.parse_args()

    go(args)