import pytest
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

def pytest_addoption(parser):
    parser.addoption("--config", action="store", help="Path to config file")

@pytest.fixture(scope='session')
def config(request):
    '''
    Read parameters from the config.yaml file.
    Input: pytest request object to get command-line options.
    Output: parameters as dictionary.
    '''
    config_path = request.config.getoption("--config")
    if config_path is None:
        pytest.fail("Please provide the config file path using --config option")
    
    print(f"Reading config file from: {config_path}")
    
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

@pytest.fixture(scope='session')
def df(config):
    # code to load in the data.
    data_path = config["data"]["processed_data_path"]
    
    if data_path is None:
        pytest.fail("Please provide the processed_data_path in the config file")

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    return df

@pytest.fixture(scope='session')
def train_data(df, config):
    """
    Sampled data from csv file used for tests
    """
    X = df.copy()
    y = X.pop("salary")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]['test_size'],
        random_state=config["data"]['random_seed'],
        stratify=df[config["data"]['stratify_by']] if config["data"]['stratify_by'] != 'none' else None,
    )

    return X_train, X_test, y_train, y_test
