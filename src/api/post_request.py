import requests
import logging
import yaml

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Load configuration
logger.info("Getting the config file")
with open('config.yaml') as yaml_file:
    config = yaml.safe_load(yaml_file)

logger.info("get the deployed url")
url = config["main"]["deployed_url"]

logger.info("update the url to get the results")
post_url = url+"/predict"

data = config["sample_data"]["data_above_50k"]

response = requests.post(post_url, json=data)
print(response.json())
