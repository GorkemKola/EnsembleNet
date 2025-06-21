### Logger

import os
import sys
import logging

logging_str = '[%(asctime)s: %(levelname)s: %(module)s: %(message)s:]'

log_dir = 'logs'
log_filename ='running_logs.log'
log_filepath = os.path.join(
    log_dir, log_filename
)
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,

    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout),
    ]
)

logger = logging.getLogger(
    'ensemblenetLogger'
)

### read yaml

from box import ConfigBox
from box.exceptions import BoxValueError
import yaml
from ensure import ensure_annotations
from pathlib import Path

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    '''
    Description:
        reads yaml file

    Args:
        path_to_yaml (Path): path like input

    Raises:
        ValueError: if yaml is empty
        e: other exceptions
    
    Returns:
        ConfigBox: ConfigBox type
    '''

    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f'yaml file {path_to_yaml} loaded successfully')
            return ConfigBox(content)
        
    except BoxValueError:
        raise ValueError('yaml file is empty')
    
    except Exception as e:
        raise e
    
### create directories

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    '''
    Description:
        create list of directories

    Args:
        path_to_directories (list): list of path of directories:
        verbose (bool): log process
    '''

    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f'created directory at: {path}')

### save-load json

import json

@ensure_annotations
def save_json(path: Path, data: dict):
    '''
    Description:
        save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json
    '''

    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

    logger.info(f'json file saved at: {path}')

@ensure_annotations
def load_json(path: Path):
    '''
    Description:
        load json data

    Args:
        path (Path): path to json file
    '''

    with open(path, 'r') as f:
        content = json.load(f)
        
    logger.info(f'json file loaded successfully from: {path}')
    return ConfigBox(content)

### save-load binary

import joblib

@ensure_annotations
def load_bin(path: Path):
    '''
    Description:
        load binary data

    Args:
        path (Path): path to binary file
    '''
    content = joblib.load(path)        
    logger.info(f'binary file loaded successfully from: {path}')
    return ConfigBox(content)

@ensure_annotations
def get_size(path: Path):
    '''
    Description:
        get size in KB

    Args:
        path (Path): path to file

    Returns:
        (str): size in KB
    '''

    size_in_kb = round(os.path.getsize(path)/1024)
    return f'~ {size_in_kb} KB'

### decode-encode image

import base64

def decodeImage(img_string, file_name):
    img_data = base64.b64decode(img_string)

    with open(file_name, 'wb') as f:
        f.write(img_data)

def encodeImageIntoBase64(cropped_image_path):

    with open(cropped_image_path, 'rb') as f:
        return base64.b64encode(f.read())