# parse the config.yaml file and return the config dictionary
# check if the config file is valid, if not, raise an error
import os
import yaml
import pathlib
from ._macro import SUPPORTED_ARCHITECTURES
from .cell import SUPPORTED_ARCHITECTURES as CELL_SUPPORTED_ARCHITECTURES
from .cell import SUPPORTED_CLASSIFIER_TYPES as CELL_SUPPORTED_CLASSIFIER_TYPES


def parse_config(config_path):
    """Parse the config.yaml file and return the config dictionary.
    
    Config.yaml format:
        data:
            train_path: path/to/train/dataset
            test_path: path/to/test/dataset
            # if train_path and test_path are not specified, then the dataset will be split into train and test sets
            split_ratio: 0.8 # must be specified if train_path and test_path are not specified
        model:
            arch: [cell|pretrained]
            # if arch is cell, then the model will be trained from scratch
            premodel: [resnet|inception] # must be specified if arch is cell
            # if arch is pretrained, then the model will be finetuned from a pretrained model
            pretrained_path: [example: facebook/dino:resnet50] # must be specified if archi is pretrained
            # for all cases, must be specified
            classifier_type: [mlp|linear]
            model_nickname: [example: mark2] # optional, random string+timestamp if not specified 
        train:
            batch_size: 32
            epochs: 100
            learning_rate: 0.001
            weight_decay: 0.0001
            optimizer: [adam|sgd]
            scheduler: [step|cosine]
            step_size: 30 # must be specified if scheduler is step
            gamma: 0.1 # must be specified if scheduler is step

        

    Returns:
        dict: The config dictionary.
    """
    if not os.path.exists(config_path):
        return ValueError(f"Config file {config_path} does not exist.")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    # check if the config file is valid
    if "data" not in config:
        raise ValueError("Config file must contain 'data' section.")
    if "model" not in config:
        raise ValueError("Config file must contain 'model' section.")
    if "train" not in config:
        raise ValueError("Config file must contain 'train' section.")

def _data_config_check(data_config):
    """Check if the data config is valid."""
    if "train_path" not in data_config and "test_path" not in data_config:
        if "split_ratio" not in data_config:
            raise ValueError("If train_path and test_path are not specified, then split_ratio must be specified.")
        if not (0 < data_config["split_ratio"] < 1):
            raise ValueError("split_ratio must be between 0 and 1.")
    elif "train_path" in data_config and "test_path" in data_config:
        if not os.path.exists(data_config["train_path"]):
            raise ValueError(f"Train dataset path {data_config['train_path']} does not exist.")
        if not os.path.exists(data_config["test_path"]):
            raise ValueError(f"Test dataset path {data_config['test_path']} does not exist.")
    else:
        raise ValueError("Either both train_path and test_path must be specified, or neither of them must be specified.")

def _model_config_check(model_config):
    """Check if the model config is valid."""
    if "arch" not in model_config:
        raise ValueError("Model config must containt 'arch' field.")
    if model_config["arch"] not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"Model arch {model_config['arch']} is not supported. Supported architectures are: {SUPPORTED_ARCHITECTURES}.")
    if model_config["arch"] == "cell":
        if "premodel" not in model_config:
            raise ValueError("If model arch is 'cell', then 'premodel' must be specified.")
        if model_config["premodel"] not in CELL_SUPPORTED_ARCHITECTURES:
            raise ValueError(f"Model premodel {model_config['premodel']} is not supported. Supported premodels are: {CELL_SUPPORTED_ARCHITECTURES}.")
    if "classifier_type" not in model_config:
        raise ValueError("Model config must containt 'classifier_type' field.")