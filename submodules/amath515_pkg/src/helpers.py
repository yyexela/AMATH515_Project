###############################
# Imports # Imports # Imports #
###############################

import os
import yaml
import pathlib
import sys

# Add custom package to import path and import it
file_dir = pathlib.Path().resolve()
pkg_dir = os.path.join(file_dir, "submodules")
sys.path.insert(0, pkg_dir)

# Load minGTP
from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import CharDataset, CfgNode as CN

#######################################
# Config Functions # Config Functions #
#######################################

def get_loader() -> yaml.SafeLoader:
    """
    Makes it so we can put directories and sub-directories in YAML

    ex:
    dataset_dir: &dataset_dir "/Datasets"
    original_data_dir: !join [*dataset_dir, "Original"]

    config['original_data_dir'] yields "/Datasets/Original"
    """
    loader = yaml.SafeLoader

    # define custom tag handler
    # (https://stackoverflow.com/questions/5484016/how-can-i-do-string-concatenation-or-string-replacement-in-yaml)
    def join(loader, node):
        seq = loader.construct_sequence(node)
        return os.path.join(*seq)

    ## register the tag handler
    loader.add_constructor('!join', join)

    return loader

def init_config() -> dict:
    """
    Initializes global dictionary of yaml config file
    Reads `../config/config.yaml` and sets it to the global `config` dictionary,
    which is returned in subsequent `get_config` calls
    """
    global config

    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'config', "config.yaml"), 'r') as config_file:
        config = yaml.load(config_file, Loader=get_loader())

    return config

def update_config(new_path: list[str]) -> dict:
    """
    Updates global dictionary of to new yaml config file from specified path `new_path`
    
    Tested
    """
    global config
    with open(new_path, 'r') as config_file:
        test_config = yaml.load(config_file, Loader=get_loader())
    # Update individual keys, keeping old ones
    for key in list(test_config.keys()):
        config[key] = test_config[key]
    return config

def get_config() -> dict:
    """
    Returns current global config dictionary
    This is initialized with `init_config` to the default config file and
    updated with `update_config`

    Tested
    """
    global config
    return config

###################################
# minGPT Helpers # minGPT Helpers #
###################################

def get_minGPT_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = os.path.join(file_dir, 'Saved_Models')

    # data
    C.data = CharDataset.get_default_config()

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-mini'

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster

    return C
