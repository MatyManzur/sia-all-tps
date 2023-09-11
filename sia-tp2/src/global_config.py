import json
from sys import argv

config = json.load(open(argv[1], mode='r'))


def change_config_file(file: str):
    global config
    config = json.load(open(file, mode='r'))
    return config


def get_config():
    return config
