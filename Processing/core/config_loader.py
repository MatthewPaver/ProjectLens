import os
import json

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

def load_config():
    config_path = os.path.join(ROOT_DIR, "Processing", "core", "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError("Config file not found at: {}".format(config_path))

    with open(config_path, "r") as f:
        return json.load(f)

def resolve_path(relative_path):
    return os.path.join(ROOT_DIR, relative_path)
