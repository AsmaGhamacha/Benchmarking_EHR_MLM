import yaml
import os
def load_config():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    config_path = os.path.join(base_dir, "src", "utils", "config", "config.yaml")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)