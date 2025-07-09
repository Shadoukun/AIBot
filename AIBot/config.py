import os
from typing import Any
from ruamel.yaml import YAML

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
yaml = YAML(typ='rt')
yaml.preserve_quotes = True
yaml.default_flow_style = None

def load_config(path=CONFIG_PATH) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.load(f)

def write_config(config_data: dict[str, Any], path=CONFIG_PATH) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(config_data, f)

config = load_config()