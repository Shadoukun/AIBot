import os
from typing import Any
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')

def load_config(path=CONFIG_PATH) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

config = load_config()