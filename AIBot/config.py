import os
from typing import Any
from ruamel.yaml import YAML

from AIBot.prompts import custom_update_prompt, fact_retrieval_system_prompt

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


MODEL_NAME = config.get("MODEL_NAME", "google/gemini-2.5-flash")
BASE_URL = config.get("BASE_URL", "http://localhost:11434/v1")


# mem0 config
memory_config = {
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "memory",
            "path": "db",
        }
    },
    "llm": {
        "provider": "ollama",
        "config": {
            "model": MODEL_NAME,
            "temperature": 0.8,
            "max_tokens": 2000,
            "ollama_base_url": "http://localhost:11434",
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text:latest",
            "ollama_base_url": "http://localhost:11434",
        },
    },
    "custom_update_memory_prompt": custom_update_prompt(),
    "custom_fact_extraction_prompt": fact_retrieval_system_prompt(),
}
