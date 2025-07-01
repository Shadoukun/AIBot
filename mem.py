import os
from mem0.configs.base import MemoryConfig
from dotenv import load_dotenv
load_dotenv()

config = {
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
            "model": os.getenv("MODEL_NAME"),
            "temperature": 0,
            "max_tokens": 2000,
            "ollama_base_url": "http://localhost:11434",  # Ensure this URL is correct
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text:latest",
            # Alternatively, you can use "snowflake-arctic-embed:latest"
            "ollama_base_url": "http://localhost:11434",
        },
    },
}

memory_config = MemoryConfig(**config) # type: ignore