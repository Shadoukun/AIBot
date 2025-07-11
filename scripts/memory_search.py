import argparse
from typing import Any
from mem0 import Memory

MODEL_NAME = "huihui_ai/qwen3-abliterated:8b"  # or any other model you want to use

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
            "temperature": 0.6,
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
}

memory = Memory.from_config(memory_config)

def search_memory(query: str, agent_id: str, limit: int = 10):
    results = memory.search(query=query, agent_id=agent_id, limit=limit)
    return results.get("results", [])

def delete_memories(memories: list[dict[str, Any]]) -> str:
    if not memories:
        return "No memories to delete."
    for memory_entry in memories:
        memory_id, memory_content = memory_entry.get("id"), memory_entry.get("memory")
        if memory_id and memory_content:
            res = memory.delete(memory_id)  # type: ignore
            if res.get("success"):
                print(f"Deleted memory with ID: {memory_id} - {memory_content}")
            else:
                print(f"Failed to delete memory with ID: {memory_id}")
    return "Deletion process completed."

def main():
    parser = argparse.ArgumentParser(description="Memory search and delete utility.")
    parser.add_argument("query", type=str, help="Search query for memory.")
    parser.add_argument("--agent_id", type=str, required=True, help="Agent ID to search/delete memories.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum number of results to return.")
    parser.add_argument("--delete", action="store_true", help="Delete found memories.")
    args = parser.parse_args()

    if args.query:
        res = search_memory(args.query, agent_id=args.agent_id, limit=args.limit)
        for entry in res:
            print(f"ID: {entry['id']}, Memory: {entry['memory']}")
        if args.delete:
            msg = delete_memories(res)
            print(msg)
    else:
        print("Please provide a query")

if __name__ == "__main__":
    main()
