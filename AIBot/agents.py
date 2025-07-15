import os
import logging
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.common_tools.tavily import tavily_search_tool

from .config import config
from .models import (
    AgentDependencies,
    AgentResponse,
    BoolResponse,
    FactResponse,
    SearchResponse,
)
from .prompts import (
    default_system_prompt,
    search_agent_system_prompt,
    fact_retrieval_system_prompt,
    custom_update_prompt
)

logger = logging.getLogger(__name__)

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
    "custom_update_memory_prompt": custom_update_prompt(),
    "custom_fact_extraction_prompt": fact_retrieval_system_prompt(),
}

# Agent models

local_model = OpenAIModel(model_name=MODEL_NAME,provider=OpenAIProvider(base_url=BASE_URL))
openrouter_model = OpenAIModel(
            'google/gemini-2.5-flash',
            provider=OpenRouterProvider(api_key=config.get("OPENROUTER_API_KEY", "")),
        )

# Agents

# Main agent used for supervising the other agents
main_agent = Agent[AgentDependencies, AgentResponse](
            model=local_model,
            instructions=[default_system_prompt],
            output_type=AgentResponse, 
            deps_type=AgentDependencies, 
        )

# Search agent for handling search queries
search_agent = Agent[AgentDependencies, SearchResponse](
            model=openrouter_model,
            instructions=[search_agent_system_prompt],
            tools=[tavily_search_tool(config.get("TAVILY_API_KEY"))], # type: ignore
            deps_type=AgentDependencies,
            output_type=SearchResponse,
        )

# Memory agent for handling fact retrieval and memory updates
memory_agent = Agent[AgentDependencies, FactResponse](
            model=local_model,
            instructions=[fact_retrieval_system_prompt],
            output_type=FactResponse,
            deps_type=AgentDependencies,
        )

# Boolean response agent for true/false questions
true_false_agent = Agent[AgentDependencies, BoolResponse](
            model=local_model,
            instructions=[default_system_prompt],
            output_type=BoolResponse,
            deps_type=AgentDependencies,
        )

# Summary agent for summarizing text
summary_agent = Agent[None, str](
    model=openrouter_model,
    instructions=["You are a summarization agent. Your only task is to summarize the provided text."],
    output_type=str,
    deps_type=type(None),
)