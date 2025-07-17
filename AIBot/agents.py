import logging
from typing import Union
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.common_tools.tavily import tavily_search_tool

from .config import config
from .models import (
    AgentDependencies,
    BasicResponse,
    BoolResponse,
    FollowUpQuestion,
    FactResponse,
    SearchResponse,
)
from .prompts import (
    default_system_prompt,
    search_agent_system_prompt,
    fact_retrieval_system_prompt,
    true_false_system_prompt
)

logger = logging.getLogger(__name__)

MODEL_NAME = config.get("MODEL_NAME", "google/gemini-2.5-flash")
BASE_URL = config.get("BASE_URL", "http://localhost:11434/v1")

# Agent models

local_model = OpenAIModel(model_name=MODEL_NAME,provider=OpenAIProvider(base_url=BASE_URL))

openrouter_config = config.get("openrouter", {})
openrouter_model = OpenAIModel(
            model_name=openrouter_config.get("model", "google/gemini-2.5-flash"),
            provider=OpenRouterProvider(api_key=openrouter_config.get("api_key", "")),
        )

# Agents

OutputType = Union[FollowUpQuestion, BasicResponse]

# Main agent used for supervising the other agents
main_agent = Agent(
            model=local_model,
            instructions=[default_system_prompt],
            deps_type=AgentDependencies,
            output_type=OutputType, # type: ignore
        )

SearchOutputType = Union[FollowUpQuestion, SearchResponse]

# Search agent for handling search queries
search_agent = Agent[None, SearchOutputType](
            model=openrouter_model,
            instructions=[search_agent_system_prompt],
            tools=[tavily_search_tool(config.get("TAVILY_API_KEY"))], # type: ignore
        )

# Memory agent for handling fact retrieval and memory updates
memory_agent = Agent[None, FactResponse](
            model=local_model,
            instructions=[fact_retrieval_system_prompt],
            output_type=FactResponse,
        )

# Boolean response agent for true/false questions
true_false_agent = Agent[None, BoolResponse](
            model=local_model,
            instructions=[true_false_system_prompt],
            output_type=BoolResponse,
        )

# Summary agent for summarizing text
summary_agent = Agent[None, str](
    model=openrouter_model,
    instructions=["You are a summarization agent. Your only task is to summarize the provided text."],
    output_type=str,
)