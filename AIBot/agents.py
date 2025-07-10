import os
import logging
import discord
from duckduckgo_search import DDGS
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.usage import UsageLimits
import wikipedia

from .duckduckgo import duckduckgo_search_tool
from .models import AgentDependencies, AgentResponse, BoolResponse, FactResponse, WikipediaSearchResult
from .prompts import default_system_prompt, search_agent_system_prompt, update_user_prompt, fact_retrieval_system_prompt
from .config import config

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
    "custom_update_memory_prompt": update_user_prompt(),
    "custom_fact_extraction_prompt": fact_retrieval_system_prompt(),
}

# supposedly fixes the 202 rate limit issue for duckduckgo
# if instead of constantly instantiating a new client, you reuse the same one
ddgs_client = DDGS()

local_model = OpenAIModel(model_name=MODEL_NAME,provider=OpenAIProvider(base_url=BASE_URL))
openrouter_model = OpenAIModel(
            'google/gemini-2.5-flash',
            provider=OpenRouterProvider(api_key=os.getenv("OPENROUTER_API_KEY", "")),
        )

main_agent = Agent[AgentDependencies, AgentResponse](
            model=local_model,
            instructions=[default_system_prompt],
            output_type=AgentResponse, 
            deps_type=AgentDependencies, 
        )

search_agent = Agent[AgentDependencies, AgentResponse](
            model=openrouter_model,
            instructions=[search_agent_system_prompt],
            tools=[duckduckgo_search_tool(duckduckgo_client=ddgs_client, max_results=3)],
            deps_type=AgentDependencies,
            output_type=AgentResponse,
        )

memory_agent = Agent[AgentDependencies, FactResponse](
            model=local_model,
            instructions=[fact_retrieval_system_prompt],
            output_type=FactResponse,
            deps_type=AgentDependencies,
        )

true_false_agent = Agent[AgentDependencies, BoolResponse](
            model=local_model,
            instructions=[default_system_prompt],
            output_type=BoolResponse,
            deps_type=AgentDependencies,
        )

@main_agent.tool(retries=0)
async def get_current_user(ctx: RunContext[AgentDependencies]) -> AgentResponse:
    """
    Return the current user information.
    """
    if ctx.deps.username:
        return AgentResponse(content=f"The current user is: {ctx.deps.username} (ID: {ctx.deps.user_id})")
    return AgentResponse(content="No user information available.")

@main_agent.tool(retries=0)
async def get_user_list(ctx: RunContext[AgentDependencies]) -> AgentResponse:
    """
    Return a list of users in the current server.
    """
    if ctx.deps.user_list:
        return AgentResponse(content=f"Users in the server: {', '.join(ctx.deps.user_list)}")
    return AgentResponse(content="No users found.")

@main_agent.tool(retries=0)
async def search(ctx: RunContext[AgentDependencies], query: str) -> AgentResponse:
    """
    Search for the given query using the DuckDuckGo search tool.
    """
    logger.debug(f"Search Query: {query}")
    query = query.strip()

    try:
        results = await search_agent.run(query, deps=ctx.deps, output_type=AgentResponse, 
                                         usage_limits=UsageLimits(request_limit=5, response_tokens_limit=2000))
        if results:
            return results.output
        return AgentResponse(content="No results found.")
    except Exception as e:
        logger.error(f"Error during search: {e}")
        return AgentResponse(content="No results found.")

@search_agent.tool(retries=1)
async def wikipedia_search(ctx: RunContext[AgentDependencies], query: str) -> list[WikipediaSearchResult]:
    """Search Wikipedia for the given query and return the summary of the first ten results."""
    # if ctx.deps.context:
        # embed = discord.Embed(
        #     title="Wikipedia Search",
        #     description=f"{query}",
        #     color=discord.Color.green())
        
        # await ctx.deps.context.send(embed=embed)

    results = []
    search_results = wikipedia.search(query, results=1)
    for title in search_results:
        try:
            # Attempt to get the summary for the title
            summary = wikipedia.summary(title, sentences=6, auto_suggest=False)
        except wikipedia.exceptions.DisambiguationError:
            continue
        except wikipedia.exceptions.PageError:
            continue
        results.append(WikipediaSearchResult(title=title, summary=summary))
    return results