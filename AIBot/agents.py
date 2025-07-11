import os
import logging
import discord
from typing import List, Set
from duckduckgo_search import DDGS
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.usage import UsageLimits
import wikipedia
from pyurbandict import UrbanDict

from .duckduckgo import duckduckgo_search_tool
from .models import AgentDependencies, AgentResponse, BoolResponse, FactResponse, WikipediaSearchResult, UrbanDefinition, LookupUrbanDictRequest, WikiPage, WikiCrawlRequest, WikiCrawlResponse
from .prompts import default_system_prompt, search_agent_system_prompt, custom_update_prompt, fact_retrieval_system_prompt
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
    "custom_update_memory_prompt": custom_update_prompt(),
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
    Search for the given query online using the search agent.
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

@search_agent.tool(retries=0)
def urbandictionary_lookup(ctx: RunContext[AgentDependencies], req: LookupUrbanDictRequest) -> list[UrbanDefinition]:
    """
    Look up a term on Urban Dictionary.
    Raises ValueError if nothing is found.
    """
    entries = UrbanDict(req.term).search()
    if not entries:
        raise ValueError(f"No Urban Dictionary results for “{req.term}”.")
    return [
        UrbanDefinition(
            word=e.word,
            definition=e.definition,
            example=e.example,
            thumbs_up=e.thumbs_up,
            thumbs_down=e.thumbs_down,
            permalink=e.permalink,
        )
        for e in entries[:10]
    ]

@search_agent.tool
def crawl_wikipedia(ctx: RunContext[AgentDependencies], req: WikiCrawlRequest) -> WikiCrawlResponse:
    """
    Crawl Wikipedia starting from a query/title, up to `depth`
    link-levels,. Returns titles, URLs, summaries,
    and outgoing links for each visited page.
    """

    visited: Set[str] = set()
    queue: List[tuple[str, int]] = []  # (title, depth_so_far)

    # Resolve initial page(s)
    try:
        if req.exact:
            queue.append((req.query, 0))
        else:
            best_title = wikipedia.search(req.query, results=1)[0]
            queue.append((best_title, 0))
    except IndexError:
        raise ValueError(f"No results for query: {req.query}")

    logger.debug(f"Starting Wikipedia crawl with query: {req.query}, depth: {req.depth}, max pages: {req.max_pages}")
    pages_out: List[WikiPage] = []

    while queue and len(visited) < req.max_pages:
        title, d = queue.pop(0)
        # skip if already visited or depth exceeded
        if title in visited or d > req.depth:
            continue
        try:
            page = _fetch_page(title, req.intro_only)
        except wikipedia.DisambiguationError as e:
            # pick the first suggested page to keep it deterministic
            if e.options:
                queue.append((e.options[0], d))
            continue
        except wikipedia.PageError:
            continue  # skip dead links
        pages_out.append(page)
        visited.add(title)

        # Enqueue links for next layer
        if d + 1 <= req.depth:
            for link_title in page.links:
                if link_title not in visited:
                    queue.append((link_title, d + 1))

    logger.debug(f"Visited {len(visited)} pages, depth reached: {min(req.depth, max((d for _, d in queue), default=0))}")
    logger.debug([p.summary for p in pages_out])
    
    return WikiCrawlResponse(
        pages=pages_out,
        visited=len(visited),
        depth_reached=min(req.depth, max((d for _, d in queue), default=0)),
    )


def _fetch_page(title: str, intro_only: bool) -> WikiPage:
    """Helper that grabs a page and returns our WikiPage model."""
    page = wikipedia.page(title, auto_suggest=False)
    text = page.summary if intro_only else page.content
    return WikiPage(
        title=page.title,
        url=page.url,
        summary=text,
        links=page.links
    )