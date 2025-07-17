import asyncio
import logging
import random
from datetime import datetime
from typing import List, Set

from pydantic_graph import End
import wikipedia
from crawl4ai import AsyncWebCrawler
from pyurbandict import UrbanDict
from pydantic_ai import RunContext
from pydantic_ai.usage import UsageLimits

from .agents import main_agent, search_agent, summary_agent
from .crawler import browser_cfg, run_config
from .models import (
    AgentDependencies,
    CrawlerInput,
    CrawlerOutput,
    DateTimeResponse,
    LookupUrbanDictRequest,
    PageSummary,
    RandomNumberInput,
    RandomNumberResponse,
    UrbanDefinition,
    User,
    WikiCrawlRequest,
    WikiCrawlResponse,
    WikiPage,
)

logger = logging.getLogger(__name__)

@main_agent.tool
async def get_current_user(ctx: RunContext[AgentDependencies]) -> User:
    """
    Retrieves the current user information from the provided context.

    Args:
        ctx (RunContext[AgentDependencies]): The runtime context containing agent dependencies.

    Returns:
        User: The current user object if available; otherwise, a default User instance with placeholder values.

    Example:
        user = await get_current_user(ctx)
        print(user.id, user.name)

    Notes:
        - If no user is present in the context dependencies, returns a User object with id="None" and name/display_name set to "No user".
    """
    return ctx.deps.user if ctx.deps.user else User(id="", name="", display_name="")


@main_agent.tool
async def get_user_list(ctx: RunContext[AgentDependencies]) -> List[User]:
    """
    Returns a list of users in the current server.

    This asynchronous function retrieves the list of users available in the current context.

    Returns:
        List[User]: A list of User objects representing users in the server.
    """
    return ctx.deps.user_list if ctx.deps.user_list else []


@main_agent.tool_plain
async def get_current_date() -> DateTimeResponse:
    """
    Returns the current date and time.

    This asynchronous function retrieves the current date and time, formats them,
    and returns a DateTimeResponse containing the formatted date and time.

    Returns:
        DateResponse: An object containing the current date in the format "MM/DD/YYYY" and the current time in the format "HH:MM:SS".
    """
    date = datetime.now().strftime("%m/%d/%Y")
    time = datetime.now().strftime("%H:%M:%S")
    return DateTimeResponse(date=date, time=time)


@main_agent.tool_plain
async def random_number(rand: RandomNumberInput) -> RandomNumberResponse:
    """
    Generate a random integer within a specified range.
    Args:
        rand (RandomNumberInput): Input object containing 'start' and 'limit' attributes defining the range.
    Returns:
        RandomNumberResponse: Response object containing the generated random integer.
    Raises:
        ValueError: If 'start' is greater than 'limit'.
    Example:
        >>> random_number(RandomNumberInput(start=1, limit=10))
        RandomNumberResponse(number=7)
    """
    number = random.randint(rand.start, rand.limit)
    return RandomNumberResponse(number=number)


@main_agent.tool_plain(retries=1)
async def search(query: str) -> str:
    """
    Performs a search online for the given query using the search agent.

    Args:
        query (str): The search query string.

    Returns:
        str: The search results as a string. (at the moment)

    Raises:
        Exception: If an error occurs during the search process.
    """
    logger.debug(f"Search Query: {query}")
    query = query.strip()
    try:
        agent_run = await search_agent.run(query, usage_limits=UsageLimits(request_limit=10))
        if agent_run and agent_run.output:
            return str(agent_run.output)
        else:
            return "No results found."
        
        
    except Exception:
        return "Error during search."


@search_agent.tool_plain
async def urbandictionary_lookup(req: LookupUrbanDictRequest) -> list[UrbanDefinition]:
    """
    Looks up the given term on Urban Dictionary and returns a list of definitions.
    Args:
        req (LookupUrbanDictRequest): The request containing the term to look up.
    Returns:
        list[UrbanDefinition]: A list of UrbanDefinition objects for the term.
    Raises:
        ValueError: If no definitions are found for the term.
    """
    entries = UrbanDict(req.term).search()
    if not entries:
        raise ValueError(f"No Urban Dictionary results for “{req.term}”.")
    return [
        UrbanDefinition(
            word=e.word,
            definition=e.definition,
        )
        for e in entries[:10]
    ]


@search_agent.tool_plain
async def search_wikipedia(req: WikiCrawlRequest) -> WikiCrawlResponse:
    """
    Crawl Wikipedia starting from a query/title, up to `depth` link-levels.
    Returns a WikiCrawlResponse containing WikiPage objects for each visited page.
    Args:
        req (WikiCrawlRequest): The crawl request parameters.
    Returns:
        WikiCrawlResponse: The crawl results.
    Raises:
        ValueError: If no results are found for the query.
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
            page = _fetch_wiki_page(title, req.intro_only)
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
    
    return WikiCrawlResponse(
        pages=pages_out,
        visited=len(visited),
        depth_reached=min(req.depth, max((d for _, d in queue), default=0)),
    )


@search_agent.tool_plain
async def crawl_page(input: CrawlerInput) -> CrawlerOutput:
    """
    Crawls a web page and returns a summary of the discovered pages.

    Args:
        input (CrawlerInput): The input parameters for crawling, including the URL, depth, 
        extraction options, domain filters, maximum pages, and summary inclusion.
    Returns:
        CrawlerOutput: An object containing a list of PageSummary instances for each crawled page, 
        including URL, title, summary, and metadata.
    Raises:
        Exception: If summarization of a page's text fails, the summary will be set to "Summary failed."
    Example:
        input = CrawlerInput(
            url="https://example.com",
            depth=2,
            extract=["text", "title"],
            max_pages=10,
            domain_filter=["example.com"],
            include_summary=True
    """

    # so much ignore

    async with AsyncWebCrawler(config=browser_cfg, run_config=run_config) as crawler:
        crawl_result = await crawler.arun(input.url)

    if not crawl_result.success: # type: ignore
        logger.debug(f"Crawl failed: {crawl_result.error_message}") # type: ignore
    
    links = []
    for link in crawl_result.links['internal']: # type: ignore
        if input.domain_filter and not any(domain in link for domain in input.domain_filter):
            continue
        else:
            links.append(link)

    summary = None
    if input.include_summary and crawl_result.markdown: # type: ignore
        logger.debug("Summarizing page content...")
        try:
            res = await summary_agent.run(crawl_result.markdown.fit_markdown) # type: ignore
            summary = res.output if res else "Summary failed."
        except Exception as e:
            logger.error(f"Error summarizing page content: {e}")
            summary = "Summary failed."

    output = CrawlerOutput(summary=PageSummary(
        url=input.url,
        summary=summary if summary else "No summary available.",
        metadata=crawl_result.metadata # type: ignore
    ), links=links)
    
    logger.debug(f"Crawled {len(links)} links from {input.url}")   
    return output

def _fetch_wiki_page(title: str, intro_only: bool) -> WikiPage:
    """Helper that grabs a page and returns our WikiPage model."""
    page = wikipedia.page(title, auto_suggest=False)
    text = page.summary if intro_only else page.content
    return WikiPage(
        title=page.title,
        url=page.url,
        summary=text,
        links=page.links
    )