from asyncio import run
import html
import logging
import random
from datetime import datetime
from typing import List, Set

from atproto import AtUri
import discord
from pydantic_graph import End
import wikipedia
from crawl4ai import AsyncWebCrawler
from pyurbandict import UrbanDict
from pydantic_ai import RunContext
from pydantic_ai.usage import UsageLimits
from pydantic_ai import CallToolsNode
from pydantic_ai.messages import FunctionToolCallEvent

from .agents import SearchOutputType, main_agent, search_agent, summary_agent
from .crawler import browser_cfg, run_config
import re
from .models import (
    AgentDependencies,
    BlueSkyPost,
    CrawlerInput,
    CrawlerOutput,
    DateTimeResponse,
    LookupUrbanDictRequest,
    PageSummary,
    RandomNumberInput,
    RandomNumberResponse,
    SearchResponse,
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


@main_agent.tool(retries=1)
async def search(ctx: RunContext[AgentDependencies], query: str) -> SearchOutputType:
    """
    Performs a search online for the given query using the search agent.

    Args:
        query (str): The search query string.

    Returns:
        str: The search results as a string. # at the moment

    Raises:
        Exception: If an error occurs during the search process.
    """
    logger.debug(f"Search Query: {query}")

    # Avoid running the same search multiple times
    search = next((s for s in ctx.deps.searches if s.get("query") == query.lower()), None)
    print(search)
    if search:
        logger.debug(f"Skipping duplicate search: {query}")
        return search["response"] + "\n\n You already searched for this query. You should finish up the reqest."

    try:
        search = {}
        searches: List[dict[str, str]] = []
        search_usage_limits = UsageLimits(request_limit=20, response_tokens_limit=2000)
        # try to run the search agent with the given query
        async with search_agent.iter(query, deps=ctx.deps, usage_limits=search_usage_limits) as agent_run:
            node = agent_run.next_node
            while not isinstance(node, End):

                # Iterate over tool calls
                if isinstance(node, CallToolsNode):
                    async with node.stream(agent_run.ctx) as handle_stream:
                        async for event in handle_stream:
                            if isinstance(event, FunctionToolCallEvent):
                                tool_name = event.part.tool_name
                                text = str(node.model_response.parts[0])
                                search = {"tool_name": tool_name, "query": text}

                                # Check if the search is already in searches
                                for s in searches:
                                    if s.get("tool_name") == tool_name and s.get("query") == text:
                                        logger.debug(f"Skipping duplicate tool call: {tool_name} : {text}")
                                        break
                                searches.append(search)

                                # Set bot status if not already busy
                                msg = ""
                                if tool_name == "tavily_search_tool":
                                    msg = "Searching Tavily..."
                                elif tool_name == "urbandictionary_lookup":
                                    msg = "Searching Urban Dictionary..."
                                elif tool_name == "search_wikipedia":
                                    msg = "Searching Wikipedia..."
                                if msg and ctx.deps.bot.status != discord.Status.dnd:
                                    await ctx.deps.bot.change_presence( # type: ignore
                                        activity=discord.CustomActivity(name=msg), status=discord.Status.online) # type: ignore

                node = await agent_run.next(node)

            if agent_run.result:
                # append the search result to the context's searches
                ctx.deps.searches.append({"query": query.lower(),"response": agent_run.result.output}) # type: ignore
                return agent_run.result.output
            else:
                return SearchResponse(results=[])

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return SearchResponse(results=[])

@search_agent.tool
async def get_bluesky_post(ctx: RunContext[AgentDependencies], url: str) -> BlueSkyPost:
    """
    Retrieves a BlueSky post by its URL.

    Args:
        ctx (RunContext[AgentDependencies]): The context containing dependencies for the agent.
        url (str): The URL of the BlueSky post to retrieve.
    Returns:
        BlueSkyPost: An object containing the username, content, and URL of the post.
    Raises:
        ValueError: If the post is not found for the given URL.
    """
    if not ctx.deps.atproto_client:
        raise ValueError("ATProto client is not available in the context dependencies.")

    if match := re.search(r"https://bsky\.app/profile/([^/]+)/post/([^/]+)", url):
        username = match.group(1)
        post_id = match.group(2)
        POST_URI = f"at://{username}/app.bsky.feed.post/{post_id}"
        post = ctx.deps.atproto_client.app.bsky.feed.post.get(username, AtUri.from_str(POST_URI).rkey)
        if not post:
            raise ValueError(f"Post not found for URL: {url}")

        return BlueSkyPost(
            username=username,
            # remove escaped \n characters from the content
            content=post.value.text.replace("\\n", "\n"),
            url=url,
        )
    else:
        raise ValueError(f"Invalid BlueSky URL: {url}")

@search_agent.tool_plain
async def urbandictionary_lookup(req: LookupUrbanDictRequest) -> list[UrbanDefinition]:
    """
    Looks up the given term on Urban Dictionary and returns a list of definitions.
    Only use for dictionary-style lookups, not for general search queries.
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
        for e in entries[:2]
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
        links=page.links,
        content=page.content
    )