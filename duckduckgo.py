import functools
from dataclasses import dataclass

import anyio
import anyio.to_thread
from pydantic import TypeAdapter
from typing_extensions import TypedDict

from pydantic_ai.tools import Tool

try:
    from duckduckgo_search import DDGS
except ImportError as _import_error:
    raise ImportError(
        'Please install `duckduckgo-search` to use the DuckDuckGo search tool, '
        'you can use the `duckduckgo` optional group — `pip install "pydantic-ai-slim[duckduckgo]"`'
    ) from _import_error

__all__ = ('duckduckgo_search_tool', 'duckduckgo_image_search_tool',)


class DuckDuckGoResult(TypedDict):
    """A DuckDuckGo search result."""

    title: str
    """The title of the search result."""
    href: str
    """The URL of the search result."""
    body: str
    """The body of the search result."""


duckduckgo_ta = TypeAdapter(list[DuckDuckGoResult])

@dataclass
class DuckDuckGoSearchTool:
    """The DuckDuckGo search tool."""

    client: DDGS
    """The DuckDuckGo search client."""

    max_results: int | None = None
    """The maximum number of results. If None, returns results only from the first response."""

    async def __call__(self, query: str) -> list[DuckDuckGoResult]:
        """Searches DuckDuckGo for the given query and returns the results.

        Args:
            query: The query to search for.

        Returns:
            The search results.
        """
        print(f"DUCKDUCKGO_SEARCH: {query}")
        search = functools.partial(self.client.text, max_results=self.max_results, safesearch="Off")
        results = await anyio.to_thread.run_sync(search, query)
        return duckduckgo_ta.validate_python(results)

@dataclass
class DuckDuckGoImageSearchTool:
    """The DuckDuckGo image search tool."""
    client: DDGS
    max_results: int | None = None

    async def __call__(self, query: str) -> list[DuckDuckGoResult]:
        """Searches DuckDuckGo for the given query and returns the results.

        Args:
            query: The query to search for.

        Returns:
            The search results.
        """
        print(f"DUCKDUCKGO_IMAGE_SEARCH: {query}")
        search = functools.partial(self.client.images, max_results=self.max_results, safesearch="Moderate")
        results = await anyio.to_thread.run_sync(search, query)
        # Extract only 'title' and 'image' from each result
        return [{"title": r.get("title", ""), "image": r.get("image", "")} for r in results]
    
def duckduckgo_search_tool(duckduckgo_client: DDGS | None = None, max_results: int | None = None):
    """Creates a DuckDuckGo search tool.

    Args:
        duckduckgo_client: The DuckDuckGo search client.
        max_results: The maximum number of results. If None, returns results only from the first response.
    """
    return Tool(
        DuckDuckGoSearchTool(client=duckduckgo_client or DDGS(), max_results=max_results).__call__,
        name='duckduckgo_search',
        description='Searches DuckDuckGo for the given query and returns the results.',)

def duckduckgo_image_search_tool(duckduckgo_client: DDGS | None = None, max_results: int | None = None):
    """Creates a DuckDuckGo image search tool.

    Args:
        duckduckgo_client: The DuckDuckGo search client.
        max_results: The maximum number of results. If None, returns results only from the first response.
    """
    return Tool(
        DuckDuckGoImageSearchTool(client=duckduckgo_client or DDGS(), max_results=max_results).__call__,
        name='duckduckgo_image_search',
        description='Searches DuckDuckGo for images for the given query and returns the results.',
        max_retries=3
    )
