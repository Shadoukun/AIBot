from typing import Optional
import logging
import discord
from pydantic_ai import RunContext
import wikipedia
from .bot import AgentDependencies
from .models import MemoryOutput, WikipediaSearchResult

async def wikipedia_search(ctx: RunContext[AgentDependencies], query: str) -> list[WikipediaSearchResult]:
    """Search Wikipedia for the given query and return the summary of the first ten results."""
    if ctx.deps.context:
        embed = discord.Embed(
            title="Wikipedia Search",
            description=f"{query}",
            color=discord.Color.green())
        
        await ctx.deps.context.send(embed=embed)

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
