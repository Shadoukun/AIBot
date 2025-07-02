from typing import Optional
import logging
from pydantic_ai import RunContext
import wikipedia
from bot import AgentDependencies
from models import MemoryOutput, WikipediaSearchResult

logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(levelname)s - %(message)s',
)

async def wikipedia_search(ctx: RunContext[AgentDependencies], query: str) -> list[WikipediaSearchResult]:
    """Search Wikipedia for the given query and return the summary of the first ten results."""
    if ctx.deps.context:
        await ctx.deps.context.send(f"WIKIPEDIA_SEARCH: {query}")

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

async def add_memory(ctx: RunContext[AgentDependencies], text: str) -> Optional[MemoryOutput]:
    """Add a memory to the memory store."""
    if ctx.deps.memory_added:
        return None
    
    logging.debug(f"Adding memory: {text} for user {ctx.deps.user_id}")
    ctx.deps.memory_added = True
    msg = {"role": "user", "content": text}
    await ctx.deps.memory.add(msg, agent_id=ctx.deps.agent_id) # type: ignore
    await ctx.deps.context.send(f"```ADDED_MEMORY: {text}```") # type: ignore
    return MemoryOutput(text=text)
        