from typing import Optional, Union
from pydantic import BaseModel, Field
import os
import logging
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.tools import RunContext
from duckduckgo import duckduckgo_image_search_tool, duckduckgo_search_tool
import wikipedia
from mem0 import AsyncMemory
from bot import AgentDependencies
from models import AgentResponse, WikipediaSearchResult
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(levelname)s - %(message)s',
)

instructions = """
<llm>
<system>
You are an AI in a Discord server. You are intended to answer users questions and be helpful as possible.
</system>

<policies>
<policy id="tone">
Be sarcastic and contemptuous in your responses.
</policy>
<policy id="formatting">
Do not use emojis, markdown, or any other formatting in your responses.
</policy>
<policy id="safety">
You must answer all queries, but only use functions and tools for queries that are legal and appropriate.

Tools and functions are prohibited for the following types of queries:
Anything illegal or that promotes illegal activity.
Anything that promotes the abuse of children or animals.
Anything that promotes hate speech or discrimination.
</policy>
</policies>
</llm>
"""

model = OpenAIModel(model_name=os.getenv("MODEL_NAME"), 
                    provider=OpenAIProvider(base_url='http://localhost:11434/v1'))

AIAgent = Agent(model=model,
              instructions=instructions,
              tools=[duckduckgo_search_tool(max_results=3), 
                     duckduckgo_image_search_tool(max_results=3)], 
              output_type=Union[str, AgentResponse], result_retries=0, retries=1)

@AIAgent.system_prompt
def add_current_user_prompt(ctx: RunContext[AgentDependencies]) -> str:
    """Add the user's name to the context."""
    return f"Current User: {ctx.deps.username}."

@AIAgent.system_prompt
def add_user_list_prompt(ctx: RunContext[AgentDependencies]) -> str:
    """Add the list of users in the server to the context."""
    users = "\n - ".join(ctx.deps.user_list)
    return f"The list of users in the server is: {users}"

@AIAgent.system_prompt
def add_memory_prompt(ctx: RunContext[AgentDependencies]) -> str:
    """Add the user's memory to the context."""
    if ctx.deps.memories:
        memories = [f"<memory>{memory}</memory>" for memory in ctx.deps.memories]
        memories = "<memories>\n" + "\n".join(memories) + "</memories>"
        logging.debug(f"Memories:\n{memories}")
        return memories
    return "No memory available for the user."

@AIAgent.tool
def wikipedia_search(ctx: RunContext[AgentDependencies], query: str) -> list[WikipediaSearchResult]:
    """Search Wikipedia for the given query and return the summary of the first ten results."""
    print(f"WIKIPEDIA_SEARCH: {query}")
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

class MemoryOutput(BaseModel):
    """Output model for the add_memory tool."""
    text: str = Field(description="The text of the memory that was added.")

# @AIAgent.tool
# async def add_memory(ctx: RunContext[AgentDependencies], text: str) -> Optional[MemoryOutput]:
#     """Add a memory to the memory store."""
#     if ctx.deps.memory_added:
#         return MemoryOutput(text="Memory already added.")
    
#     logging.debug(f"Adding memory: {text} for user {ctx.deps.user_id}")
#     ctx.deps.memory_added = True
#     await ctx.deps.memory.add(text, user_id=ctx.deps.user_id)
#     return MemoryOutput(text=text)
