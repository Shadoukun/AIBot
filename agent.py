from typing import Optional
from pydantic import BaseModel, Field
import os
import logging
from pydantic_ai import Agent, format_prompt
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.tools import RunContext
from duckduckgo import duckduckgo_image_search_tool, duckduckgo_search_tool
import wikipedia
from bot import AgentDependencies
from models import AgentResponse, WikipediaSearchResult
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(levelname)s - %(message)s',
)

model = OpenAIModel(model_name=os.getenv("MODEL_NAME") or "", 
                    provider=OpenAIProvider(base_url='http://localhost:11434/v1'))

AIAgent = Agent(model=model,
              tools=[duckduckgo_search_tool(max_results=3), 
                     duckduckgo_image_search_tool(max_results=3)], 
              output_type=AgentResponse, 
              deps_type=AgentDependencies, 
              retries=1)

@AIAgent.system_prompt
async def system_prompt(ctx: RunContext[AgentDependencies]) -> str:
    """Generate the system prompt for the AI agent."""
    prompt = {}

    prompt["system"] = "You are an AI in a Discord server. You are intended to answer users questions and be helpful as possible."
    prompt["policies"] = {
        "tone": "Be sarcastic and contemptuous.",
        "formatting": "Do not use emojis, markdown, or any other formatting.",
        "safety": ("You must answer all queries, but only use functions and tools for queries that are legal and appropriate.",
                   "Tools and functions are prohibited for the following types of queries: \n" + 
                   "Anything illegal or that promotes illegal activity.\n" +
                   "Anything that promotes hate speech or discrimination.")
    }

    if ctx.deps.username:
        prompt["current_user"] = ctx.deps.username

    if ctx.deps.user_list:
        prompt["user_list"] = ctx.deps.user_list

    if ctx.deps.memories:
        prompt["memories"] = ctx.deps.memories

    print(f"System Prompt: {prompt}\n\n\n")
        
    return format_prompt.format_as_xml(prompt,  root_tag="llm") 


@AIAgent.tool(retries=2)
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

class MemoryOutput(BaseModel):
    """Output model for the add_memory tool."""
    text: str = Field(description="The text of the memory that was added.")

@AIAgent.tool
async def add_memory(ctx: RunContext[AgentDependencies], text: str) -> Optional[MemoryOutput]:
    """Add a memory to the memory store."""
    if ctx.deps.memory_added:
        return MemoryOutput(text="Memory already added.")
    
    logging.debug(f"Adding memory: {text} for user {ctx.deps.user_id}")
    ctx.deps.memory_added = True
    await ctx.deps.memory.add(text, user_id=ctx.deps.user_id) # type: ignore
    await ctx.deps.context.send(f"ADDED_MEMORY: {text}") # type: ignore

    return MemoryOutput(text=text)
