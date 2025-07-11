from dataclasses import dataclass
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, PositiveInt, ValidationError
from mem0 import AsyncMemory
from discord.ext import commands
from discord.abc import GuildChannel

class AgentDependencies:
    user_list: Optional[list[str]]
    agent_id: str
    username: str
    user_id: str
    message_id: str
    context:  Optional[commands.Context] = None
    memory: Optional[AsyncMemory] = None
    memories: Optional[list[str]] = None
    memory_added: bool = False
    bot_channel: Optional[GuildChannel] = None

    def __init__(self, bot, ctx: commands.Context, memories: Optional[list[str]] = None):
        self.user_list = [f'"{member.display_name}"' for member in ctx.guild.members if member != ctx.author] # type: ignore
        self.user_id = str(ctx.author.id if ctx.author else "None")
        self.agent_id = str(bot.user.id) if bot.user else "None"
        self.username = ctx.author.display_name if ctx.author else "None"
        self.user_id = str(ctx.author.id) if ctx.author else "None"
        self.context = ctx
        self.memory = bot.memory
        self.memories = memories
        self.message_id = str(ctx.message.id) if ctx.message else "None"
        self.bot_channel = bot.bot_channel if hasattr(bot, 'bot_channel') else None


class AgentResponse(BaseModel):
    content: str = Field(description="The main answer to the user's question.")

    def __str__(self) -> str:
        """Return a string representation of the AgentResponse."""
        return self.content.strip() if self.content else ""

class Fact(BaseModel):
    topic: str = Field(..., description="The topic or subject of the fact. one or two keywords")
    content: str = Field(..., description="The content of the fact.")
    user_id: str = Field(..., description="The ID of the user who provided the fact.")

class BoolResponse(BaseModel):
    result: bool = Field(..., description="a single boolean True or False response.")

class FactResponse(BaseModel):
    facts: list[Fact] = Field(default_factory=list, description="A list of facts extracted from the input text.")

    def __str__(self) -> str:
        """Return a string representation of the FactResponse."""
        return ", ".join([f"{fact.content} (User ID: {fact.user_id})" for fact in self.facts]).strip() if self.facts else ""

class WikipediaSearchResult(BaseModel):
    title: str
    summary: str

class LookupUrbanDictRequest(BaseModel):
    term: str = Field(..., description="Word or phrase to define (case-insensitive)")


class UrbanDefinition(BaseModel):
    word: str
    definition: str
    example: Optional[str] = None
    thumbs_up: int
    thumbs_down: int
    permalink: str

class WikiCrawlRequest(BaseModel):
    query: str = Field(..., description="A page title or a search phrase. If ambiguous, "
                                          "the first result is used unless `exact=True`."
    )

    depth: PositiveInt = Field(1, description="How many link-levels deep to crawl. 1 = just the page itself.")
    max_pages: PositiveInt = Field(10, description="Hard cap on total pages visited (safety valve).")
    exact: bool = Field(False, description="If True, treat `query` as an exact page title; otherwise use Wikipedia search.")
    intro_only: bool = Field(True, description="Return only the summary/introduction instead of full content.")

class WikiPage(BaseModel):
    title: str
    url: str
    summary: str
    links: List[str]


class WikiCrawlResponse(BaseModel):
    pages: List[WikiPage]
    visited: int
    depth_reached: int

class CrawlerInput(BaseModel):
    url: str = Field(..., description="Starting URL to crawl")
    depth: int = Field(default=1, description="How deep to crawl")
    extract: List[Literal["text", "metadata", "links"]] = Field(
        default=["text"], description="What to extract from each page"
    )
    domain_filter: Optional[List[str]] = Field(
        default=None, description="Only include URLs containing these domains"
    )
    include_summary: bool = Field(
        default=True, description="Whether to summarize page content"
    )
    max_pages: Optional[int] = Field(
        default=10, description="Maximum number of pages to crawl"
    )

class PageSummary(BaseModel):
    url: str
    title: Optional[str] = None
    summary: Optional[str] = None
    metadata: Optional[dict] = None


class CrawlerOutput(BaseModel):
    summary: PageSummary = Field(default_factory=PageSummary, description="Summary of the crawled page")
    links: List[dict[str, str]] = Field(default_factory=list, description="List of URLs found during crawling")

class SummarizeInput(BaseModel):
    text: str
