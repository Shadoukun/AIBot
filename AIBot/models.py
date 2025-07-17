from typing import List, Optional, Literal
from pydantic import BaseModel, Field, PositiveInt, ValidationError
from mem0 import AsyncMemory
from discord.ext import commands
from discord.abc import GuildChannel
import json

class JSONBaseModel(BaseModel):
    """Base model that provides a JSON string representation."""
    def __str__(self) -> str:
        """Return a JSON string representation of the model."""
        return json.dumps(self.model_dump_json(), indent=4)

class User(BaseModel):
    """Model for user information"""
    id          : str = Field(..., description="The unique identifier of the user.")
    name        : str = Field(..., description="The name of the user.")
    display_name: str = Field(..., description="The display name of the user.")

    def __str__(self) -> str:
        """Return a string representation of the User."""
        return f"{self.name} (ID: {self.id})"

class AgentDependencies:
    """Dependencies for the main agent, including user and context information."""
    user_list  : Optional[list[User]]
    agent      : Optional[User]
    user       : Optional[User]
    channel    : Optional[GuildChannel]
    message_id : Optional[str] = None
    context    : Optional[commands.Context] = None
    memory     : Optional[AsyncMemory] = None
    memories   : Optional[list[str]] = None
    bot_channel: Optional[GuildChannel] = None
    searches   : List[dict[str, str]] = []

    def __init__(self, bot, ctx: commands.Context, memories: Optional[list[str]] = None):
        self.agent = User(
            id          = str(bot.user.id) if bot.user else "None",
            name        = bot.user.name if bot.user else "None",
            display_name= bot.user.display_name if bot.user else "None"
        )

        self.user = User(
            id          = str(ctx.author.id) if ctx.author else "None",
            name        = ctx.author.name if ctx.author else "None",
            display_name= ctx.author.display_name if ctx.author else "None"
        )

        self.user_list = [User(
            id          = str(member.id),
            name        = member.name,
            display_name= member.display_name
        ) for member in ctx.guild.members if member != ctx.author] if ctx.guild is not None else []

        self.context    = ctx
        self.channel    = ctx.channel # type: ignore
        self.memory     = bot.memory_handler.memory
        self.memories   = memories
        self.message_id = str(ctx.message.id) if ctx.message else "None"
        self.bot_channel= bot.bot_channel if hasattr(bot, 'bot_channel') else None

class BasicResponse(JSONBaseModel):
    """Base model for responses from agents"""
    response: str = Field(..., description="The response content from the agent.")

class FollowUpQuestion(JSONBaseModel):
    """Model for follow-up questions"""
    question: str = Field(..., description="The question to ask the user.")

class SearchResult(JSONBaseModel):
    """Model for individual search results"""
    title  : str = Field(..., description="The title of the search result.")
    url    : str = Field(..., description="The URL of the search result.")
    summary: str = Field(..., description="A brief snippet or summary of the search result.")

class DictSearchResult(JSONBaseModel):
    """Model for individual dictionary results"""
    word      : str           = Field(..., description="The word being defined.")
    definition: str          = Field(..., description="The definition of the word.")
    example   : Optional[str] = Field(None, description="An example usage of the word.")

class SearchResponse(JSONBaseModel):
    """Model for Search Agent Responses"""
    results: list[SearchResult | DictSearchResult] = Field(default_factory=list, description="A list of search results.")

class Fact(JSONBaseModel):
    """Model for individual facts extracted from text"""
    topic  : str = Field(..., description="The topic or subject of the fact. one or two keywords")
    content: str = Field(..., description="The content of the fact.")

class FactResponse(JSONBaseModel):
    """Model for Fact Agent Responses"""
    facts: list[Fact] = Field(default_factory=list, description="A list of facts extracted from the input text.")

class BoolResponse(JSONBaseModel):
    """Model for True/False Agent Responses"""
    result: bool = Field(..., description="a single boolean True or False response.")

class RandomNumberInput(JSONBaseModel):
    """Model for input to generate a random number"""
    start: PositiveInt = Field(1, description="The starting range for the random number (inclusive).")
    limit: PositiveInt = Field(100, description="The upper limit for the random number (inclusive).")

    @classmethod
    def validate(cls, value):
        """Custom validation to ensure start is less than or equal to limit."""
        if value.start > value.limit:
            raise ValidationError("Start must be less than or equal to limit.")
        return value

class RandomNumberResponse(JSONBaseModel):
    """Model for Random Number Responses"""
    number: PositiveInt = Field(..., description="The generated random number.")

class DateTimeResponse(JSONBaseModel):
    """Model for Date and Time Responses"""
    date: str = Field(..., description="The current date in the format 'MM/DD/YYYY'.")
    time: str = Field(..., description="The current time in the format 'HH:MM:SS'.")

class WikipediaSearchResult(JSONBaseModel):
    """Model for individual Wikipedia search results"""
    title  : str = Field(..., description="The title of the Wikipedia page.")
    summary: str = Field(..., description="A brief summary of the Wikipedia page.")

class LookupUrbanDictRequest(JSONBaseModel):
    """Model for requests to look up a term in Urban Dictionary"""
    term: str = Field(..., description="Word or phrase to define (case-insensitive)")

class UrbanDefinition(JSONBaseModel):
    """Model for individual Urban Dictionary definitions"""
    word      : str = Field(..., description="The word being defined.")
    definition: str = Field(..., description="The definition of the word.")

class WikiCrawlRequest(JSONBaseModel):
    """Model for requests to crawl Wikipedia pages"""
    query     : str         = Field(..., description="A page title or a search phrase. If ambiguous, the first result is used unless `exact=True`.")
    depth     : PositiveInt = Field(1, description="How many link-levels deep to crawl. 1 = just the page itself.")
    max_pages : PositiveInt = Field(5, description="Hard cap on total pages visited (safety valve).")
    exact     : bool        = Field(False, description="If True, treat `query` as an exact page title; otherwise use Wikipedia search.")
    intro_only: bool        = Field(False, description="Return only the summary/introduction instead of full content.")

class WikiPage(JSONBaseModel):
    """Model for individual Wikipedia pages"""
    title  : str       = Field(..., description="The title of the Wikipedia page.")
    content: str       = Field(..., description="The main content of the Wikipedia page.")
    url    : str       = Field(..., description="The URL of the Wikipedia page.")
    summary: str       = Field(..., description="A brief summary of the Wikipedia page.")
    links  : List[str] = Field(default_factory=list, description="A list of links found on the Wikipedia page.")

class WikiCrawlResponse(JSONBaseModel):
    """Model for responses from crawling Wikipedia pages"""
    pages        : List[WikiPage] = Field(default_factory=list, description="A list of crawled Wikipedia pages.")
    visited      : int            = Field(..., description="The total number of pages visited during the crawl.")
    depth_reached: int            = Field(..., description="The maximum depth reached during the crawl.")

class CrawlerInput(JSONBaseModel):
    """Model for input to the web crawler"""
    url           : str                        = Field(..., description="Starting URL to crawl")
    depth         : int                        = Field(default=1, description="How deep to crawl")
    extract       : List[Literal["text", "metadata", "links"]] = Field(
        default=["text"], description="What to extract from each page"
    )
    domain_filter  : Optional[List[str]]        = Field(default=None, description="Only include URLs containing these domains")
    include_summary: bool                       = Field(default=True, description="Whether to summarize page content")
    max_pages      : Optional[int]              = Field(default=10, description="Maximum number of pages to crawl")

class PageSummary(JSONBaseModel):
    """Model for summarizing a crawled page"""
    url     : str            = Field(..., description="The URL of the crawled page")
    title   : Optional[str]  = Field(..., description="The title of the crawled page")
    summary : Optional[str]  = Field(..., description="A summary of the crawled page")
    metadata: Optional[dict] = Field(..., description="Metadata extracted from the crawled page")

class CrawlerOutput(JSONBaseModel):
    """Model for output from the web crawler"""
    summary: PageSummary           = Field(
        default_factory=lambda: PageSummary(url="", title="", summary="", metadata={}),
        description="Summary of the crawled page"
    )
    links  : List[dict[str, str]] = Field(default_factory=list, description="List of URLs found during crawling")
