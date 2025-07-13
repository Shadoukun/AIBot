from typing import List, Optional, Literal
from pydantic import BaseModel, Field, PositiveInt, ValidationError
from mem0 import AsyncMemory
from discord.ext import commands
from discord.abc import GuildChannel

class User(BaseModel):
    """Model for user information"""
    id: str = Field(..., description="The unique identifier of the user.")
    name: str = Field(..., description="The name of the user.")
    display_name: str = Field(..., description="The display name of the user.")

    def __str__(self) -> str:
        """Return a string representation of the User."""
        return f"{self.name} (ID: {self.id})"

class AgentDependencies:
    """Dependencies for the main agent, including user and context information."""
    user_list: Optional[list[User]]
    agent: Optional[User]
    user: Optional[User]
    channel: Optional[GuildChannel]
    message_id: Optional[str] = None
    context:  Optional[commands.Context] = None
    memory: Optional[AsyncMemory] = None
    memories: Optional[list[str]] = None
    memory_added: bool = False
    bot_channel: Optional[GuildChannel] = None

    def __init__(self, bot, ctx: commands.Context, memories: Optional[list[str]] = None):
        self.agent = User(
            id=str(bot.user.id) if bot.user else "None",
            name=bot.user.name if bot.user else "None",
            display_name=bot.user.display_name if bot.user else "None"
        )

        self.user = User(
            id=str(ctx.author.id) if ctx.author else "None",
            name=ctx.author.name if ctx.author else "None",
            display_name=ctx.author.display_name if ctx.author else "None"
        )

        self.user_list = [User(
            id=str(member.id),
            name=member.name,
            display_name=member.display_name
        ) for member in ctx.guild.members if member != ctx.author] if ctx.guild is not None else []

        self.context = ctx
        self.memory = bot.memory
        self.memories = memories
        self.message_id = str(ctx.message.id) if ctx.message else "None"
        self.bot_channel = bot.bot_channel if hasattr(bot, 'bot_channel') else None

class AgentResponse(BaseModel):
    """Model for Main Agent Responses"""
    content: str = Field(description="The main answer to the user's question.")

    def __str__(self) -> str:
        """Return a string representation of the AgentResponse."""
        return self.content.strip() if self.content else ""

class SearchResult(BaseModel):
    """Model for individual search results"""
    title: str = Field(..., description="The title of the search result.")
    url: str = Field(..., description="The URL of the search result.")
    snippet: str = Field(..., description="A brief snippet or summary of the search result.")

    def __str__(self) -> str:
        """Return a string representation of the SearchResult."""
        return f"<title>{self.title}</title>\n<url>({self.url})</url>\n<summary>{self.snippet}</summary>"

class SearchResponse(BaseModel):
    """Model for Search Agent Responses"""
    results: list[SearchResult] = Field(default_factory=list, description="A list of search results.")

class Fact(BaseModel):
    """Model for individual facts extracted from text"""
    topic: str = Field(..., description="The topic or subject of the fact. one or two keywords")
    content: str = Field(..., description="The content of the fact.")
    user_id: str = Field(..., description="The ID of the user who provided the fact.")

class FactResponse(BaseModel):
    """Model for Fact Agent Responses"""
    facts: list[Fact] = Field(default_factory=list, description="A list of facts extracted from the input text.")

    def __str__(self) -> str:
        """Return a string representation of the FactResponse."""
        return ", ".join([f"{fact.content} (User ID: {fact.user_id})" for fact in self.facts]).strip() if self.facts else ""

class BoolResponse(BaseModel):
    """Model for True/False Agent Responses"""
    result: bool = Field(..., description="a single boolean True or False response.")

class RandomNumberInput(BaseModel):
    """Model for input to generate a random number"""
    start: PositiveInt = Field(1, description="The starting range for the random number (inclusive).")
    limit: PositiveInt = Field(100, description="The upper limit for the random number (inclusive).")

    @classmethod
    def validate(cls, value):
        """Custom validation to ensure start is less than or equal to limit."""
        if value.start > value.limit:
            raise ValidationError("Start must be less than or equal to limit.")
        return value

class RandomNumberResponse(BaseModel):
    """Model for Random Number Responses"""
    number: PositiveInt = Field(..., description="The generated random number.")

    def __str__(self) -> str:
        """Return a string representation of the RandomNumberResponse."""
        return f"Your random number is: {self.number}"
    
class DateTimeResponse(BaseModel):
    """Model for Date and Time Responses"""
    date: str = Field(..., description="The current date in the format 'MM/DD/YYYY'.")
    time: str = Field(..., description="The current time in the format 'HH:MM:SS'.")

    def __str__(self) -> str:
        """Return a string representation of the DateResponse."""
        return f"The current date is {self.date}. The current time is {self.time}."
    
class WikipediaSearchResult(BaseModel):
    """Model for individual Wikipedia search results"""
    title: str
    summary: str

class LookupUrbanDictRequest(BaseModel):
    """Model for requests to look up a term in Urban Dictionary"""
    term: str = Field(..., description="Word or phrase to define (case-insensitive)")

class UrbanDefinition(BaseModel):
    """Model for individual Urban Dictionary definitions"""
    word: str
    definition: str
    example: Optional[str] = None
    thumbs_up: int
    thumbs_down: int
    permalink: str

class WikiCrawlRequest(BaseModel):
    """Model for requests to crawl Wikipedia pages"""
    query: str = Field(..., description="A page title or a search phrase. If ambiguous, "
                                          "the first result is used unless `exact=True`."
    )

    depth: PositiveInt = Field(1, description="How many link-levels deep to crawl. 1 = just the page itself.")
    max_pages: PositiveInt = Field(10, description="Hard cap on total pages visited (safety valve).")
    exact: bool = Field(False, description="If True, treat `query` as an exact page title; otherwise use Wikipedia search.")
    intro_only: bool = Field(True, description="Return only the summary/introduction instead of full content.")

class WikiPage(BaseModel):
    """Model for individual Wikipedia pages"""
    title: str
    url: str
    summary: str
    links: List[str]

class WikiCrawlResponse(BaseModel):
    """Model for responses from crawling Wikipedia pages"""
    pages: List[WikiPage]
    visited: int
    depth_reached: int

class CrawlerInput(BaseModel):
    """Model for input to the web crawler"""
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
    """Model for summarizing a crawled page"""
    url: str
    title: Optional[str] = None
    summary: Optional[str] = None
    metadata: Optional[dict] = None

class CrawlerOutput(BaseModel):
    """Model for output from the web crawler"""
    summary: PageSummary = Field(default_factory=lambda: PageSummary(url=""), description="Summary of the crawled page")
    links: List[dict[str, str]] = Field(default_factory=list, description="List of URLs found during crawling")

class SummarizeInput(BaseModel):
    """Model for input to the summarization agent"""
    text: str
