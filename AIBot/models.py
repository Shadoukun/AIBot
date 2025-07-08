from dataclasses import dataclass
from typing import Optional
from pydantic import BaseModel, Field
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
        self.username = ctx.author.name if ctx.author else "None"
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
    content: str = Field(..., description="The content of the fact.")
    user_id: str = Field(..., description="The ID of the user who provided the fact.")

class FactResponse(BaseModel):
    facts: list[Fact] = Field(default_factory=list, description="A list of facts extracted from the input text.")

    def __str__(self) -> str:
        """Return a string representation of the FactResponse."""
        return ", ".join([f"{fact.content} (User ID: {fact.user_id})" for fact in self.facts]).strip() if self.facts else ""

class WikipediaSearchResult(BaseModel):
    title: str
    summary: str
