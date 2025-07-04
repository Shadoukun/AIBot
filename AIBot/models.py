from dataclasses import dataclass
from typing import Optional
from pydantic import BaseModel, Field
from mem0 import AsyncMemory
from discord.ext import commands
from discord import TextChannel
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
        self.ctx = ctx
        self.memory = bot.memory
        self.memories = memories
        self.message_id = str(ctx.message.id) if ctx.message else "None"
        self.bot_channel = bot.bot_channel if hasattr(bot, 'bot_channel') else None
        
@dataclass
class FactAgentDependencies:
    memory: Optional[AsyncMemory] = None
    memories: Optional[list[str]] = None
    memory_added: bool = False
    bot_channel: Optional[GuildChannel] = None

class MemoryOutput(BaseModel):
            """Output model for the add_memory tool."""
            text: str = Field(description="The text of the memory that was added.")

class WikipediaSearchResult(BaseModel):
    title: str
    summary: str

class Fact(BaseModel):
    text: str = Field(description="The text of the fact.")

class FactResponse(BaseModel):
    save: bool = Field(
        default=True,
        description="Whether to save the facts in the memory."
    )
    facts: list[Fact] = Field(
        default_factory=list,
        description="A list of facts to be saved."
    )

    def __str__(self) -> str:
        """Return a string representation of the FactResponse."""
        return "\n".join(str(fact) for fact in self.facts)

class AgentResponse(BaseModel):
    content: str = Field(description="The main answer to the user's question.")

    def __str__(self) -> str:
        """Return a string representation of the AgentResponse."""
        return self.content.strip() if self.content else ""