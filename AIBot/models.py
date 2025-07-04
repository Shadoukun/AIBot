from dataclasses import dataclass
from typing import Optional
from pydantic import BaseModel, Field
from mem0 import AsyncMemory
from discord.ext import commands
from discord import TextChannel
from discord.abc import GuildChannel

@dataclass
class AgentDependencies:
    user_list: list[str]
    agent_id: str
    username: str
    user_id: str
    message_id: str
    context:  Optional[commands.Context] = None
    memory: Optional[AsyncMemory] = None
    memories: Optional[list[str]] = None
    memory_added: bool = False
    bot_channel: Optional[GuildChannel] = None


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
    username: str = Field(description="The username of the user who provided the fact.")
    user_id: str = Field(description="The ID of the user who provided the fact.")

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