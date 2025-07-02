from dataclasses import dataclass
from typing import Optional
from pydantic import BaseModel, Field
from mem0 import AsyncMemory
from discord.ext import commands


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

class MemoryOutput(BaseModel):
            """Output model for the add_memory tool."""
            text: str = Field(description="The text of the memory that was added.")

class WikipediaSearchResult(BaseModel):
    title: str
    summary: str

class AgentResponse(BaseModel):
    content: str = Field(description="The main answer to the user's question.")

    def __str__(self) -> str:
        """Return a string representation of the AgentResponse."""
        return self.content.strip() if self.content else ""