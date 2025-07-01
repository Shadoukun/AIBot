from dataclasses import dataclass
from pydantic import BaseModel, Field
from mem0 import AsyncMemory

@dataclass
class AgentDependencies:
    user_list: list[str]
    username: str
    user_id: str
    message_id: str = None
    memory: AsyncMemory = None
    memories: list[str] = None
    memory_added: bool = False

class WikipediaSearchResult(BaseModel):
    title: str
    summary: str

class AgentResponse(BaseModel):
    content: str = Field(description="The main answer to the user's question.")
    tools: list[str] = Field(description="A list of tools used to generate the answer.")

    def __str__(self) -> str:
        """Return a string representation of the AgentResponse."""
        return self.content.strip() if self.content else ""