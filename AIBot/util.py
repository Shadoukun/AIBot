import discord
import logging
from typing import Any, Callable
from datetime import datetime, timedelta, timezone
from mem0 import AsyncMemory
from pydantic_ai import Agent

from .prompts import memory_prompt
from .models import AgentDependencies, AgentResponse, FactResponse

logger = logging.getLogger(__name__)

class AgentUtilities:
    """ A mixin class that provides utility methods for AIBot"""

    # attributes that the main bot class will have
    user: discord.User
    watched_channels: list[int]
    command_prefix: str
    agent: Agent[AgentDependencies, AgentResponse]
    memory_agent: Agent[AgentDependencies, FactResponse]
    memory: AsyncMemory

    get_channel: Callable[[int], discord.abc.GuildChannel | None]

    @staticmethod
    def remove_command_prefix(msg, prefix='!') -> str:
        """
        Remove the command prefix from the message.
        """
        if msg.startswith(prefix):
            msg = " ".join(msg.split()[1:])
        return msg

    @staticmethod
    def update_message_history(history: list[str], new: list[str], max_length: int = 20) -> list[str]:
        """
        Trim the message history to the last `max_length` messages.
        """
        if len(history) > max_length:
            return history[-max_length:]
        history = history + [m for m in new if m not in history]
        return history

    async def check_facts(self, messages: dict[int, list[discord.Message]]) -> dict[int, FactResponse]:
        """
        Check the messages for any facts that should be remembered.
        """
        parsed: dict[int, list[dict[str, str]]] = {}
        for channel_id, msgs in messages.items():
            parsed[channel_id] = []
            logger.debug(f"check_facts | Checking {len(msgs)} for facts in channel {channel_id}")
            for msg in msgs:
                parsed[channel_id].append({"role": "assistant" if self.user and msg.author.id == self.user.id else "user",
                    "content": msg.content,
                    "user_id": str(msg.author.id)})
    
        output = {}
        for c, msgs in parsed.items():
            prompt = memory_prompt(msgs)
            logger.debug(f"check_facts | Running memory agent for channel {c}")
            res = await self.memory_agent.run(prompt, deps=None, output_type=FactResponse) # type: ignore
            if res:
                output[c] = res.output

        if output:
            return output

        return {}
    
    @staticmethod
    def format_memory_messages(messages: dict[int, FactResponse]) -> list[dict[str, str]]:
        """
        Format the messages for memory addition.
        """
        formatted = []
        for channel_id, msgs in messages.items():
            for fact in msgs.facts:
                formatted.append({
                    "role": "assistant" if fact.user_id == str(channel_id) else "user",
                    "content": fact.content,
                    "user_id": fact.user_id
                })
        return formatted
    
    @staticmethod
    def is_bot_announcement(msg) -> bool:

        """
        Check if the message is from the bot.
        """
        return msg.content.startswith("BOT: ")

    async def add_memories(self, messages: dict[int, list[discord.Message]]) -> list[dict[str, Any]]:
        '''
        Adds messages to the bot's memories and returns the results.
        '''
        
        # checks each channels messages for facts, returns a {channel_id: FactResponse}
        fact_res: dict[int, FactResponse] = await self.check_facts(messages)
        if not fact_res:
            logger.debug("No facts found in watched messages.")
            return []

        for channel_id, facts in fact_res.items():
            if not facts.facts:
                logger.debug(f"No facts to add for channel {channel_id}.")
                continue
            
            logger.debug(f"add_memories | Processing {len(facts.facts)} messages in channel {channel_id}")
            if facts := [{"role": "assistant" if self.user and f.user_id == self.user.id else "user",
                        "content": f"{f.content}" if hasattr(f, 'topic') else f.content,
                        "topic": f.topic if hasattr(f, 'topic') else "",
                        "user_id": f.user_id}
                        for f in facts.facts]:

                for f in facts:
                    metadata = {"user_id": str(f["user_id"])}
                    if "topic" in f:
                        metadata["topic"] = f["topic"]

                    res = await self.memory.add([f], agent_id=str(self.user.id) if self.user and self.user.id else "",
                                            metadata=metadata, infer=False)
                    
                    return res.get("results", []) # type: ignore
            
        return []

    async def check_watched_channels(self) -> dict[int, list[discord.Message]]:
        logger.debug("Checking watched channels for new messages...")
    
        after = datetime.now(timezone.utc) - timedelta(minutes=5)
        watched_msgs: dict[int, list[discord.Message]] = {}
        for c in self.watched_channels:
            channel: discord.abc.GuildChannel | None = self.get_channel(c)
            if isinstance(channel, discord.TextChannel):
                watched_msgs[c] = [
                    m async for m in channel.history(after=after)
                    if not self.is_bot_announcement(m)
                    and not m.content.startswith(self.command_prefix)
                    and m.id not in self.seen_messages # type: ignore
                    and len(m.embeds) == 0  # Exclude messages with embeds
                ]

        return watched_msgs
