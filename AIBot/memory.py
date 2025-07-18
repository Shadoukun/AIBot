from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
import discord

from AIBot.util import is_bot_announcement
from .models import FactResponse
from .asyncmemory import CustomAsyncMemory
from .config import config, memory_config
from .prompts import memory_prompt
import logging

logger = logging.getLogger(__name__)

class MemoryHandler:
    """
    Handles all the memory operations for the bot.
    This includes checking watched channels for new messages, parsing facts, and adding them to the bot's memory.
    """
    memory: CustomAsyncMemory
    watched_channels: set[int] = set()

    def __init__(self, bot):
        self.bot = bot
        self.seen_messages: List[int] = []
        self.watched_channels = set(config.get("DISCORD", {}).get("watched_channels", []))

    @classmethod
    async def create(cls, bot) -> 'MemoryHandler':
        """
        Initialize the MemoryHandler with the bot instance.

        Args:
            bot: The bot instance to use for memory operations.

        Returns:
            An instance of MemoryHandler.
        """
        handler = cls(bot)
        handler.memory = await CustomAsyncMemory.from_config(memory_config)
        return handler

    async def check_facts(self, messages: Dict[int, List[discord.Message]]) -> Dict[int, FactResponse]:
        """
        Check the messages for any facts that should be remembered.

        Args:
            messages: A dictionary with channel IDs as keys and lists of messages as values.

        Returns:
            A dictionary with channel IDs as keys and FactResponse objects as values.
        """
        parsed: Dict[int, List[Dict[str, str]]] = {}
        for channel_id, msgs in messages.items():
            parsed[channel_id] = []
            logger.debug(f"check_facts | Checking {len(msgs)} messages for facts in channel {channel_id}")
            for msg in msgs:
                # Skip bot announcements and messages with embeds
                if len(msg.embeds) > 0:
                    continue

                parsed[channel_id].append({
                    "role": "assistant" if self.bot.user and msg.author.id == self.bot.user.id else "user",
                    "content": msg.content,
                    "user_id": str(msg.author.id)
                })

        output = {}
        for c, msgs in parsed.items():
            prompt = memory_prompt(msgs)
            logger.debug(f"check_facts | Running memory agent for channel {c}")
            try:
                res = await self.bot.memory_agent.run(prompt)
                if res:
                    output[c] = res.output
            except Exception as e:
                logger.error(f"Error running memory agent for channel {c}: {e}")

        return output

    async def add_memories(self, messages: Dict[int, List[discord.Message]]) -> List[Dict[str, Any]]:
        """
        Adds messages to the bot's memories and returns the results.

        Args:
            messages: A dictionary with channel IDs as keys and lists of messages as values.

        Returns:
            A list of dictionaries containing the results of the memory addition.
        """
        fact_res: Dict[int, FactResponse] = await self.check_facts(messages)
        if not fact_res:
            logger.debug("No facts found in watched messages.")
            return []

        results = []
        for channel_id, facts in fact_res.items():
            if not facts.facts:
                logger.debug(f"No facts to add for channel {channel_id}.")
                continue

            logger.debug(f"add_memories | Processing {len(facts.facts)} messages in channel {channel_id}")
            try:
                formatted_facts = [
                    {
                        "role": "user",
                        "content": f"{f.content}" if hasattr(f, 'topic') else f.content,
                        "topic": f.topic if hasattr(f, 'topic') else ""
                    } for f in facts.facts
                ]

                res = await self.memory.add(
                    formatted_facts,
                    agent_id=str(self.bot.user.id) if self.bot.user and self.bot.user.id else "",
                    infer=False
                )

                results.extend(res.get("results", []))  # type: ignore
            except Exception as e:
                logger.error(f"Error adding memories for channel {channel_id}: {e}")

        return results

    async def add_memories_task(self) -> None:
        """
        Adds memories from watched channels to the bot's memory.
        """
        logger.debug("add_memories_task | Running memory task...")

        watched_msgs = await self.check_watched_channels()
        if not watched_msgs:
            logger.debug("add_memories_task | No new messages found for memory check.")
            return

        # Add the IDs of the messages to the seen_messages list
        self.seen_messages.extend(msg.id for _, msgs in watched_msgs.items() for msg in msgs)

        # Change bot status to busy
        await self.bot.change_presence(
            activity=discord.CustomActivity(name="Updating Memory..."), status=discord.Status.dnd)

        # Add memories
        if res := await self.add_memories(watched_msgs):
            added = [
                f"**{r['event']} |** "
                + (f"{prev_m} **->**\n{r['memory']}" if (prev_m := r.get('previous_memory')) and prev_m != r['memory'] else r['memory'])
                for r in res
            ]

            chunks = [added[i:i + 5] for i in range(0, len(added), 5)]
            for chunk in chunks:
                embed = discord.Embed(
                    title="Memory",
                    description="\n\n".join(chunk),
                    color=discord.Color.green()
                )

                if self.bot.bot_channel and isinstance(self.bot.bot_channel, discord.TextChannel):
                    await self.bot.bot_channel.send(embed=embed)

        # Reset the bot status
        await self.bot.change_presence(activity=None, status=discord.Status.online)

    async def check_watched_channels(self) -> Dict[int, List[discord.Message]]:
        """
        Check the watched channels for new messages.

        Returns:
            A dictionary with channel IDs as keys and lists of messages as values.
        """
        logger.debug("Checking watched channels for new messages...")

        after = datetime.now(timezone.utc) - timedelta(minutes=5)
        watched_msgs: Dict[int, List[discord.Message]] = {}
        for c in self.watched_channels:
            channel = self.bot.get_channel(c)
            if isinstance(channel, discord.TextChannel):
                watched_msgs[c] = [
                    m async for m in channel.history(after=after)
                    if not m.content.startswith(str(self.bot.command_prefix))
                    and not is_bot_announcement(m)
                    and m.id not in self.seen_messages  # type: ignore
                    and len(m.embeds) == 0  # Exclude messages with embeds
                ]

        return watched_msgs