from datetime import datetime, timedelta, timezone
from typing import Any
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
        self.seen_messages = []
        self.watched_channels = set(config.get("DISCORD", {}).get("watched_channels", []))
        
    async def initialize_memory(self):
        self.memory = await CustomAsyncMemory.from_config(memory_config)

    async def check_facts(self, messages: dict[int, list[discord.Message]]) -> dict[int, FactResponse]:
        """
        Check the messages for any facts that should be remembered.
        """
        parsed: dict[int, list[dict[str, str]]] = {}
        for channel_id, msgs in messages.items():
            parsed[channel_id] = []
            logger.debug(f"check_facts | Checking {len(msgs)} for facts in channel {channel_id}")
            for msg in msgs:
                # Skip bot announcements and messages with embeds
                if len(msg.embeds) > 0:
                    continue 
                    
                parsed[channel_id].append(
                    {"role": "assistant" if self.bot.user and msg.author.id == self.bot.user.id else "user",
                    "content": msg.content,
                    "user_id": str(msg.author.id)})
    
        output = {}
        for c, msgs in parsed.items():
            prompt = memory_prompt(msgs)
            logger.debug(f"check_facts | Running memory agent for channel {c}")
            res = await self.bot.memory_agent.run(prompt) 
            if res:
                output[c] = res.output

        if output:
            return output

        return {}

    async def add_memories(self, messages: dict[int, list[discord.Message]]) -> list[dict[str, Any]]:
        """
        Adds messages to the bot's memories and returns the results.
        """
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
            try:
                if facts := [{"role": "user",
                              "content": f"{f.content}" if hasattr(f, 'topic') else f.content,
                              "topic": f.topic if hasattr(f, 'topic') else ""} for f in facts.facts]:

                    res = await self.memory.add(facts, agent_id=str(self.bot.user.id) if self.bot.user and self.bot.user.id else "", infer=False)

                    return res.get("results", []) # type: ignore
            except Exception as e:
                logger.error(f"Error adding memories for channel {channel_id}: {e}")
                continue
            
        return []
    
    async def add_memories_task(self) -> None:
        """ Adds memories from watched channels to the bot's memory."""
        logger.debug("add_memories_task | Running memory task...")

        watched_msgs = await self.check_watched_channels()
        if not watched_msgs:
            logger.debug("add_memories_task |No new messages found for memory check.")
            return
        
        # add the IDs of the messages to the seen_messages list
        for _, msgs in watched_msgs.items():
            for msg in msgs:
                self.seen_messages.append(msg.id)
       
        # Change bot status to busy
        await self.bot.change_presence(activity=discord.Game(name="Updating Memory..."), status=discord.Status.dnd)

        # add memories
        if res := await self.add_memories(watched_msgs):
            added = [
                f"**{r['event']} |** "
                + (f"{prev_m} **->**\n{r['memory']}" if (prev_m := r.get('previous_memory')) and prev_m != r['memory'] else r['memory'])
                for r in res]

            chunks = [added[i:i + 5] for i in range(0, len(added), 5)]
            for chunk in chunks:
                embed = discord.Embed(
                    title="Memory",
                    description="\n\n".join(chunk),
                    color=discord.Color.green()
                )

                if self.bot.bot_channel and isinstance(self.bot.bot_channel, discord.TextChannel):
                    await self.bot.bot_channel.send(embed=embed)

        # reset the bot status
        await self.bot.change_presence(activity=None, status=discord.Status.online)

    async def check_watched_channels(self) -> dict[int, list[discord.Message]]:
        """
        Check the watched channels for new messages.
        Returns a dictionary with channel IDs as keys and lists of messages as values.
        """

        logger.debug("Checking watched channels for new messages...")
    
        after = datetime.now(timezone.utc) - timedelta(minutes=5)
        watched_msgs: dict[int, list[discord.Message]] = {}
        for c in self.watched_channels:
            channel: discord.abc.GuildChannel | discord.abc.PrivateChannel | discord.Thread | None = self.bot.get_channel(c)
            if isinstance(channel, discord.TextChannel):
                watched_msgs[c] = [
                    m async for m in channel.history(after=after)
                    if not m.content.startswith(str(self.bot.command_prefix))
                    and not is_bot_announcement(m)
                    and m.id not in self.seen_messages # type: ignore
                    and len(m.embeds) == 0  # Exclude messages with embeds
                ]

        return watched_msgs