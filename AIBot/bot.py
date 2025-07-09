import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any
import discord
from discord.ext import commands
from pydantic_ai.messages import ModelMessage
from pydantic_ai.agent import AgentRunResult

from mem0 import AsyncMemory
from . import util
from .models import AgentDependencies, Fact, FactResponse
from .agents import main_agent, memory_agent, memory_config

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

MODEL_SETTINGS = {
    'temperature': 0.7
}

class AIBot(commands.Bot): # type: ignore
    def __init__(self, command_prefix: str, intents: discord.Intents, **options: dict):
        super().__init__(command_prefix=command_prefix, intents=intents)
        self.agent = main_agent
        self.memory_agent = memory_agent
        self.message_history:    list[ModelMessage] = []  # type: ignore
        self.seen_messages:      list[int] = []
        self.watched_channels:   list[int] = []
        self.seen_cleared_at   = datetime.now(timezone.utc)
        self.memory_checked_at = datetime.now(timezone.utc)
    
    async def setup_hook(self):
        self.memory = await AsyncMemory.from_config(memory_config)

        self.loop.create_task(util.memory_timer(self))
        self.loop.create_task(util.seen_messages_timer(self))

    async def ask_agent(self, ctx: commands.Context):
        """
        Ask the AI agent a question.

        Args:
            ctx (commands.Context): the discord context.
        """
        msg = util.remove_command_prefix(ctx.message.content, prefix=ctx.prefix if ctx.prefix else "")
        user_id = str(self.user.id) if self.user and self.user.id else ""
       
        memories = []
        memory_results = await self.memory.search(query=msg, agent_id=user_id, limit=10)
        for entry in memory_results["results"]:
            if entry and "memory" in entry:
                logger.debug(f"Memory: {entry['memory']}")
                m = entry["memory"].format(user=ctx.author.display_name if ctx.author else "User")
                memories.append(m)
    
        deps = AgentDependencies(
            bot=self,
            ctx=ctx,
            memories=memories
        )

        async with ctx.typing():
            result = await self.agent.run(msg, deps=deps, # type: ignore
                                          model_settings=MODEL_SETTINGS, # type: ignore
                                          message_history=self.message_history)

            logger.debug(f"Agent Result: {result}")
            if result.output and result.output.content:
                await ctx.send(result.output.content)
                self.add_message_to_chat_history(result)

    def add_message_to_chat_history(self, result: AgentRunResult[Any]) -> None:
        """
        Add a message to the bot's message history.
        """
        self.message_history.extend(result.new_messages())

        # Limit the message history to the last 20 messages
        if self.message_history and len(self.message_history) > 20:
            self.message_history = self.message_history[-20:]
        
        logger.debug("Message History:\n\n")
        for msg in self.message_history:
            logger.debug(msg)

    async def on_ready(self):
        if self.user and self.user.id:
            logger.info(f'Logged in as {self.user} (ID: {self.user.id})')
            logger.info('------')
        self.bot_channel = self.get_channel(int(os.getenv("BOT_CHANNEL_ID") or 0))

    async def on_message(self, message):
        if message.author == self.user:
            return
        
        # for now ignore messages with URLs
        if "http://" in message.content or "https://" in message.content:
            logger.debug("Message contains a URL. Ignoring.")
            return

        ctx = await self.get_context(message)
        if self.user and self.user.mentioned_in(message):
            logger.debug(f"Bot Mentioned | {message.content}")
            await self.ask_agent(ctx)
        elif message.content.startswith(self.command_prefix):
            await self.process_commands(message)

    async def add_memories_task(self) -> None:
        logger.debug("Running memory task...")

        watched_msgs = await util.check_watched_channels(self)
        if not watched_msgs:
            logger.debug("No new messages found for memory check.")
            return
        
        # add the IDs of the messages to the seen_messages list
        for channel_id, msgs in watched_msgs.items():
            for msg in msgs:
                self.seen_messages.append(msg.id)
       
        # Change bot status to busy
        await self.change_presence(activity=discord.Game(name="Updating Memory..."), status=discord.Status.dnd)

        if res := await util.add_memories(self, watched_msgs):
            logger.debug(f"Memory results: {res}")
            chunks = [res[i:i + 5] for i in range(0, len(res), 5)]
            for chunk in chunks:
                embed = discord.Embed(
                    title="Memory Update",
                    description="\n\n".join(chunk),
                    color=discord.Color.green()
                )

                if self.bot_channel and isinstance(self.bot_channel, discord.TextChannel):
                    await self.bot_channel.send(embed=embed)

        # reset the bot status
        await self.change_presence(activity=None, status=discord.Status.online)

intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = AIBot(command_prefix='!', intents=intents)

@bot.command(name='chat')
async def chat(ctx, *, query: str):
    logging.debug(f"Bot Command | {ctx.message.content}")
    await bot.ask_agent(ctx)

@bot.command(name="clear")
async def clear_history(ctx):
    logging.debug(f"Clear History Command | {ctx.message.content}")
    bot.message_history = []
    await ctx.send("Chat history cleared.")

@bot.command(name="watch")
async def add_watched_channel(ctx: commands.Context):
    """Add the current channel to the watched channels."""
    if isinstance(ctx.channel, discord.abc.GuildChannel):
        if ctx.channel.id not in bot.watched_channels:
            bot.watched_channels.append(ctx.channel.id)
            await ctx.message.add_reaction("✅")
        else:
            await ctx.send(f"BOT: #{ctx.channel.name} is already in watched channels.")

@bot.command(name="unwatch")
async def remove_watched_channel(ctx: commands.Context):
    """Remove the current channel from the watched channels."""
    if isinstance(ctx.channel, discord.abc.GuildChannel):
        if ctx.channel.id in bot.watched_channels:
            bot.watched_channels.remove(ctx.channel.id)
            await ctx.send(f"BOT: Removed #{ctx.channel.name} from watched channels.")
        else:
            await ctx.send(f"BOT: #{ctx.channel.name} is not in watched channels.")
