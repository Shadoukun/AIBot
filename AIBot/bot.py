import logging
import os
from typing import Any
import discord
from discord.ext import commands
from pydantic_ai import Agent, CallToolsNode
from pydantic_ai.messages import ToolCallPart, ThinkingPart, ModelMessage
from pydantic_graph import End, BaseNode
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from mem0.configs.base import MemoryConfig

from mem0 import AsyncMemory
from .models import AgentDependencies, AgentResponse, FactAgentDependencies, FactResponse
from .prompts import default_system_prompt, update_user_prompt, fact_retrieval_prompt
from . import util
from .tools import wikipedia_search, add_memory
from .duckduckgo import duckduckgo_image_search_tool, duckduckgo_search_tool

from dotenv import load_dotenv
import random
from datetime import datetime, timedelta, timezone
load_dotenv()

logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("MODEL_NAME")
BASE_URL = os.getenv("BASE_URL")
MODEL_SETTINGS = {
    'temperature': 0.7
}

# mem0 config
config = {
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "memory",
            "path": "db",
        }
    },
    "llm": {
        "provider": "ollama",
        "config": {
            "model": MODEL_NAME,
            "temperature": 0.6,
            "max_tokens": 2000,
            "ollama_base_url": "http://localhost:11434",
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text:latest",
            "ollama_base_url": "http://localhost:11434",
        },
    },
    "custom_update_memory_prompt": update_user_prompt(),
    "custom_fact_extraction_prompt": fact_retrieval_prompt(),
}

memory_config = MemoryConfig(**config) # type: ignore

class AIBot(commands.Bot): # type: ignore
    def __init__(self, command_prefix: str, intents: discord.Intents, **options: dict):
        super().__init__(command_prefix=command_prefix, intents=intents)
        self.message_history: list[ModelMessage] = []  # type: ignore
        self.watched_channels: list[int] = []
        self.memory_checked_at = datetime.now(timezone.utc)
        self.seen_messages: list[int] = []
        self.seen_cleared_at = datetime.now(timezone.utc)

    async def setup_hook(self):
        # create the agents
        self.model = OpenAIModel(model_name=MODEL_NAME,provider=OpenAIProvider(base_url=BASE_URL))
        self.agent = Agent[AgentDependencies, AgentResponse](
            model=self.model,
            instructions=[default_system_prompt],
            output_type=AgentResponse, 
            deps_type=AgentDependencies, 
            tools=[duckduckgo_search_tool(max_results=3), 
                # duckduckgo_image_search_tool(max_results=3), 
                add_memory, 
                wikipedia_search], 
        )

        self.fact_agent = Agent[FactAgentDependencies, FactResponse](
            model=self.model,
            tools=[],
            output_type=FactResponse,
            deps_type=FactAgentDependencies,
            system_prompt=fact_retrieval_prompt(),
        )

        self.memory = await AsyncMemory.from_config(config)

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
        memory_results = await self.memory.search(query=msg, agent_id=user_id, limit=15)
        for entry in memory_results["results"]:
           memories.append(entry["memory"])
           logger.debug(f"Memory: {entry['memory']}")

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

    async def memory_check(self) -> None:
            logger.debug("Running memory check...")
            now = datetime.now(timezone.utc)
            five_mins_ago = now - timedelta(minutes=5)

            watched_msgs: dict[int, list[discord.Message]] = {}
            
            logger.debug(self.watched_channels)
            for c in self.watched_channels:
                watched_msgs[c] = []
                channel = self.get_channel(c)
                if not isinstance(channel, discord.TextChannel):
                    continue

                async for m in channel.history(after=five_mins_ago):
                    if util.is_bot_announcement(m):
                        continue
                    if m.content.startswith(self.command_prefix): # type: ignore
                        continue
                    if m.id in self.seen_messages:
                        continue
                    else:
                        watched_msgs[c].append(m)

            if not watched_msgs:
                logger.debug("No new messages found for memory check.")
                return

            msgs_to_add = []
            for channel_id, msgs in watched_msgs.items():
                result_msgs = []
                logger.debug(f"Processing {len(msgs)} messages in channel {channel_id}")
                for m in msgs:
                    if m.id not in self.seen_messages:
                        logger.debug(f"New message found for memory check: {m.content}")
                        self.seen_messages.append(m.id)

                        if self.user and m.author.id == self.user.id:
                            role = "assistant"
                        else:
                            role = "user"
                        
                        # Add the message to the memory
                        msg = {"role": role, "content": m.content}
                        msgs_to_add.append(msg)

                    res = await self.memory.add(msgs_to_add, agent_id=str(self.user.id) if self.user and self.user.id else "")
                    if isinstance(res, dict):
                        results = res.get("results", [])
                        for result in results:
                            result_msgs.append(f"Memory: {result['memory']} Event: {result['event']}")

                if result_msgs:
                    logger.debug(f"Memory results: {result_msgs}")
                    embed = discord.Embed(
                        title="Memory Added",
                        description=f"Added {len(result_msgs)} messages to memory.",
                        color=discord.Color.blue()
                    )
                    truncated_msgs = [msg[:1021] + "..." if len(msg) > 1024 else msg for msg in result_msgs]
                    embed.add_field(name="Messages", value="\n".join(truncated_msgs), inline=False)
                    await self.bot_channel.send(embed=embed) # type: ignore

                    
    async def process_thinking_msg(self, ctx: commands.Context, part: ThinkingPart) -> None: 
        if util.print_thinking(ctx):
            msg = part.content
            # discord has a 2000 character limit per message
            chunks = [f"```{msg[i:i+2000]}```" for i in range(0, len(msg), 2000)]
            for chunk in chunks:
                await ctx.send(chunk)

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
            await ctx.send(f"BOT: Added #{ctx.channel.name} to watched channels.")
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