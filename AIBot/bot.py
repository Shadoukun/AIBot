import logging
import os
from typing import Any
import discord
from discord.ext import commands
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ThinkingPart, ModelMessage
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider
from mem0.configs.base import MemoryConfig
from .duckduckgo import DuckDuckGoResult

from mem0 import AsyncMemory
from . import util
from .models import AgentDependencies, AgentResponse, FactAgentDependencies, FactResponse
from .prompts import default_system_prompt, search_agent_system_prompt, update_user_prompt, fact_retrieval_prompt, search_preamble_prompt
from .tools import wikipedia_search
from .duckduckgo import duckduckgo_image_search_tool, duckduckgo_search_tool
from duckduckgo_search import DDGS

from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
load_dotenv()

logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("MODEL_NAME") or ""
BASE_URL = os.getenv("BASE_URL") or "http://localhost:11434/v1"
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

        # supposedly fixes the 202 rate limit issue 
        # if instead of constantly instantiating a new client, you reuse the same one
        self.ddgs_client = DDGS()

    async def setup_hook(self):
        self.memory = await AsyncMemory.from_config(config)
        self.create_agents()

        self.loop.create_task(util.memory_timer(self))
        self.loop.create_task(util.seen_messages_timer(self))

    def create_agents(self):
        # create the agents
        self.model = OpenAIModel(model_name=MODEL_NAME,provider=OpenAIProvider(base_url=BASE_URL))
        self.openrouter_model = OpenAIModel(
            'google/gemini-2.5-flash',
            provider=OpenRouterProvider(api_key=os.getenv("OPENROUTER_API_KEY", "")),
        )

        self.agent = Agent[AgentDependencies, AgentResponse](
            model=self.model,
            instructions=[default_system_prompt],
            output_type=AgentResponse, 
            deps_type=AgentDependencies, 
        )

        @self.agent.tool
        async def search(ctx: RunContext[AgentDependencies], query: str) -> list[DuckDuckGoResult]:
            """
            Search for the given query using the DuckDuckGo search tool.
            """
            logger.debug(f"Search Query: {query}")
            query = query.strip()
            prompt = search_preamble_prompt.format(query=query) if query else "I don't know what to search for."
            try:
                res = await self.search_agent.run(prompt, deps=ctx.deps, output_type=str, model_settings={'temperature': 0.9}) # type: ignore
                await ctx.deps.context.send(res.output) # type: ignore
                results = await self.search_agent.run(query, deps=ctx.deps, output_type=list[DuckDuckGoResult]) # type: ignore
                if results:
                    return results.output
                return []
            except Exception as e:
                logger.error(f"Error during search: {e}")
                return []
        
        self.fact_agent = Agent[FactAgentDependencies, FactResponse](
            model=self.model,
            tools=[],
            output_type=FactResponse,
            deps_type=FactAgentDependencies,
            system_prompt=fact_retrieval_prompt(),
        )

        self.search_agent = Agent(
            model=self.openrouter_model,
            tools=[
                duckduckgo_search_tool(duckduckgo_client=self.ddgs_client, max_results=3),
                wikipedia_search
            ],
        ) # type: ignore


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
        after = datetime.now(timezone.utc) - timedelta(minutes=5)

        # get all messages from the watched channels
        logger.debug(self.watched_channels)
        watched_msgs: dict[int, list[discord.Message]] = {}
        for c in self.watched_channels:
            channel = self.get_channel(c)
            if isinstance(channel, discord.TextChannel):
                watched_msgs[c] = [
                    m async for m in channel.history(after=after)
                    if not util.is_bot_announcement(m)
                    and not m.content.startswith(self.command_prefix)  # type: ignore
                    and m.id not in self.seen_messages
                ]

        if not watched_msgs:
            logger.debug("No new messages found for memory check.")
            return

        if res := await util.add_memories(self, watched_msgs):
            logger.debug(f"Memory results: {res}")
            chunks = [res[i:i+1024] for i in range(0, len(res), 1024)]
            for chunk in chunks:
                embed = discord.Embed(
                    title="Memory Added",
                    description=chunk,
                    color=discord.Color.blue()
                )

                if self.bot_channel and \
                    isinstance(self.bot_channel, discord.TextChannel):
                    await self.bot_channel.send(embed=embed)


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
