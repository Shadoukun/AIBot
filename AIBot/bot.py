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

MODEL_NAME = os.getenv("MODEL_NAME") or "huihui_ai/qwen3-abliterated:8b"
base_url = 'http://localhost:11434/v1'
MODEL_SETTINGS = {
    'temperature': 0.9
}

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
        self.memory_checked_at = datetime.now(timezone.utc)
        self.seen_messages: list[int] = []  # to track seen messages
        self.seen_cleared_at = datetime.now(timezone.utc)
        self.watched_channels: list[int] = []  # to track watched channels

    async def setup_hook(self):
        # create the agents
        self.model = OpenAIModel(model_name=MODEL_NAME,provider=OpenAIProvider(base_url=base_url))
        self.agent = Agent[AgentDependencies, AgentResponse](
            model=self.model,
            tools=[duckduckgo_search_tool(max_results=3), 
                # duckduckgo_image_search_tool(max_results=3), 
                add_memory, 
                wikipedia_search], 
            output_type=AgentResponse, 
            deps_type=AgentDependencies, system_prompt=default_system_prompt(ctx=None), 
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

        user_list = []
        if ctx.guild:
           user_list = [f'"{member.display_name}"' for member in ctx.guild.members if member != ctx.author]

        deps = AgentDependencies(
            user_list=user_list,
            agent_id=str(self.user.id) if self.user and self.user.id else "",
            username=ctx.author.display_name,
            message_id=str(ctx.message.id),
            user_id=str(ctx.author.id),
            context=ctx,
            memory=self.memory,
            memories=memories,
            bot_channel=self.bot_channel,  # type: ignore
        )

        async with ctx.typing():
            result = await self.agent.run(msg, deps=deps, # type: ignore
                                          model_settings=MODEL_SETTINGS, # type: ignore
                                          message_history=self.message_history)

            logger.debug(f"Agent Result: {result}")
            if result.output and result.output.content:
                await ctx.send(result.output.content)
                self.add_message_to_history(result)
    
    def add_message_to_history(self, result: AgentRunResult[Any]) -> None:
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

        ctx = await self.get_context(message)
        if self.user and self.user.mentioned_in(message):
            logger.debug(f"Bot Mentioned | {message.content}")
            await self.ask_agent(ctx)
        elif message.content.startswith(self.command_prefix):
            await self.process_commands(message)
     
    async def random_memory_check(self) -> None:
            
            now = datetime.now(timezone.utc)
            five_mins_ago = now - timedelta(minutes=5)

            msgs = []
            for channel in self.watched_channels:
                channel = self.get_channel(channel)
                if not channel or not isinstance(channel, discord.TextChannel):
                    continue
                async for m in channel.history(after=five_mins_ago):
                    if m.author == self.user:
                        continue
                    if m.id in self.seen_messages:
                        continue
                    else:
                        msgs.append(m)

            if not msgs:
                logger.debug("No new messages found for memory check.")
                return
            
            prompt = (
                "Does this conversation contain any information worth remembering?\n\n",
                "<conversation>\n",
                    "\n".join([f"<user>: {m.content}" for m in msgs]),
                "\n</conversation>"
            )

            logger.debug(f"Random Memory Check | {prompt}")
            deps = FactAgentDependencies(
                memory=self.memory,
                memories=[],
                bot_channel=self.bot_channel if isinstance(self.bot_channel, discord.abc.GuildChannel) else None
            )

            result = await self.fact_agent.run(prompt,
                                                 deps=deps,
                                                 output_type=FactResponse,
                                                 model_settings={"temperature": 0.4})
            
            if result.output and result.output.facts:
                if facts := [f for f in result.output.facts if f.text.strip()]:
                    logger.debug("Memory Check Result:")
                    logger.debug(f"Facts to save: {facts}")
                    embed = discord.Embed(
                        title="Memory Check Result",
                        description="The following facts were extracted from the conversation:",
                        color=discord.Color.blue()
                    )
                    await self.bot_channel.send(embed=embed) # type: ignore
            
            for m in msgs:
                if m.id in self.seen_messages:
                    continue
                else:
                    self.seen_messages.append(m.id)
                # for fact in facts:
                #     if fact.strip():
                #         # add the fact to memory
                #         memory_result = await self.memory.add(
                #             {"role": "user", "content": fact.strip()},
                #             agent_id=deps.agent_id,
                #             metadata={"user_id": deps.user_id, "username": deps.username}
                #         )
                #         if memory_result:
                #             logger.debug(f"Memory added successfully: {memory_result}")

    async def process_thinking_msg(self, ctx: commands.Context, part: ThinkingPart) -> None: 
        if util.print_thinking(ctx):
            msg = part.content
            # discord has a 2000 character limit per message
            chunks = [f"```{msg[i:i+2000]}```" for i in range(0, len(msg), 2000)]
            for chunk in chunks:
                await ctx.send(chunk)

    async def print_messages(self, ctx: commands.Context, deps: AgentDependencies, node: BaseNode) -> None:
        if isinstance(node, CallToolsNode):
            for part in node.model_response.parts:
                if isinstance(part, ThinkingPart):
                    await self.process_thinking_msg(ctx, part)
                    continue

                if isinstance(part, ToolCallPart):
                    name = part.tool_name
                    # pydantic seems to uses a final_result tool when it has a structured input. don't print it
                    if name == "final_result":
                        return
                    else:
                        logging.debug(f"Tool Call: {name} | Args: {part.args}")
                        embed = discord.Embed(
                            title=f"{name}",
                            color=discord.Color.green()
                        )
                        embed.add_field(name="\n", value=str(part.args), inline=False)
                        await ctx.send(embed=embed)
                        return

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
            await ctx.send(f"Added #{ctx.channel.name} to watched channels.")
        else:
            await ctx.send(f"#{ctx.channel.name} is already in watched channels.")