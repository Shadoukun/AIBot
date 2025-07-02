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
from models import AgentDependencies, AgentResponse
from prompts import default_system_prompt, update_user_prompt, fact_retrieval_prompt
import util
from tools import wikipedia_search, add_memory
from duckduckgo import duckduckgo_image_search_tool, duckduckgo_search_tool

from dotenv import load_dotenv
import random
load_dotenv()

logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(levelname)s - %(message)s',
)

MODEL_NAME = os.getenv("MODEL_NAME")
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
            "model": os.getenv("MODEL_NAME"),
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
    "update_memory_prompt": update_user_prompt(),
    "fact_retrieval_prompt": fact_retrieval_prompt(),
}

memory_config = MemoryConfig(**config) # type: ignore

class AIBot(commands.Bot): # type: ignore
    def __init__(self, command_prefix: str, intents: discord.Intents, **options: dict):
        super().__init__(command_prefix=command_prefix, intents=intents)
        self.message_history: list[ModelMessage] = []  # type: ignore
        
    async def setup_hook(self):
        self.model = OpenAIModel(model_name=MODEL_NAME or "gpt-3.5-turbo", 
                            provider=OpenAIProvider(base_url=base_url))

        # create the agents
        self.agent = Agent[AgentDependencies, AgentResponse](
            model=self.model,
            tools=[duckduckgo_search_tool(max_results=3), 
                duckduckgo_image_search_tool(max_results=3), 
                add_memory, 
                wikipedia_search], 
            output_type=AgentResponse, 
            deps_type=AgentDependencies, system_prompt=default_system_prompt(ctx=None), 
        )
        
        self.random_agent = Agent[AgentDependencies, AgentResponse](
            model=self.model,
            tools=[],
            output_type=AgentResponse,
            deps_type=AgentDependencies,
        )

        self.memory = await AsyncMemory.from_config(config)

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
        )

        async with ctx.typing():
            result = await self.agent.run(msg, deps=deps, # type: ignore
                                          model_settings=MODEL_SETTINGS, # type: ignore
                                          message_history=self.message_history)
            
            logging.debug(f"Agent Result: {result}")
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
        
        logging.debug("Message History:\n\n")
        for msg in self.message_history:
            logging.debug(msg)

    async def on_ready(self):
        if self.user and self.user.id:
            logging.info(f'Logged in as {self.user} (ID: {self.user.id})')
            logging.info('------')

    async def on_message(self, message):
        if message.author == self.user:
            return

        ctx = await self.get_context(message)
        if self.user and self.user.mentioned_in(message):
            logging.debug(f"Bot Mentioned | {message.content}")
            await self.ask_agent(ctx)
        elif message.content.startswith(self.command_prefix):
            await self.process_commands(message)
     
        else: 
            # ask the agent if the message is worth remembering at random intervals
            if random.random() < 0.35:
                deps = AgentDependencies(
                    user_list=[],
                    agent_id=str(self.user.id) if self.user and self.user.id else "",
                    username=message.author.display_name,
                    message_id=str(message.id),
                    user_id=str(message.author.id),
                    context=ctx, 
                    memory=self.memory,
                    memories=[],
                )

                msg = ("Does this message contain any information worth remembering?\n\n" + message.content +
                    "\n\nIf so, add it to memory.")
                
                logging.debug(f"Random Memory Check | {msg}")
                result = await self.random_agent.run(msg, deps=deps, model_settings={"temperature": 0.4}) # type: ignore
                if result.output and result.output.content:
                    await ctx.send(result.output.content)
       
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
                        await ctx.send(f"{name}: {part.args}")
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