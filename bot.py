import logging
import os
import discord
from discord.ext import commands
from pydantic_ai import CallToolsNode
from pydantic_ai.messages import ToolCallPart, ThinkingPart
from pydantic_graph import End, BaseNode
from mem0 import AsyncMemory
from models import AgentDependencies
from mem import config
import util
import agent
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(levelname)s - %(message)s',
)

MODEL_NAME = os.getenv("MODEL_NAME")

MODEL_SETTINGS = {
    'temperature': 0.9
}

class AIBot(commands.Bot): # type: ignore
    def __init__(self, command_prefix: str, intents: discord.Intents, **options: dict):
        super().__init__(command_prefix=command_prefix, intents=intents)
        self.agent = agent.AIAgent
        self.message_history = []
        
    async def setup_hook(self):
        self.memory = await AsyncMemory.from_config(config)

    async def ask_agent(self, ctx: commands.Context):
        """
        Ask the AI agent a question.

        Args:
            ctx (commands.Context): the discord context.
        """

        logging.debug(f"Received message: {ctx.message.content}")
        msg = util.remove_command_prefix(ctx.message.content, prefix=ctx.prefix if ctx.prefix else "")
        
        relevant_memories = await self.memory.search(query=msg, user_id=str(ctx.author.id), limit=5)
        memories = []
        for entry in relevant_memories["results"]:
            memories.append(entry["memory"])
        
        user_list = []
        if ctx.guild:
            user_list = [f'"{member.display_name}"' for member in ctx.guild.members if member != ctx.author]

        deps = AgentDependencies(
            user_list=user_list,
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
        else:
            await self.process_commands(message)
       
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

@bot.command(name="clear_history")
async def clear_history(ctx):
    logging.debug(f"Clear History Command | {ctx.message.content}")
    bot.message_history = []
    await ctx.send("Chat history cleared.")