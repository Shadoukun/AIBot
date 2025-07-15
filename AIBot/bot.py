import asyncio
import io
import logging
import random
from datetime import datetime, timedelta, timezone
import re
from typing import Any, Set
from urllib.parse import urlparse

import discord
import matplotlib.pyplot as plt
import numpy as np
from pydantic_graph import End
from pydantic_ai.usage import UsageLimits
import umap
from discord.ext import commands
from discord.utils import escape_mentions

from pydantic_ai.agent import AgentRunResult
from pydantic_ai.messages import ModelMessage

from .agents import main_agent, memory_agent, true_false_agent, memory_config
from .config import config, write_config
from .memory import CustomAsyncMemory
from .models import AgentDependencies
from .util import AgentUtilities, is_admin

# import all the tools after the agents are defined
from . import tools  # noqa: F401

logger = logging.getLogger(__name__)

MODEL_SETTINGS = config.get("OLLAMA", {}).get("MODEL_SETTINGS", {})

class AIBot(commands.Bot, AgentUtilities):
    def __init__(self, command_prefix: str, intents: discord.Intents, **options: dict):
        super().__init__(command_prefix=command_prefix, intents=intents)
        
        self.agent         = main_agent
        self.memory_agent  = memory_agent

        self.watched_channels:   list[int] = config.get("DISCORD", {}).get("watched_channels", [])
        self.watched_domains:    Set[str] = set(config.get("DISCORD", {}).get("watched_domains", []))
        self.message_history:    list[ModelMessage] = []  # type: ignore
        self.seen_messages:      list[int] = []
        self.seen_cleared_at   = datetime.now(timezone.utc)
        self.memory_checked_at = datetime.now(timezone.utc)
        self.last_message_was  = datetime.now(timezone.utc)

    async def setup_hook(self):
        self.memory = await CustomAsyncMemory.from_config(memory_config)

        self.loop.create_task(memory_timer(self))
        self.loop.create_task(seen_messages_timer(self))

    async def on_ready(self):
        if self.user and self.user.id:
            logger.info(f'Logged in as {self.user} (ID: {self.user.id})')
            logger.info('------')
        self.bot_channel = self.get_channel(int(config.get("DISCORD", {}).get("bot_channel_id", 0)))
    
    async def on_message(self, message):
        if message.author == self.user:
            return
        
        ctx = await self.get_context(message)

        # for now ignore messages with URLs
        if "http://" in message.content or "https://" in message.content:
            urls = re.findall(r'https?://[^\s]+', message.content)
            for url in urls:
                parsed_url = urlparse(url)
                if parsed_url.netloc in self.watched_domains:
                    logger.debug(f"on_message | Ignoring message with URL: {url}")
            
            return

        if self.user and self.user.mentioned_in(message):
            logger.debug(f"on_message | Bot Mentioned | {message.content}")
            await self.ask_agent(ctx)

        elif message.content.startswith(self.command_prefix):
            logger.debug(f"on_message | Bot Command | {message.content}")
            await self.process_commands(message)

        elif random.random() < 0.05:
            logger.debug("on_message | Random Event")

            msg = ctx.message.content
            res = await true_false_agent.run("Does the following message contain anything worth replying to? \n\n" 
                                             + msg + " /nothink") # type: ignore
            
            if res.output.result:
                logger.debug("on_message | Generating Random Event Message")
                msg = ("Generate a random message based on the following content: \n\n"
                       + msg 
                       + "\n\n Don't use any tools for this. Don't simply repeat the message, but generate a new response based on it."
                       + " /nothink")

                res = await self._agent_run(
                    msg,
                    AgentDependencies(bot=self, ctx=ctx, memories=[]),
                    usage_limits=UsageLimits(request_limit=5)
                )

                if res.output and res.output.content:
                    logger.debug(f"on_message | Random Event Result: {res.output.content}")
                    await ctx.send(res.output.content)

    async def ask_agent(self, ctx: commands.Context):
        """
        Ask the AI agent a question.

        Args:
            ctx (commands.Context): the discord context.
        """
        # Reset message history if it has been more than 5 minutes since the last agent message was added
        if datetime.now(timezone.utc) - timedelta(minutes=5) > self.last_message_was:
            self.message_history = []
        self.last_message_was = datetime.now(timezone.utc)
        
        async with ctx.typing():
            user_id = str(self.user.id) if self.user and self.user.id else ""
            msg = self.remove_command_prefix(ctx.message.content, prefix=ctx.prefix if ctx.prefix else "")
            msg = escape_mentions(msg)
        
            memories = []
            memory_results = await self.memory.search(query=msg, agent_id=user_id, limit=8)
            for entry in memory_results["results"]:
                if entry and "memory" in entry:
                    logger.debug(f"ask_agent | Memory: {entry['memory']}")
                    memories.append(entry["memory"].format(user=ctx.author.display_name if ctx.author else "User"))

            result = await self._agent_run(msg, AgentDependencies(bot=self, ctx=ctx, memories=memories))
            
            self.add_message_to_chat_history(result)

            if result.output and result.output.content:
                logger.debug(f"Agent Result: {result.output}")
                await ctx.send(result.output.content)
            
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
        await self.change_presence(activity=discord.Game(name="Updating Memory..."), status=discord.Status.dnd)

        # add memories
        if res := await self.add_memories(watched_msgs):
            added: list[str] = [
                f"**{r['event']} |** {r['previous_memory']} **->**\n{r['memory']}" if r.get('previous_memory')
                else f"**{r['event']} |** {r['memory']}"
                for r in res
            ]
            logger.debug(f"add_memories_task | Added {len(added)} memories.")

            chunks = [added[i:i + 5] for i in range(0, len(added), 5)]
            for chunk in chunks:
                embed = discord.Embed(
                    title="Memory",
                    description="\n\n".join(chunk),
                    color=discord.Color.green()
                )

                if self.bot_channel and isinstance(self.bot_channel, discord.TextChannel):
                    await self.bot_channel.send(embed=embed)

        # reset the bot status
        await self.change_presence(activity=None, status=discord.Status.online)
    
    def add_message_to_chat_history(self, result: AgentRunResult[Any]) -> None:
        """
        Add a message to the bot's message history.
        """
        self.message_history.extend(result.new_messages())

        # Limit the message history to the last 20 messages
        if self.message_history and len(self.message_history) > 20:
            self.message_history = self.message_history[-20:]

    async def _agent_run(self, 
                         query: str, 
                         deps: AgentDependencies, 
                         usage_limits: UsageLimits = UsageLimits(request_limit=5)
                         ) -> AgentRunResult[Any]:
        """
        Run the agent with the given query and dependencies and limits.
        """
        async with self.agent.iter(query, 
                                   deps=deps, 
                                   usage_limits=usage_limits, 
                                   model_settings=MODEL_SETTINGS, 
                                   message_history=self.message_history
                                   ) as agent_run:
            
            node = agent_run.next_node
            all_nodes = [node]

            while not isinstance(node, End):
                node = await agent_run.next(node)
                all_nodes.append(node)
        
        if agent_run.result is None:
            raise ValueError("The agent run did not return a result.")

        return agent_run.result


async def memory_timer(bot):
    """ Regularly check for new messages to add to memory every 5 minutes."""
    while True:
        await asyncio.sleep(300)  # Wait for 5 minutes
        await bot.add_memories_task()  # Run the memory check function

async def seen_messages_timer(bot):
    """ Regularly clear the seen messages list every 30 minutes."""
    while True:
        await asyncio.sleep(1800)  # Wait for 30 minutes
        bot.seen_messages = []  # Clear the seen messages every 30 minutes

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

            # update config
            config["DISCORD"]["watched_channels"] = bot.watched_channels
            write_config(config, path="config.yaml")

            await ctx.message.add_reaction("✅")
        else:
            await ctx.message.add_reaction("🚫")

@bot.command(name="unwatch")
async def remove_watched_channel(ctx: commands.Context):
    """Remove the current channel from the watched channels."""
    if isinstance(ctx.channel, discord.abc.GuildChannel):
        if ctx.channel.id in bot.watched_channels:
            bot.watched_channels.remove(ctx.channel.id)
            
            # update config
            config["DISCORD"]["watched_channels"] = bot.watched_channels
            write_config(config, path="config.yaml")
            
            await ctx.message.add_reaction("❌")
        else:
            await ctx.message.add_reaction("🚫")

@bot.command(name="memories")
async def memories(ctx: commands.Context):
    """
    Visualize the bot's memories using UMAP and Matplotlib.
    """
    logging.debug(f"Memories Command | {ctx.message.content}")

    if bot.memory.vector_store.client:
        client = bot.memory.vector_store.client
        collection = client.get_collection(name="memory")

    # 2. Get all embeddings and associated metadata
    results = collection.get(include=["embeddings", "metadatas", "documents"])
    embeddings = results["embeddings"]
    documents = results["metadatas"]
    if documents is None:
        documents = []

    for doc in documents:
        if "data" not in doc:
            doc["data"] = "No text" # type: ignore
        else:
            doc["data"] = " ".join(str(doc["data"]).split()) # type: ignore

    # extract labels from metadata
    # Generate labels, but remove labels that are too close in 2D space to avoid clutter
    raw_labels = [" ".join(doc.get("data", "No text").split()[:2]) for doc in documents]  # type: ignore

    # 3. Reduce dimensionality with UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.15, metric='cosine')
    embedding_2d = reducer.fit_transform(embeddings)
    embedding_2d = np.array(embedding_2d)  # Ensure it's a NumPy array

    # Remove labels that are too close to each other
    # Minimum distance for label spreading
    min_dist = 0.3
    labels = []
    label_positions = []

    for i, (x, y) in enumerate(embedding_2d):
        too_close = False
        for (lx, ly) in label_positions:
            if np.sqrt((x - lx) ** 2 + (y - ly) ** 2) < min_dist:
                too_close = True
                break
        if not too_close:
            labels.append(raw_labels[i])
            label_positions.append((x, y))
        else:
            labels.append("")  # Empty label for crowded points

    # 4. Plot the 2D UMAP projection
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=0.5)  # Scatter plot
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=0.5)

    # Optional: annotate a few points
    for i, label in enumerate(labels):
        plt.text(embedding_2d[i, 0], embedding_2d[i, 1], label, fontsize=6) # only show a few to avoid clutter  
    
    plt.title("UMAP projection of ChromaDB embeddings")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    file = discord.File(buf, filename=f"memory_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    await ctx.send(file=file)

@is_admin
@bot.command(name="delete_memory")
async def delete_memory(ctx: commands.Context, *, memory_content: str):
    """
    Delete a specific memory from the bot's memory.
    
    Args:
        ctx (commands.Context): The context of the command.
        memory_content (str): The content of the memory to delete.
    """
    memory_content = memory_content.replace("‘", "'").replace("’", "'")

    memory = await bot.memory.search(query=memory_content, agent_id=str(bot.user.id if bot.user else ""), limit=1)
    if memory and memory["results"]:
        if memory_content != memory["results"][0].get("memory", ""):
            return await ctx.send("No matching memory found.")
        mem_entry = memory["results"][0]
        embed = discord.Embed(
            title="Delete Memory?",
            description=f"Do you want to delete this memory?\n\n{mem_entry['memory']}",
            color=discord.Color.red()
        )
        msg = await ctx.send(embed=embed)
        await msg.add_reaction("✅")
        await msg.add_reaction("❌")

        def check(reaction, user):
            return (user == ctx.author and 
                    reaction.message.id == msg.id and
                    str(reaction.emoji) in ["✅", "❌"])

        try:
            reaction, _ = await bot.wait_for("reaction_add", timeout=60.0, check=check)
            if str(reaction.emoji) == "✅":
                await bot.memory.delete(mem_entry["id"])
                embed = discord.Embed(
                    title="Memory Deleted",
                    description=f"DELETE | {mem_entry['memory']}",
                    color=discord.Color.red()
                )
                await ctx.send(embed=embed)
            else:
                embed = discord.Embed(
                    title="Memory Deletion Cancelled",
                    description="CANCEL | Memory deletion cancelled.",
                    color=discord.Color.red()
                )
                await ctx.send(embed=embed)
        except asyncio.TimeoutError:
            await ctx.send("No reaction received. Memory deletion cancelled.")

@bot.command(name="search_memory")
async def search_memory(ctx: commands.Context, *, query: str):
    """
    Query the bot's memory for a specific term.
    """
    memory = await bot.memory.search(query=query, agent_id=str(bot.user.id if bot.user else ""), limit=20)
    if memory and memory["results"]:
        chunks = [memory["results"][i:i + 5] for i in range(0, len(memory["results"]), 5)]
        current_chunk = 0
        embed = discord.Embed(
                    title="Memory",
                    description="\n\n".join(f"{c['id']}: {c['memory']}" for c in chunks[current_chunk]),
                    color=discord.Color.green()
                )
        msg = await ctx.send(embed=embed)
        await msg.add_reaction("⬅️")
        await msg.add_reaction("➡️")

    def check(reaction, user):
        return (user == ctx.author and 
                reaction.message.id == msg.id and
                str(reaction.emoji) in ["⬅️", "➡️"])
    try:
        while True:
            reaction, _ = await bot.wait_for("reaction_add", timeout=60.0, check=check)
            if str(reaction.emoji) == "➡️":
                if current_chunk >= len(chunks) - 1:
                    await msg.remove_reaction("➡️", ctx.author)
                else:
                    current_chunk += 1
                    if current_chunk < len(chunks):
                        embed.description = "\n\n".join(f"{c['id']}: {c['memory']}" for c in chunks[current_chunk])
                        await msg.edit(embed=embed)
                        await msg.remove_reaction("➡️", ctx.author)

            elif str(reaction.emoji) == "⬅️":
                if current_chunk == 0:
                    await msg.remove_reaction("⬅️", ctx.author)
                if current_chunk > 0:
                    current_chunk -= 1
                    embed.description = "\n\n".join(f"{c['id']}: {c['memory']}" for c in chunks[current_chunk])
                    await msg.edit(embed=embed)
                    await msg.remove_reaction("⬅️", ctx.author)
    except asyncio.TimeoutError:
        return