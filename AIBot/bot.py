import asyncio
import io
import logging
import random
from datetime import datetime, timedelta, timezone
from typing import Any
import html

from atproto import Client
import discord
import matplotlib.pyplot as plt
import numpy as np
from AIBot.memory import MemoryHandler
import umap
from discord.ext import commands
from discord.utils import escape_mentions
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import UsageLimits

from .agents import main_agent, memory_agent, true_false_agent, search_agent
from .config import config, write_config
from .models import AgentDependencies, FollowUpQuestion
from .prompts import random_message_prompt, random_search_prompt
from .util import is_admin, remove_command_prefix, user_msg, sys_msg
from urllib.parse import urlparse

# import all the tools after the agents are defined
from . import tools  # noqa: F401

logger = logging.getLogger(__name__)

MODEL_SETTINGS = config.get("OLLAMA", {}).get("MODEL_SETTINGS", {})
atproto_config = config.get("bluesky", {})
atproto_client = Client()
atproto_client.login(atproto_config.get("username"), atproto_config.get("app_password"))

class AIBot(commands.Bot):
    def __init__(self, command_prefix: str, intents: discord.Intents, **options: dict):
        super().__init__(command_prefix=command_prefix, intents=intents)

        self.agent         = main_agent
        self.memory_agent  = memory_agent
        self.atproto_client = atproto_client

        self.watched_channels:   list[int] = config.get("DISCORD", {}).get("watched_channels", [])
        self.watched_domains:    list[str] = config.get("DISCORD", {}).get("watched_domains", [])
        self.message_history:    dict[int, Any] = {}
        self.seen_messages:      list[int] = []
        self.seen_cleared_at   = datetime.now(timezone.utc)
        self.memory_checked_at = datetime.now(timezone.utc)
        self.last_message_was:   dict[int, datetime] = {}
        # memories that aren't necessarily from messages, and are injected outside the message history
        self.extra_memories:     list[dict[str, str]] = []

    def active_conversation(self, channel_id: int) -> bool:
        """
        Check if there is an active conversation in the given channel.
        """
        if not self.message_history.get(channel_id):
            return False

        # Check if the last message was within the last 2 minutes
        last_time = self.last_message_was.get(channel_id, datetime.min.replace(tzinfo=timezone.utc))
        return (datetime.now(timezone.utc) - last_time) < timedelta(minutes=2)

    async def setup_hook(self):
        # Initialize the memory handler
        self.memory_handler = await MemoryHandler.create(self)

        async def memory_timer(bot):
            """ Regularly check for new messages to add to memory every 10 minutes."""
            while True:
                await asyncio.sleep(10 * 60)  # Wait for 10 minutes
                await bot.memory_handler.add_memories_task()

        async def seen_messages_timer(bot):
            """ Regularly clear the seen messages list every 30 minutes."""
            while True:
                await asyncio.sleep(30 * 60)  # Wait for 30 minutes
                bot.seen_messages = []

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

        # bot mentioned
        if self.user and self.user.mentioned_in(message):
            logger.debug(f"on_message | Bot Mentioned | {message.content}")
            await self.ask_agent(ctx)

        # bot command
        elif message.content.startswith(self.command_prefix):
            logger.debug(f"on_message | Bot Command | {message.content}")
            await self.process_commands(message)

        elif "http://" in message.content or "https://" in message.content:
            # Extract the domain name from the URL
            parsed_url = urlparse(message.content)
            domain_name = parsed_url.netloc
            if not domain_name:
                return

            # if a domain is in the watched domains, search for it
            if any(domain in domain_name for domain in self.watched_domains):
                res = await search_agent.run(random_search_prompt(message.content),
                    deps=AgentDependencies(bot=self, ctx=ctx, memories=[]),
                    usage_limits=UsageLimits(request_limit=5),
                    output_type=str,
                    message_history=None) # type: ignore

                # append the result to the extra memories
                if res and res.output:
                    logger.debug(f"on_message | URL Result: {res.output}")
                    self.extra_memories.append({
                        "content": res.output,
                        "user_id": str(ctx.author.id),
                        "channel_id": str(ctx.channel.id),
                    }) # type: ignore

                    await ctx.channel.send(res.output) # type: ignore

        # random event
        elif random.random() < 0.05:
            logger.debug("on_message | Random Event")

            # Check if the channel is watched
            if ctx.channel.id not in self.watched_channels:
                logger.debug(f"on_message | Channel {ctx.channel.id} not watched, skipping random event.")
                return

            # get the message history for the channel, if it doesn't exist, get the last 5 messages
            if not (message_history := self.message_history.get(ctx.channel.id, [])):
                message_history = await self.get_message_history(ctx)

            msg = ctx.message.content
            res = await true_false_agent.run("Does the following message contain anything worth replying to? \n\n"
                                             + msg + " /no_think", message_history=message_history) # type: ignore

            # if the msg is worth replying to
            if res.output.result:
                logger.debug("on_message | Generating Random Event Message")

                res = await self._agent_run(
                    random_message_prompt(msg),
                    AgentDependencies(bot=self, ctx=ctx, memories=[]),
                    message_history=message_history
                )

                if res and res.output and res.output.content:
                    logger.debug(f"on_message | Random Event Result: {res.output.content}")
                    await ctx.send(res.output.content)

        # Add the message to the message history
        if self.active_conversation(ctx.channel.id) and self.is_valid_message(message):
            self.message_history[ctx.channel.id].append(sys_msg(message.content))

    @staticmethod
    def update_message_history(history: list[str], new: list[str], max_length: int = 20) -> list[str]:
        """
        Trim the message history to the last `max_length` messages.
        """
        if len(history) > max_length:
            return history[-max_length:]
        history = history + [m for m in new if m not in history]
        return history


    def is_valid_message(self, message: discord.Message) -> bool:
        """
        Check if the message is valid for processing.
        """
        # ignore commands that aren't the chat command
        if message.content.startswith(str(self.command_prefix)) and \
            not message.content.startswith(f"{self.command_prefix}chat"):
            return False
        if message.content.startswith("BOT:"):
            return False

        # skip code blocks
        if "```" in message.content:
            return False
        # skip messages with URLs
        if "http://" in message.content or "https://" in message.content:
            return False

        # Ignore messages with embeds
        if len(message.embeds) > 0:
            return False

        # Ignore messages that are too short (lol, etc)
        if len(message.content.strip()) < 5:
            return False

        return True


    def check_message_history(self, ctx: commands.Context):
        """
        Check if the message history needs to be reset based on the last message time.
        """
        # Reset message history if it has been more than 5 minutes since the last message to the agent.
        last_message = self.last_message_was.get(ctx.channel.id, datetime.min.replace(tzinfo=timezone.utc))

        if datetime.now(timezone.utc) - timedelta(minutes=5) > last_message :
            self.message_history[ctx.channel.id] = []

    async def get_message_history(self, ctx: commands.Context) -> list[ModelMessage]:
        """
        Get the message history for the current channel.
        If the history is empty, fetch the last 5 messages.
        """

        messages = []
        # If the message history is empty, fetch the last 5 messages
        if not self.message_history[ctx.channel.id]:
            async for m in ctx.channel.history(limit=5):
                if not self.is_valid_message(m):
                    continue
                # remove the command keyword from the beginning of the message
                if m.content.startswith(f"{self.command_prefix}chat"):
                    msg = m.content[5:].strip()
                else:
                    msg = m.content.strip()

                msg = sys_msg(msg)
                messages.append(msg)

            self.message_history[ctx.channel.id] = messages

            return messages
        else:
            messages = self.message_history[ctx.channel.id]
            return messages

    def add_message_to_chat_history(self, ctx: commands.Context, result: AgentRunResult[Any]) -> None:
        """
        Add a message to the bot's message history.
        """
        self.message_history[ctx.channel.id].extend(result.new_messages())
        # Limit the message history to the last 10 messages
        if self.message_history[ctx.channel.id] and len(self.message_history[ctx.channel.id]) > 10:
            self.message_history[ctx.channel.id] = self.message_history[ctx.channel.id][-10:]

    async def ask_agent(self, ctx: commands.Context):
        """
        Ask the AI agent a question.

        Args:
            ctx (commands.Context): the discord context.
        """
        # Reset message history if it has been more than 5 minutes since the last message to the agent.
        self.check_message_history(ctx)
        self.last_message_was[ctx.channel.id] = datetime.now(timezone.utc)
        self.message_history[ctx.channel.id] = await self.get_message_history(ctx)

        async with ctx.typing():
            user_id = str(self.user.id) if self.user and self.user.id else ""
            query = escape_mentions(remove_command_prefix(ctx.message.content, prefix=ctx.prefix if ctx.prefix else "!"))

            memories = []
            memory_results = await self.memory_handler.memory.search(query=query, agent_id=user_id, limit=8)
            for entry in memory_results["results"]:
                if entry and "memory" in entry:
                    memories.append(entry["memory"].format(user=ctx.author.display_name if ctx.author else "User"))

            result = await self._agent_run(query, AgentDependencies(bot=self, ctx=ctx, memories=memories))
            if not result:
                await ctx.send("Sorry, I couldn't process your request.")
                return
            self.add_message_to_chat_history(ctx, result)
            await ctx.send(html.unescape(result.output.response))

    async def _agent_run(self,
                         query: str,
                         deps: AgentDependencies,
                         message_history: list[ModelMessage] = None # type: ignore
                         ) -> AgentRunResult[Any] | None:
        """
        Run the agent with the given query and dependencies and limits.
        """
        try:
            while True:
                logger.debug(f"Running agent with query: {query}")
                # failsafe check for message history
                if message_history:
                    history = message_history
                else:
                    if not (history := self.message_history.get(deps.channel.id if deps.channel else 0, [])):
                        history = await self.get_message_history(deps.context) # type: ignore

                agent_run = await self.agent.run(query, deps=deps,
                                    usage_limits=UsageLimits(request_limit=5),
                                    model_settings=MODEL_SETTINGS,
                                    message_history=history
                                    )

                # reset the bot's status to online after the agent run in case any tools changed it
                if self.status != discord.Status.dnd:
                    await self.change_presence(status=discord.Status.online) # type: ignore

                if agent_run and agent_run.output:
                    if isinstance(agent_run.output, FollowUpQuestion):
                        logger.debug(f"Follow-Up Question: {agent_run.output.question}")
                        # Ask the user a follow-up question
                        await deps.context.channel.send(agent_run.output.question) # type: ignore
                        # Wait for the user's response
                        try:
                            logger.debug("Waiting for user response to follow-up question...")
                            response = await self.wait_for('message', timeout=60.0, check=lambda m: m.author == deps.context.author) # type: ignore
                            # append the follow up question and the user's response to the message history
                            self.message_history[deps.channel.id].append(agent_run.new_messages()[0]) # type: ignore
                            self.message_history[deps.channel.id].append(user_msg(response.content)) # type: ignore
                            continue
                        except asyncio.TimeoutError:
                            logger.debug("No response received for clarification question.")
                            return agent_run
                    else:
                        break
            return agent_run
        except Exception as e:
            logger.error(f"Agent run failed: {e}")
            if self.status != discord.Status.dnd:
                await self.change_presence(status=discord.Status.online) # type: ignore
            return None

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
    bot.message_history[ctx.channel.id] = []
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

    if bot.memory_handler.memory.vector_store.client:
        client = bot.memory_handler.memory.vector_store.client
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

    memory = await bot.memory_handler.memory.search(query=memory_content, agent_id=str(bot.user.id if bot.user else ""), limit=1)
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
                await bot.memory_handler.memory.delete(mem_entry["id"])
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
    memory = await bot.memory_handler.memory.search(query=query, agent_id=str(bot.user.id if bot.user else ""), limit=20)
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

