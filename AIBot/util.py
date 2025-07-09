import asyncio
from datetime import datetime, timedelta, timezone
import os
from typing import Any, Union
import discord
from discord.ext import commands
import logging
from .prompts import memory_fact_prompt
from .models import FactResponse

logger = logging.getLogger(__name__)

PRINT_THINKING = os.getenv("PRINT_THINKING", "True").lower() 

def remove_command_prefix(msg, prefix='!') -> str:
    """
    Remove the command prefix from the message.
    """
    if msg.startswith(prefix):
        msg = " ".join(msg.split()[1:])
    return msg

def print_thinking(ctx: commands.Context) -> bool:
    """Print the thinking node if PRINT_THINKING is enabled."""
    if PRINT_THINKING == "true" or ctx.message.content.endswith("/printthink"):
        return True
    
    return False

def update_message_history(history: list[str], new: list[str], max_length: int = 20) -> list[str]:
    """
    Trim the message history to the last `max_length` messages.
    """
    if len(history) > max_length:
        return history[-max_length:]
    history = history + [m for m in new if m not in history]
    return history

async def memory_timer(bot):
    while True:
        await asyncio.sleep(300)  # Wait for 5 minutes
        await bot.add_memories_task()  # Run the memory check function

async def seen_messages_timer(bot):
    while True:
        await asyncio.sleep(1800)  # Wait for 30 minutes
        bot.seen_messages = []  # Clear the seen messages every 30 minutes

async def check_facts(bot, messages: dict[int, list[discord.Message]]) -> dict[int, FactResponse]:
    """
    Check the messages for any facts that should be remembered.
    """
    parsed: dict[int, list[dict[str, str]]] = {}
    for channel_id, msgs in messages.items():
        parsed[channel_id] = []
        logger.debug(f"Checking {len(msgs)} for facts in channel {channel_id}")
        for msg in msgs:
            parsed[channel_id].append({"role": "assistant" if bot.user and msg.author.id == bot.user.id else "user", 
                 "content": msg.content, 
                 "user_id": str(msg.author.id)})
    
    output = {}
    for c, msgs in parsed.items():
        prompt = memory_fact_prompt(msgs)
        res = await bot.memory_agent.run(prompt, deps=None, output_type=FactResponse)
        if res:
            output[c] = res.output

    if output:
        return output

    return {}

async def add_memories(bot, messages: dict[int, list[discord.Message]]) -> list[str]:
    '''
    Adds messages to the bot's memories and returns the results.
    '''
    
    # checks each channels messages for facts, returns a {channel_id: FactResponse}
    fact_res: dict[int, FactResponse] = await check_facts(bot, messages)
    if not fact_res:
        logger.debug("No facts found in watched messages.")
        return []

    result_msgs = []
    for channel_id, msgs in fact_res.items():
        logger.debug(f"add_memories | Processing {len(msgs.facts)} facts in channel {channel_id}")
        
        msgs_to_add = []
        msgs_to_add.append({"role": "assistant" if bot.user and m.user_id == bot.user.id else "user", 
                            "content": m.content, 
                            "user_id": m.user_id}
                            for m in msgs.facts)
        
        if not msgs_to_add:
            logger.debug(f"No messages to add for channel {channel_id}.")
            continue

        res = await bot.memory.add(msgs_to_add,
                                   agent_id=str(bot.user.id) if bot.user and bot.user.id else "",
                                   metadata={"user_id": str(m["user_id"]) for m in msgs_to_add})

        if isinstance(res, dict):
            results = res.get("results", [])
            for result in results:
                result_msgs.append(f"Memory: {result['memory']} Event: {result['event']}")
    
    return result_msgs

def format_memory_messages(messages: dict[int, FactResponse]) -> list[dict[str, str]]:
    """
    Format the messages for memory addition.
    """
    formatted = []
    for channel_id, msgs in messages.items():
        for fact in msgs.facts:
            formatted.append({
                "role": "assistant" if fact.user_id == str(channel_id) else "user",
                "content": fact.content,
                "user_id": fact.user_id
            })
    return formatted

async def check_watched_channels(bot) -> dict[int, list[discord.Message]]:
    logger.debug("Checking watched channels for new messages...")
   
    after = datetime.now(timezone.utc) - timedelta(minutes=5)
    watched_msgs: dict[int, list[discord.Message]] = {}
    for c in bot.watched_channels:
        channel: Union[discord.TextChannel, discord.VoiceChannel, None] = bot.get_channel(c)
        if isinstance(channel, discord.TextChannel):
            watched_msgs[c] = [
                m async for m in channel.history(after=after)
                if not is_bot_announcement(m)
                and not m.content.startswith(bot.command_prefix)  # type: ignore
                and m.id not in bot.seen_messages
            ]

    return watched_msgs

def is_bot_announcement(msg) -> bool:

    """
    Check if the message is from the bot.
    """
    return msg.content.startswith("BOT: ")