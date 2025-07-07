import asyncio
import os
import discord
from discord.ext import commands
import logging

from AIBot.models import Fact
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
        await asyncio.sleep(120)  # Wait for 2 minutes
        await bot.memory_check()  # Run the memory check function

async def seen_messages_timer(bot):
    while True:
        await asyncio.sleep(1800)  # Wait for 30 minutes
        bot.seen_messages = []  # Clear the seen messages every 30 minutes

async def add_memories(bot, messages):
    '''
    Adds messages to the bot's memories and returns the results.
    '''
    
    result_msgs = []
    for channel_id, msgs in messages.items():
        msgs_to_add = []
        logger.debug(f"Processing {len(msgs)} messages in channel {channel_id}")
        msgs_to_add.extend(
            {"role": "assistant" if bot.user and m.author.id == bot.user.id else "user", "content": m.content}
            for m in msgs if m.id not in bot.seen_messages and not bot.seen_messages.append(m.id)
        )

        res = await bot.memory.add(msgs_to_add, agent_id=str(bot.user.id) if bot.user and bot.user.id else "")
        if isinstance(res, dict):
            results = res.get("results", [])
            for result in results:
                result_msgs.append(f"Memory: {result['memory']} Event: {result['event']}")
    
    return result_msgs

def is_bot_announcement(msg) -> bool:
    """
    Check if the message is from the bot.
    """
    return msg.content.startswith("BOT: ")