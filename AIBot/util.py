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

async def add_memory(bot, fact: Fact):
    if fact.text.strip():
        # add the fact to memory
        memory_result = await bot.memory.add(
            fact.text.strip(),
            agent_id=str(bot.user.id) if bot.user and bot.user.id else "")
        if memory_result:
            logger.debug(f"Memory added successfully: {memory_result}")
            embed = discord.Embed(
                title="Memory Added",
                description=f"*{fact.text.strip()}*",
                color=discord.Color.blue()
            )
            
            await bot.bot_channel.send(embed=embed)

def is_bot_announcement(msg) -> bool:
    """
    Check if the message is from the bot.
    """
    return msg.content.startswith("BOT: ")