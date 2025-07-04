import asyncio
import os
from discord.ext import commands

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
        await bot.random_memory_check()  # Run the memory check function

async def seen_messages_timer(bot):
    while True:
        await asyncio.sleep(1800)  # Wait for 30 minutes
        bot.seen_messages = []  # Clear the seen messages every 30 minutes