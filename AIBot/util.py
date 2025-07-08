import asyncio
import os
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

async def check_facts(bot, messages) -> dict[int, FactResponse]:
    """
    Check the messages for any facts that should be remembered.
    """
    if not messages:
        return {}

    parsed: dict[int, list[dict[str, str]]] = {}
    for channel_id, msgs in messages.items():
        parsed[channel_id] = []
        logger.debug(f"Checking {len(msgs)} for facts in channel {channel_id}")
        for msg in msgs:
            parsed[channel_id].append({"role": "assistant" if bot.user and msg.author.id == bot.user.id else "user", 
                 "content": msg.content, 
                 "user_id": str(msg.author.id)})
    
    print(parsed)
    output = {}
    for c, msgs in parsed.items():
        prompt = memory_fact_prompt(msgs)
        res = await bot.memory_agent.run(prompt, deps=None, output_type=FactResponse)
        if res:
            output[c] = res.output

    if output:
        return output

    return {}

async def add_memories(bot, messages: dict[int, FactResponse]) -> list[str]:
    '''
    Adds messages to the bot's memories and returns the results.
    '''
    if not messages:
        logger.debug("No messages to add to memory.")
        return []
    
    result_msgs = []
    for channel_id, msgs in messages.items():
        msgs_to_add = []
        logger.debug(f"add_memories | Processing {len(msgs.facts)} facts in channel {channel_id}")
        msgs_to_add.extend(
            {"role": "assistant" if bot.user and m.user_id == bot.user.id else "user", "content": m.content}
            for m in msgs.facts if m.user_id not in bot.seen_messages and not bot.seen_messages.append(m.user_id)
        )

        res = await bot.memory.add(msgs_to_add, 
                                   agent_id=str(bot.user.id) if bot.user and bot.user.id else "", 
                                   metadata={"user_id": str(bot.user.id) if bot.user and bot.user.id else ""})
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