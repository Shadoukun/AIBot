from datetime import datetime, timezone
import discord
import logging
from typing import Callable
from pydantic_ai.messages import ModelRequest, UserPromptPart, SystemPromptPart

logger = logging.getLogger(__name__)

def is_bot_announcement(msg: discord.Message) -> bool:
    """
    Check if the message is a preformatted bot announcement.
    """
    return msg.content.startswith("BOT: ")

def remove_command_prefix(msg, prefix='!') -> str:
    """
    Remove the command prefix from the message.
    """
    if msg.startswith(prefix):
        msg = " ".join(msg.split()[1:])
    return msg

async def is_admin(ctx, *args, **kwargs) -> Callable:
    """
    Decorator to check if the user is an admin.
    """
    async def decorator(func):
        async def wrapper(self, ctx, *args, **kwargs):
            if ctx.author.id in self.bot.config.get("DISCORD", {}).get("bot_admins", []):
                return await func(self, ctx, *args, **kwargs)
            await ctx.send("You do not have permission to use this command.")
        return wrapper
    return decorator

def user_msg(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=text, timestamp=datetime.now(timezone.utc))])

def sys_msg(text: str) -> ModelRequest:
    return ModelRequest(parts=[SystemPromptPart(content=text, timestamp=datetime.now(timezone.utc))])