import os
import logging
from AIBot.bot import bot
from AIBot.config import config

os.environ["MEM0_TELEMETRY"] = "False"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

logging.getLogger("AIBot").setLevel(logging.DEBUG)

TOKEN = config.get("DISCORD", {}).get("token")

if __name__ == "__main__":
    try:
        bot.run(TOKEN or "")
    except Exception as e:
        logging.error(f"Failed to start the bot: {e}")
        raise e
    