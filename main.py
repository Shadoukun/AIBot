import os
import logging
from AIBot.bot import bot
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

logging.getLogger("AIBot").setLevel(logging.DEBUG)

TOKEN = os.getenv("TOKEN")

if __name__ == "__main__":
    try:
        bot.run(TOKEN or "")
    except Exception as e:
        logging.error(f"Failed to start the bot: {e}")
        raise e
    