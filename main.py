import os
import logging
from bot import bot
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
)

TOKEN = os.getenv("TOKEN")

if __name__ == "__main__":
    bot.run(TOKEN)