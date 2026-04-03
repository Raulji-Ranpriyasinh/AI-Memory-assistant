import logging
import os
from datetime import datetime

# Create logs directory in project root
LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Log file name with date
log_filename = os.path.join(
    LOGS_DIR, f"memory_{datetime.now().strftime('%Y-%m-%d')}.log"
)

# Configure logger
logger = logging.getLogger("ai_memory")
logger.setLevel(logging.DEBUG)  # Change to INFO in production

# File handler
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers (avoid duplicate handlers)
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
