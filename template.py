import os
from pathlib import Path
import logging

# Set up logging to see what the script is doing
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

list_of_files = [
    "src/__init__.py",
    "src/helper.py",
    ".env",
    "app.py",
    "store_index.py",
    "static/style.css",
    "templates/chat.html"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    # Create directories if they don't exist
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    # Create empty files if they don't exist
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")
