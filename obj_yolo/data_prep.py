import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
BASE_DATA_PATH = os.getenv("BASE_DATA_PATH")
if not BASE_DATA_PATH or not os.path.exists(BASE_DATA_PATH):
    raise Exception("Base data path not found")


