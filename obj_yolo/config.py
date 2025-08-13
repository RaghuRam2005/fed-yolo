# config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- General ---
# The base model to start with (e.g., 'yolov8n.pt')
MODEL_NAME = r"C:\Users\lingu\study\projects\obj_yolo\obj_yolo\yolo11n_baseline.yaml"
# The number of clients to simulate in the federation
CLIENTS = 2

# --- Server Configuration ---
SERVER_HOST = "localhost"
SERVER_PORT = 5000
# Path to the global validation dataset YAML file
GLOBAL_DATA_PATH = os.getenv("GLOBAL_DATA_PATH")
# Number of epochs for initial centralized training (optional, can be 0)
GLOBAL_EPOCHS = 50

# --- Client Configuration ---
# Path where all client data partitions are stored
BASE_CLIENT_DATA_PATH = os.getenv("CLIENT_DATA_PATH")
# Number of local epochs for each client to train
LOCAL_EPOCHS = 3
# Number of data samples each client has (for weighted average)
# In a real scenario, clients would report this themselves. We simulate it here.
CLIENT_DATA_COUNT = 50