import os
import logging
from pathlib import Path

BASE_REPO_DIR = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")).resolve()

# Data Paths
BASE_DATA_PATH = Path(os.path.join(BASE_REPO_DIR, "base_data"))
# data path for global model train and val data
GLOBAL_DATA_PATH = Path(os.path.join(BASE_REPO_DIR, "prepared_data"))
# data path for client model train and val data
CLIENT_DATA_PATH = Path(os.path.join(BASE_REPO_DIR, "prepared_data", "clients"))
# model file path
MODEL_PATH = Path(os.path.join(BASE_REPO_DIR, "yolo_config", "yolo11n.yaml"))

# data configurations
DATA_SPLIT = 0.8

# global configuations
GLOBAL_DATA_COUNT = 100
GLOBAL_EPOCHS = 2

# client configurations
CLIENTS_COUNT = 2
CLIENT_DATA_COUNT = 100
CLIENT_EPOCHS = 2

# classes configuations
NC = 8
CLASSES = ["Car", "Pedestrian", "Van", "Cyclist", "Truck", "Misc", "Tram", "Person_sitting"]

# server configurations
SERVER_HOST = "localhost"
SERVER_PORT = 5000
STRATEGY = "fedweg"
