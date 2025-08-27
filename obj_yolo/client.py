# client.py
import requests
import torch
import io
import logging
from ultralytics import YOLO
from pathlib import Path
import time

# local imports
from strategy import FedAvg, FedWeg

# import configuration
from config import (
    MODEL_PATH,

    CLIENTS_COUNT,
    CLIENT_DATA_COUNT,
    CLIENT_EPOCHS,
    CLIENT_DATA_PATH,

    SERVER_HOST,
    SERVER_PORT,
    STRATEGY
)

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [Client %(client_id)s] - %(message)s')

class ClientApp:
    def __init__(self, client_id: int):
        self.client_id = client_id
        self.model = YOLO(MODEL_PATH)
        self.server_url = f"http://{SERVER_HOST}:{SERVER_PORT}"
        self.data_path = Path(CLIENT_DATA_PATH) / f"client_{self.client_id}" / "data.yaml"
        self.server_state = None

    def get_global_model(self):
        """
        Fetches the global model from the server and loads only the weights
        for layers that have matching shapes, ignoring the final detection head
        if the number of classes differs.
        """
        try:
            logging.info("Requesting global model from server...")
            response = requests.get(f"{self.server_url}/get_parameters", timeout=60)
            response.raise_for_status()

            # Load the state_dict sent by the server
            params_buffer = io.BytesIO(response.content)
            self.server_state = torch.load(params_buffer)

            # update model state dict directly
            try:
                self.model.load_state_dict(self.server_state, strict=False)
            except Exception as e:
                logging.error(f"could not adapt model: {e}")

        except requests.exceptions.RequestException as e:
            logging.error(f"Could not fetch model from server: {e}")
            raise

    def train(self):
        """Trains the model on the client's local data."""
        if not Path(self.data_path).exists():
            logging.error(f"Data path not found: {self.data_path}. Please check your config.")
            return None
        
        logging.info(f"Starting local training for {CLIENT_EPOCHS} epochs.")
        
        if STRATEGY == "fedavg":
            logging.info(f"Starting local training for {CLIENT_EPOCHS} epochs (FedAvg).")
            client_state_dict = FedAvg.client_train(model=self.model, client_id=self.client_id, epochs=CLIENT_EPOCHS, data_path=self.data_path)
            return client_state_dict
        elif STRATEGY == "fedweg":
            logging.info(f"Starting local training for {CLIENT_EPOCHS} epochs (FedWeg).")
            sparse_update, masks = FedWeg.client_train(self.model, self.server_state, self.client_id, CLIENT_EPOCHS, self.data_path)
            return (sparse_update, masks)
        else:
            raise ValueError(f"Unknown strategy {STRATEGY}")

    def send_update(self, update):
        if update is None:
            return
        logging.info("Sending model update to the server...")
        buffer = io.BytesIO()
        torch.save(update, buffer)
        buffer.seek(0)
        try:
            files = {'model_file': ('client_model.pth', buffer, 'application/octet-stream')}
            data = {'data_count': CLIENT_DATA_COUNT}
            response = requests.post(f"{self.server_url}/submit_update", files=files, data=data, timeout=120)
            response.raise_for_status()
            logging.info(f"Server response: {response.json()}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to send update to server: {e}")

def run_single_client(client_id: int):
    """Simulates the full lifecycle of a single client."""
    client = ClientApp(client_id=client_id)
    client.get_global_model()
    updated_state_dict = client.train()
    client.send_update(updated_state_dict)

if __name__ == "__main__":
    print(f"\n--- Starting Simulation for {CLIENTS_COUNT} Clients ---")
    for i in range(CLIENTS_COUNT):
        print(f"\n--- Running Client {i} ---")
        run_single_client(i)
        time.sleep(2)
        