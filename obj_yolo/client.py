# client.py
import requests
import torch
import io
import logging
import os
from ultralytics import YOLO
from pathlib import Path
import time

# Import configuration
from config import (
    MODEL_NAME,
    SERVER_HOST,
    SERVER_PORT,
    BASE_CLIENT_DATA_PATH,
    LOCAL_EPOCHS,
    CLIENT_DATA_COUNT
)

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [Client %(client_id)s] - %(message)s')

class YoloClient:
    def __init__(self, client_id: int):
        self.client_id = client_id
        self.model = YOLO(MODEL_NAME)
        self.server_url = f"http://{SERVER_HOST}:{SERVER_PORT}"
        self.data_path = str(Path(BASE_CLIENT_DATA_PATH) / f"client_{self.client_id}" / "data.yaml")
        self.logger = logging.getLogger(__name__)
        self.log_adapter = logging.LoggerAdapter(self.logger, {'client_id': self.client_id})

    def get_global_model(self):
        """
        Fetches the global model from the server and loads only the weights
        for layers that have matching shapes, ignoring the final detection head
        if the number of classes differs.
        """
        try:
            self.log_adapter.info("Requesting global model from server...")
            response = requests.get(f"{self.server_url}/get_parameters", timeout=60)
            response.raise_for_status()

            # Load the state_dict sent by the server
            params_buffer = io.BytesIO(response.content)
            server_state_dict = torch.load(params_buffer)

            # Get the client's current model state
            client_state_dict = self.model.model.state_dict()

            # Create a new state_dict to load, containing only matching layers
            filtered_state_dict = {
                k: v for k, v in server_state_dict.items()
                if k in client_state_dict and v.shape == client_state_dict[k].shape
            }

            # Load the filtered state_dict. strict=False is safer.
            self.model.model.load_state_dict(filtered_state_dict, strict=False)

            num_loaded = len(filtered_state_dict)
            num_total = len(server_state_dict)
            if num_loaded < num_total:
                self.log_adapter.warning(
                    f"Mismatch detected. Loaded {num_loaded}/{num_total} layers from the server. "
                    "This is expected if the number of classes differs."
                )
            else:
                self.log_adapter.info("Successfully loaded all model layers from the server.")

        except requests.exceptions.RequestException as e:
            self.log_adapter.error(f"Could not fetch model from server: {e}")
            raise

    def train(self):
        """Trains the model on the client's local data."""
        if not Path(self.data_path).exists():
            self.log_adapter.error(f"Data path not found: {self.data_path}. Please check your config.")
            return None
        
        self.log_adapter.info(f"Starting local training for {LOCAL_EPOCHS} epochs.")
        
        # The train function will automatically adapt the model head to the number
        # of classes in self.data_path if it hasn't been adapted already.
        self.model.train(
            data=self.data_path,
            epochs=LOCAL_EPOCHS,
            batch=8,
            save=False,
            project="fed_yolo",
            name=f"client_{self.client_id}_train",
            exist_ok=True,
            device=-1,
            cos_lr=True,
            warmup_epochs=3,

            augment=True,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0
        )
        
        self.log_adapter.info("Local training complete.")
        return self.model.model.state_dict()

    def send_update(self, state_dict):
        """Sends the updated model parameters to the server."""
        if state_dict is None:
            return

        self.log_adapter.info("Sending model update to the server...")
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        buffer.seek(0)
        
        try:
            files = {'model_file': ('client_model.pth', buffer, 'application/octet-stream')}
            data = {'data_count': CLIENT_DATA_COUNT}
            response = requests.post(f"{self.server_url}/submit_update", files=files, data=data, timeout=120)
            response.raise_for_status()
            self.log_adapter.info(f"Server response: {response.json()}")
        except requests.exceptions.RequestException as e:
            self.log_adapter.error(f"Failed to send update to server: {e}")

def run_single_client(client_id: int):
    """Simulates the full lifecycle of a single client."""
    client = YoloClient(client_id=client_id)
    client.get_global_model()
    updated_state_dict = client.train()
    client.send_update(updated_state_dict)

if __name__ == "__main__":
    from config import CLIENTS
    print(f"\n--- Starting Simulation for {CLIENTS} Clients ---")
    print("NOTE: Run server.py in a separate terminal before running the clients.")
    for i in range(CLIENTS):
        print(f"\n--- Running Client {i} ---")
        run_single_client(i)
        time.sleep(2)