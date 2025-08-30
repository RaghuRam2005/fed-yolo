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
            logging.info(f"Client {self.client_id}: Requesting global model from server...")
            response = requests.get(f"{self.server_url}/get_parameters")
            response.raise_for_status()
            
            # Load the state_dict sent by the server
            params_buffer = io.BytesIO(response.content)
            self.server_state = torch.load(params_buffer)
            
            logging.info(f"Client {self.client_id}: Received server state with {len(self.server_state)} parameters")
            
            # Get current client model state for comparison
            client_state_before = self.model.state_dict()
            logging.info(f"Client {self.client_id}: Current client model has {len(client_state_before)} parameters")
            
            try:
                # Load with strict=False to handle mismatched keys gracefully
                missing_keys, unexpected_keys = self.model.load_state_dict(self.server_state, strict=False)
                
                # Log loading results
                loaded_count = len(self.server_state) - len(missing_keys)
                logging.info(f"Client {self.client_id}: Model state loading summary:")
                logging.info(f"  - Successfully loaded: {loaded_count} parameters")
                logging.info(f"  - Missing keys (not loaded): {len(missing_keys)}")
                logging.info(f"  - Unexpected keys (ignored): {len(unexpected_keys)}")
                
            except Exception as e:
                logging.error(f"Client {self.client_id}: Could not load server state into model: {e}")
                logging.error(f"Client {self.client_id}: This may indicate severe model architecture mismatch")
                raise
                
        except requests.exceptions.RequestException as e:
            logging.error(f"Client {self.client_id}: Could not fetch model from server: {e}")
            raise

    def train(self):
        """Trains the model on the client's local data."""
        if not Path(self.data_path).exists():
            logging.error(f"Data path not found: {self.data_path}. Please check your config.")
            return None

        logging.info(f"Starting local training for {CLIENT_EPOCHS} epochs.")
        
        if STRATEGY == "fedavg":
            logging.info(f"Starting local training for {CLIENT_EPOCHS} epochs (FedAvg).")
            client_state_dict = FedAvg.client_train(
                model=self.model, 
                client_id=self.client_id, 
                epochs=CLIENT_EPOCHS, 
                data_path=self.data_path
            )
            return client_state_dict
            
        elif STRATEGY == "fedweg":
            logging.info(f"Starting local training for {CLIENT_EPOCHS} epochs (FedWeg).")
            pruned_state, sparsity_stats = FedWeg.client_train(
                self.model, 
                self.server_state, 
                self.client_id, 
                CLIENT_EPOCHS, 
                self.data_path,
            )
            return (pruned_state, sparsity_stats)
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
            response = requests.post(f"{self.server_url}/submit_update", files=files, data=data)
            response.raise_for_status()
            logging.info(f"Server response: {response.json()}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to send update to server: {e}")

def run_single_client(client_id: int):
    """Simulates the full lifecycle of a single client for one communication round."""
    client = ClientApp(client_id=client_id)
    
    # Step 1: Request global model parameters from server
    client.get_global_model()
    
    # Step 2: Train locally with the global model
    updated_artifacts = client.train()
    
    # Step 3: Send local updates to server
    client.send_update(updated_artifacts)

def run_federated_learning():
    """Runs the complete federated learning process with multiple communication rounds."""
    print(f"--- Starting Federated Learning with {CLIENTS_COUNT} clients")
    for client_id in range(CLIENTS_COUNT):
        print(f"\n--- Client {client_id} Training ---")
        run_single_client(
            client_id=client_id, 
        )
        time.sleep(1)  # Small delay between clients
    
if __name__ == "__main__":
    run_federated_learning()
