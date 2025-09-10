# server.py
import logging
import os
import random
from collections import OrderedDict
from typing import List

from ultralytics import YOLO

from strategy import Strategy
from client import Client, ClientManager, FitRes
from dataset import load_data
from utils import set_parameters, get_whole_parameters

class Server:
    """
    The Server is the central orchestrator of the federated learning process.
    """
    def __init__(self, strategy: Strategy, yolo_config: str, communication_rounds: int = 1):
        self.strategy = strategy
        self.communication_rounds = communication_rounds
        self.clients: List[ClientManager] = []

        self.global_model = YOLO(yolo_config).load("yolo11n.pt")
        self.global_state = {k:v.clone() for k, v in self.global_model.model.model.state_dict().items()}

    def add_client(self, client_manager: ClientManager) -> None:
        self.clients.append(client_manager)

    def update_global_state(self, new_state: OrderedDict) -> None:
        """
        Updates the global model's state dictionary.
        """
        self.global_model = set_parameters(
            self.global_model,
            new_state,
            name="Global Server"
        )
        self.global_state = {k:v.clone() for k, v in self.global_model.model.model.state_dict().items()}

    def run(self, base_path: str, client_base_path: str, client_data_count: int) -> None:
        """
        Executes the entire federated learning process for the configured number of rounds.
        """
        logging.info("Starting federated learning process...")

        # --- NEW: Centralized Data Partitioning ---
        # 1. Get all image filenames from the base directory.
        image_dir = os.path.join(base_path, "image_2")
        all_images = os.listdir(image_dir)
        random.shuffle(all_images)

        # 2. Create unique data partitions for each client.
        client_partitions = {}
        start_idx = 0
        val_count = 50 # Number of validation samples per client
        for client_manager in self.clients:
            client_id = client_manager.client_id
            end_idx = start_idx + client_data_count
            
            # Ensure we don't run out of images
            if end_idx + val_count > len(all_images):
                raise ValueError("Not enough images in the base dataset to partition among all clients.")

            train_split = all_images[start_idx:end_idx]
            val_split = all_images[end_idx:end_idx + val_count]
            client_partitions[client_id] = (train_split, val_split)
            
            # Move the index for the next client's partition
            start_idx = end_idx + val_count
        logging.info(f"Created {len(client_partitions)} unique data partitions.")
        # --- END NEW SECTION ---

        for i in range(self.communication_rounds):
            round_num = i + 1
            logging.info(f"--- Starting Communication Round {round_num}/{self.communication_rounds} ---")

            # 1. Client Selection
            selected_clients = self.strategy.sample(available_clients=self.clients)
            logging.info(f"Selected {len(selected_clients)} clients for this round: {[c.client_id for c in selected_clients]}")

            results: List[FitRes] = []
            data_paths = {}

            # 2. Client Training
            for client_manager in selected_clients:
                # Update client model with the latest global state
                client_fn = Client(client_manager=client_manager)
                client_fn.update_client_model(parameters=self.global_state)
                logging.info(f"Starting training for client {client_manager.client_id}")

                # Get the client's unique file lists from the partition map
                train_files, val_files = client_partitions[client_manager.client_id]

                data_path = load_data(
                    base_path=base_path,
                    client_base_path=client_base_path,
                    client_id=client_manager.client_id,
                    train_images=train_files, # Pass the unique training file list
                    val_images=val_files    # Pass the unique validation file list
                )
                data_paths[client_manager.client_id] = data_path
                logging.info(f"Prepared data for client {client_manager.client_id} at {data_path}")

                res = client_fn.fit(data_path=str(data_path))
                results.append(res)

                logging.info(f"Client {client_manager.client_id} training complete. mAP50-95: {res.results['map50-95']:.4f}")

            # 3. Aggregation
            if not results:
                logging.warning("No results to aggregate. Skipping round.")
                continue

            logging.info("Aggregation started.")
            aggregated_state = self.strategy.aggregate_fit(results=results, global_state=self.global_state)
            self.update_global_state(new_state=aggregated_state)
            logging.info("Aggregation complete. Global model updated.")

            # 4. Evaluation and State Update
            for client_manager in selected_clients:
                client_fn = Client(client_manager=client_manager)
                data_path = data_paths[client_manager.client_id]

                logging.info(f"Validation started for client {client_manager.client_id}")
                val_res = client_fn.evaluate(paramaters=self.global_state, data_path=str(data_path))
                logging.info(f"Validation results for client {client_manager.client_id}: mAP50-95: {val_res['mAP50-95']:.4f}")

                # Update client state for the next round
                client_fn.update_client_model(self.global_state)
                client_manager.rounds_completed += 1
                self.strategy.update_sparsity(manager=client_manager)
                logging.info(f"Client {client_manager.client_id} sparsity for next round: {client_manager.fit_params.sparsity:.2f}")

        logging.info("--- Federated learning process complete. ---")
