# simulation.py
import logging
from dataclasses import dataclass

from server import Server
from client import ClientManager, ClientFitParams
from strategy import Strategy
from ultralytics import YOLO

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class SimulationConfig:
    """
    Dataclass to hold all simulation configuration parameters.
    """
    yolo_config: str
    base_data_path: str
    client_data_path: str
    num_clients: int = 3
    communication_rounds: int = 2
    epochs_per_round: int = 10
    client_data_count: int = 1000
    min_clients_aggregation: int = 2

def setup_and_run(config: SimulationConfig):
    """
    Sets up the server, clients, and strategy, then starts the simulation.
    """
    # 1. Initialize Strategy
    strategy = Strategy(
        min_clients_aggregation=config.min_clients_aggregation
    )

    # 2. Initialize Server
    server = Server(
        strategy=strategy,
        yolo_config=config.yolo_config,
        communication_rounds=config.communication_rounds
    )

    # 3. Create and Register Clients with the Server
    for i in range(config.num_clients):
        fit_params = ClientFitParams(
            epochs=config.epochs_per_round,
        )
        # Each client gets an instance of the initial model structure from a pre-trained model
        client_model = YOLO(config.yolo_config).load("yolo11n.pt")
        client_manager = ClientManager(
            client_id=i,
            fit_params=fit_params,
            model=client_model
        )
        server.add_client(client_manager)
        logging.info(f"Registered client {i} with the server.")

    # 4. Run the simulation
    logging.info("Configuration complete. Handing control to the server.")
    server.run(
        base_path=config.base_data_path,
        client_base_path=config.client_data_path,
        client_data_count=config.client_data_count
    )

if __name__ == "__main__":
    # Define paths and configuration
    sim_config = SimulationConfig(
        yolo_config="C:\\Users\\lingu\\study\\projects\\obj_yolo\\yolo_config\\yolo11n.yaml",
        base_data_path="C:\\Users\\lingu\\study\\projects\\obj_yolo\\base_data\\training",
        client_data_path="C:\\Users\\lingu\\study\\projects\\obj_yolo\\prepared_data\\clients",
        num_clients=2,
        communication_rounds=2,
        epochs_per_round=1,
        min_clients_aggregation=2,
        client_data_count=1000,
    )

    logging.info("Starting the simulation with the following configuration:")
    logging.info(f"  Total clients: {sim_config.num_clients}")
    logging.info(f"  Communication rounds: {sim_config.communication_rounds}")
    logging.info(f"  Epochs per round: {sim_config.epochs_per_round}")
    logging.info(f"  Clients per round: {sim_config.min_clients_aggregation}")
    
    setup_and_run(config=sim_config)
    