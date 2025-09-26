# simulation.py
import os
import logging
from pathlib import Path

from ultralytics import YOLO

from obj_yolo import (
    Client,
    Server,
    Strategy,
    create_yolo_dataset
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_experiment(experiment_type: str):
    """
    Runs a complete federated learning experiment for a given type ('weather' or 'scene').
    """
    logging.info(f"##################################################")
    logging.info(f"##### STARTING EXPERIMENT: {experiment_type.upper()} #####")
    logging.info(f"##################################################")

    strategy = Strategy(min_clients_for_aggregation=2)

    # --- Paths (Update these to your BDD100K dataset location) ---
    base_path = os.path.dirname(os.path.abspath(__file__))
    yolo_config = Path(base_path) / "yolo_config" / "yolo11n.yaml"
    model = YOLO(yolo_config)
    
    # --- Experiment Parameters ---
    communication_rounds = 3
    epochs_per_round = 1
    client_data_count = 500  # Number of images per client
    num_clients = 3

    server = Server(
        communication_rounds=communication_rounds,
        model=model,
        strategy=strategy,
        num_nodes=num_clients,
        experiment_type=experiment_type
    )

    clients = server.create_clients()
    logging.info(f"Created {len(clients)} clients with tags: {[c.tag for c in clients]}")

    for i in range(server.communication_rounds):
        logging.info(f"------------------ Communication Round {i + 1}/{server.communication_rounds} --------------------")
        
        configure_fit = server.strategy.configure_fit(epochs=epochs_per_round)
        fit_results = []
        data_paths = {}
        logging.info("-------------- Client Data Preparation & Training Started ------------")
        for client in clients:
            logging.info(f"------ Client {client.client_id} (Tag: {client.tag}) ------")
            
            data_path_dict = create_yolo_dataset(
                base_path=base_path,
                client_id=str(client.client_id),
                filter_key=experiment_type,
                filter_value=client.tag,
                data_count=client_data_count,
            )
            data_yaml_path = data_path_dict["data_yaml_path"]
            data_paths[client.client_id] = data_yaml_path
            
            client.model = YOLO(yolo_config)
            client.update_model(parameters=server.global_state)
            result = client.fit(client_config=configure_fit, data_path=data_yaml_path)
            fit_results.append(result)

        logging.info("------------- Aggregation Started -------------------- ")
        aggregate_state = server.strategy.aggregate_fit(global_state=server.global_state, results=fit_results)
        server.global_model.model.model.load_state_dict(aggregate_state, strict=False)
        server.global_state = server.global_model.model.model.state_dict()
        logging.info("------------- Aggregation & Global Model Update Completed ------------------- ")

        logging.info("------------------ CLIENT EVALUATION STARTED --------------- ")
        evaluation_results = {}
        for client in clients:
            client.update_model(parameters=server.global_state)
            metrics = client.evaluate(data_paths[client.client_id])
            evaluation_results[client.client_id] = metrics
            logging.info(f"--- Evaluation Client {client.client_id} (Tag: {client.tag}) --- mAP 50-95: {metrics.box.map:.4f} ---")
        
        logging.info("------------------ SPARSITY UPDATE STARTED ------------------")
        client_ranks = server.strategy.position_clients(results=evaluation_results)
        logging.info(f"Client Ranks (Higher mAP = Lower Rank): {client_ranks}")
        for client in clients:
            rank = client_ranks.get(client.client_id)
            if rank is not None:
                client.update_sparsity(rank=rank, num_clients=server.num_nodes)
        logging.info("------------------ SPARSITY UPDATE COMPLETE -------------------")

    logging.info(f" ----------------- FEDERATED LEARNING FOR '{experiment_type.upper()}' COMPLETE -------------------- ")

if __name__ == "__main__":
    # Run both experiments sequentially
    run_experiment(experiment_type="weather")
    run_experiment(experiment_type="scene")