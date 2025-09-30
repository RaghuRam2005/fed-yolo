# simulation.py
import os
import logging
import random
import yaml
from pathlib import Path
from typing import Dict, List

from ultralytics import YOLO

from obj_yolo import (
    Client,
    Server,
    Strategy,
    kitti_client_data,
    build_dicts,
    bdd_client_data
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_model_config(template_path: Path, num_classes: int) -> Path:
    """
    Creates a temporary model YAML config file with a specific number of classes.
    """
    with open(template_path, 'r') as f:
        config = yaml.safe_load(f)

    config['nc'] = num_classes
    
    output_path = template_path.parent / f"{template_path.stem}_{num_classes}nc.yaml"
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f)
        
    logging.info(f"Generated model config for {num_classes} classes at {output_path}")
    return output_path

# --- Experiment 1: FedWeg on KITTI with Linear Sparsity Update ---
def run_fed_weg_kitti():
    """
    Runs the FedWeg simulation on the KITTI dataset.
    """
    logging.info("==========================================================")
    logging.info("= Running Experiment 1: FedWeg on KITTI Dataset          =")
    logging.info("==========================================================")
    strategy = Strategy(min_clients_for_aggregation=3)

    base_path = Path(__file__).parent
    # Use a template file
    yolo_config_template = base_path / "yolo_config" / "yolo11n.yaml"
    # Create the correct model config for KITTI (8 classes)
    yolo_config = create_model_config(yolo_config_template, num_classes=8)
    
    img_base_path = base_path / "base_data" / "training"
    client_base_path = base_path / "prepared_data" / "kitti_clients"
    
    model = YOLO(yolo_config).load("yolo11n.pt")
    epochs = 1
    num_clients = 3
    communication_rounds = 5
    client_data_count = 1000
    
    try:
        image_list = os.listdir(Path(img_base_path) / "image_2")
    except FileNotFoundError:
        logging.error(f"KITTI image directory not found at {img_base_path / 'image_2'}")
        return

    random.shuffle(image_list)

    server = Server(
        communication_rounds=communication_rounds,
        model=model,
        strategy=strategy,
        num_nodes=num_clients
    )
    clients = server.create_clients(tag_list=[])

    completed_images = 0
    data_paths = {}

    for i in range(server.communication_rounds):
        logging.info(f"------------------ Communication Round {i+1}/{server.communication_rounds} --------------------")
        configure_fit = server.strategy.configure_fit(epochs=epochs)
        results = []
        
        logging.info("-------------- Client Training Started ------------")
        for client in clients:
            logging.info(f"------ Training Client {client.client_id} (Sparsity: {client.sparsity:.2f}) ------")
            client.model = YOLO(yolo_config) # Use the dynamically generated config
            # ... rest of the loop is the same
            client.update_model(parameters=server.global_state)
            
            start_idx = completed_images
            end_idx = start_idx + client_data_count
            train_list = image_list[start_idx:end_idx]
            val_list = image_list[end_idx:end_idx+100]
            completed_images = end_idx + 100

            data_path = kitti_client_data(
                base_path=img_base_path,
                client_base_path=client_base_path,
                client_id=client.client_id,
                train_images=train_list,
                val_images=val_list)
            data_paths[client.client_id] = data_path
            
            result = client.fit(client_config=configure_fit, data_path=data_path)
            results.append(result)
            client.update_sparsity_linear()

        # ... rest of the function is the same
        logging.info("------------- Aggregation Started --------------------")
        aggregate_state = server.strategy.aggregate_fit(global_state=server.global_state, results=results)
        logging.info("------------- Aggregation Completed ------------------")
        
        server.global_model.model.model.load_state_dict(aggregate_state, strict=False)
        server.global_state = server.global_model.model.model.state_dict()
        logging.info("Global model state updated.")

        logging.info("------------------ Client Evaluation Started ---------------")
        for client in clients:
            client.update_model(parameters=server.global_state)
            metrics = client.evaluate(data_paths[client.client_id])
            logging.info(f"------ Evaluation Client {client.client_id} | mAP50-95: {metrics.box.map:.4f} ------")

    logging.info("========== FEDERATED LEARNING (KITTI) COMPLETE ==========\n")


# --- Experiments 2 & 3: FedWeg on BDD100K with Performance-based Sparsity Update ---
def run_fed_weg_bdd(partition_attribute: str):
    """
    Runs the FedWeg simulation on the BDD100K dataset.
    """
    logging.info("===============================================================")
    logging.info(f"= Running Experiment: FedWeg on BDD100K (Partition: {partition_attribute.upper()}) =")
    logging.info("===============================================================")
    
    strategy = Strategy(min_clients_for_aggregation=3)
    
    base_path = Path(__file__).parent
    # Use the same template file
    yolo_config_template = base_path / "yolo_config" / "yolo11n.yaml"
    # Create the correct model config for BDD (10 classes)
    yolo_config = create_model_config(yolo_config_template, num_classes=10)

    base_data_path = base_path / "bdd100k_kaggle"
    client_base_path = base_path / "prepared_data" / "bdd_clients"
    
    model = YOLO(yolo_config).load("yolo11n.pt")
    epochs = 3
    num_clients = 3
    communication_rounds = 5
    client_data_count = 1500

    label_path = base_data_path / "labels" / "bdd100k_labels_images_train.json"
    if not label_path.exists():
        logging.error(f"BDD100K label file not found at {label_path}")
        return
        
    weather_dict, scene_dict = build_dicts(label_file=str(label_path))
    
    data_dict = weather_dict if partition_attribute == 'weather' else scene_dict
    valid_tags = [tag for tag, images in data_dict.items() if len(images) > (client_data_count + 200)]
    if len(valid_tags) < num_clients:
        logging.error(f"Not enough data categories in '{partition_attribute}' for {num_clients} clients.")
        return

    server = Server(
        communication_rounds=communication_rounds,
        model=model,
        strategy=strategy,
        num_nodes=num_clients
    )
    
    clients = server.create_clients(tag_list=valid_tags)
    data_paths = {}

    for i in range(server.communication_rounds):
        logging.info(f"------------------ Communication Round {i+1}/{server.communication_rounds} --------------------")
        configure_fit = server.strategy.configure_fit(epochs=epochs)
        results = []

        logging.info("-------------- Client Training Started ------------")
        for client in clients:
            logging.info(f"------ Training Client {client.client_id} (Tag: {client.tag}, Sparsity: {client.sparsity:.2f}) ------")
            client.model = YOLO(yolo_config) # Use the dynamically generated config
            # ... rest of the loop is the same
            client.update_model(parameters=server.global_state)
            
            image_list = data_dict[client.tag]
            random.shuffle(image_list)
            train_list = image_list[:client_data_count]
            val_list = image_list[client_data_count:client_data_count+200]
            
            data_path = bdd_client_data(
                base_data_path=str(base_data_path),
                client_data_path=str(client_base_path),
                client_id=client.client_id,
                train_images=train_list,
                val_images=val_list,
            )
            data_paths[client.client_id] = data_path
            
            result = client.fit(client_config=configure_fit, data_path=str(data_path))
            results.append(result)

        # ... rest of the function is the same
        logging.info("------------- Aggregation Started --------------------")
        aggregate_state = server.strategy.aggregate_fit(global_state=server.global_state, results=results)
        logging.info("------------- Aggregation Completed ------------------")
        
        server.global_model.model.model.load_state_dict(aggregate_state, strict=False)
        server.global_state = server.global_model.model.model.state_dict()
        logging.info("Global model state updated.")
        
        logging.info("------------------ Client Evaluation & Sparsity Update Started ---------------")
        for client in clients:
            client.update_model(parameters=server.global_state)
            metrics = client.evaluate(data_paths[client.client_id])
            logging.info(f"------ Evaluation Client {client.client_id} | mAP50-95: {metrics.box.map:.4f} ------")
            client.update_sparsity_performance(current_map=metrics.box.map)

    logging.info(f"========== FEDERATED LEARNING (BDD100K - {partition_attribute.upper()}) COMPLETE ==========\n")


if __name__ == "__main__":
    # Experiment 1
    run_fed_weg_kitti()
    
    # Experiment 2
    run_fed_weg_bdd(partition_attribute='weather')
    
    # Experiment 3
    run_fed_weg_bdd(partition_attribute='scene')