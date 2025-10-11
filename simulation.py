# simulation.py
import os
import logging
import random
from pathlib import Path

from ultralytics import YOLO

from obj_yolo import (
    FedWegServer,
    FedTagServer,
    FedWeg,
    FedTag,
    BddData,
    ServerConfigFedTag,
    ServerConfigFedWeg
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fedtag_weather_experiment():
    communication_rounds = 2
    initial_sparsity = 0.2
    super_num_nodes = 3
    min_clients = 2

    base_path = os.path.dirname(os.path.abspath(__file__))
    yolo_config = Path(base_path) / "yolo_config" / "yolo11n.yaml"
    base_data_path = Path(base_path) / "base_data" / "training"
    prep_data_path = Path(base_path) / "prepared_data" / "clients"

    train_data_count = 1000
    val_data_count = 200

    model = YOLO(yolo_config).load("yolo11n.pt")
    
    data_class = BddData(base_data_path=base_data_path, prep_data_path=prep_data_path, exist_ok=False)
    weather_dict, scene_dict = data_class.create_tag_dicts()

    for key, values in weather_dict:
        if not len(values) > train_data_count + val_data_count:
            weather_dict.pop(key)
    
    strategy = FedTag(
        data_class=data_class,
        initial_sparsity=initial_sparsity,
        min_clients=min_clients
    )

    server_config = ServerConfigFedTag(
        client_model_path=yolo_config,
        client_tag_dict=weather_dict,
        client_train_data_count=train_data_count,
        client_val_data_count=val_data_count,
        communication_rounds=communication_rounds,
        num_nodes=super_num_nodes,
    )

    server = FedTagServer(
        model=model,
        strategy=strategy,
        config=server_config,
    )

    logging.info("-------------------- FEDTAG Server Started -------------------- ")
    logging.info("-------------------- CREATING CLIENTS ------------------------- ")

    server.start_clients()

    logging.info("-------------------- START CLIENT TRAINING -------------------- ")

    for x in range(server.config.communication_rounds):
        logging.info(f"--------------------- COMMUNICATION ROUND {x} ------------------- ")
        results = server.fit_clients()
        logging.info("--------------------- CLIENT TRAINING COMPLETE ------------------ ")
        logging.info("--------------------- STARTED AGGREGATION ------------------------ ")
        agg_state = server.start_aggregation()
        logging.info("--------------------- AGGREGATION COMPLETE ------------------------- ")
        logging.info("---------------------- STARTED EVALUATION --------------------------")
        tag_based_results = server.start_evaluation()
        logging.info("----------------------- EVALUATION COMPLETE -------------------------")
        logging.info("----------------------- SPARSITY UPDATION ---------------------------")
        server.update_sparsity(tag_based_results)
    
    tag_based_results = server.start_evaluation()
    print("------------------ FINAL RESULTS ----------------- ")
    print(tag_based_results)
