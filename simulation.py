# simulation.py
import os
import logging
import random
from pathlib import Path

from ultralytics import YOLO

from obj_yolo import (
    Client,
    Server,
    Strategy,
    load_data
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":

    strategy = Strategy(
        min_clients_for_aggregation=2
    )

    base_path = os.path.dirname(os.path.abspath(__file__))
    yolo_config = Path(base_path) / "yolo_config" / "yolo11n.yaml"
    img_base_path = Path(base_path) / "base_data" / "training"
    client_base_path = Path(base_path) / "prepared_data" / "clients"
    model = YOLO(yolo_config)
    epochs = 1
    image_list = os.listdir(Path(img_base_path) / "image_2")
    client_data_count = 1000
    random.shuffle(image_list)

    # instead of global data for simualation we take client id as 100 and generate data
    global_data_path = load_data(
        base_path=img_base_path,
        client_base_path=client_base_path,
        client_id=100,
        train_images=image_list[:100],
        val_images=image_list[100:151]
    )
    model.train(
        epochs=10,
        data=global_data_path,
        project="fed_yolo",
        name="global_train",
        exist_ok=True,
    )

    server = Server(
        communication_rounds=1,
        model=model,
        strategy=strategy,
        num_nodes=3
    )

    clients = server.create_clients()

    completed_images = 151
    for i in range(server.communication_rounds):
        logging.info(f"------------------ Communication Round 1 --------------------")
        configure_fit = server.strategy.configure_fit(epochs=epochs)
        results = []
        data_paths = []
        logging.info("-------------- client training started ------------")
        for client in clients:
            logging.info(f"------ client {client.client_id} ------")
            client.model = YOLO(yolo_config)
            client.update_model(parameters=server.global_state)
            train_list = image_list[completed_images:completed_images+client_data_count]
            completed_images += client_data_count
            val_list = image_list[completed_images:completed_images+100]
            completed_images += 1
            data_path = load_data(
                base_path=img_base_path, 
                client_base_path=client_base_path, 
                client_id=client.client_id,
                train_images=train_list,
                val_images=val_list)
            data_paths.append(data_path)
            result = client.fit(client_config=configure_fit, data_path=data_path)
            results.append(result)
        logging.info("------------- Aggregation Started -------------------- ")
        aggregate_state = server.strategy.aggregate_fit(global_state=server.global_state, results=results)
        logging.info("------------- Aggregation Completed ------------------- ")
        logging.info("updating global model state")
        server.global_model.model.model.load_state_dict(aggregate_state)
        server.global_state = server.global_model.model.model.state_dict()
        logging.info("global model updation complete")
        logging.info("------------------ CLIENT EVALUATION STARTED --------------- ")
        for client in clients:
            client.update_model(parameters=server.global_state)
            metrics = client.evaluate(data_paths[client.client_id])
            logging.info(f"---------- Evaluation Client {client.client_id} ----------- ")
            logging.info(f"mAP 50-95: {metrics.box.map}")
            logging.info(f"mAP 50   : {metrics.box.map50}")
            logging.info(f"mAP 75   : {metrics.box.map75}")
        logging.info("---------------- CLIENT EVALUATION COMPLETE ----------------- ")
    logging.info(" ----------------- FEDERATED LEARNING LOOP COMPLETE -------------------- ")