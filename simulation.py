# simulation.py
import os
import logging
from pathlib import Path

from ultralytics import YOLO

from obj_yolo import (
    FedWegServer,
    FedTagServer,
    FedWeg,
    FedTag,
    BddData,
    KittiData,
    ServerConfigFedTag,
    ServerConfigFedWeg,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def warm_up_server_model(model: YOLO, data_class, image_dict=None):
    logging.info("-------------------- WARMING UP SERVER MODEL --------------------")
    if isinstance(data_class, KittiData):
        warmup_data_yaml = data_class.prepare_client_data(
            client_id=100, train_data_count=16, val_data_count=16
        )
    elif isinstance(data_class, BddData):
        _, scene_dict = data_class.create_tag_dicts()
        image_list = next(iter(image_dict.values()))
        warmup_data_yaml = data_class.prepare_client_data(
            client_id=100, train_img_list=image_list[:16], val_img_list=image_list[16:32]
        )
    else:
        raise TypeError(f"Unsupported data class for model warm-up: {type(data_class)}")

    model.train(
        data=warmup_data_yaml,
        epochs=5,
        batch=4,
        imgsz=320, 
        project="fed_yolo",
        name="server_model_init",
        exist_ok=True,
        save=False,
        plots=False,
        verbose=False
    )
    logging.info("-------------------- SERVER MODEL WARM-UP COMPLETE --------------------")
    return model

def fedweg_kitti_experiment():
    """Simulation of FedWeg with the KITTI dataset."""
    logging.info("==================== STARTING FEDWEG KITTI EXPERIMENT ====================")
    
    communication_rounds = 3
    initial_sparsity = 0.2
    super_num_nodes = 4
    min_clients = 3
    client_epochs = 5

    base_path = Path(os.path.dirname(os.path.abspath(__file__)))
    yolo_config = base_path / "yolo_config" / "yolo11n.yaml"
    base_data_path = base_path / "kitti"
    prep_data_path = base_path / "prepared_data" / "kitti_clients"

    train_data_count = 500
    val_data_count = 100

    model = YOLO(str(yolo_config)).load("yolo11n.pt")
    data_class = KittiData(base_data_path=str(base_data_path), prep_data_path=str(prep_data_path), exist_ok=True)

    model = warm_up_server_model(model, KittiData)

    strategy = FedWeg(
        data_class=data_class,
        initial_sparsity=initial_sparsity,
        min_clients=min_clients,
        client_epochs=client_epochs
    )

    server_config = ServerConfigFedWeg(
        client_model_path=str(yolo_config),
        client_train_data_count=train_data_count,
        client_val_data_count=val_data_count,
        communication_rounds=communication_rounds,
        num_nodes=super_num_nodes,
    )

    server = FedWegServer(model=model, strategy=strategy, config=server_config)

    logging.info("-------------------- FEDWEG (KITTI) Server Started --------------------")
    logging.info("-------------------- CREATING CLIENTS -------------------------")
    server.start_clients()

    for r in range(server.config.communication_rounds):
        logging.info(f"--------------------- COMMUNICATION ROUND {r + 1}/{communication_rounds} -------------------")
        results = server.fit_clients()
        logging.info(f"Round {r + 1}: Client training complete.")
        for i, res in enumerate(results):
            logging.info(f"  Client {i} metrics: mAP50-95={res.metrics.box.map:.4f}, Sparsity={res.sparsity:.2f}")

        server.start_aggregation()
        logging.info(f"Round {r + 1}: Aggregation complete.")

        eval_results = server.start_evaluation()
        logging.info(f"Round {r + 1}: Evaluation complete.")
        for i, res in enumerate(eval_results):
            logging.info(f"  Client {i} (with global model): mAP50-95={res.box.map:.4f}")

        server.update_sparsity()
        logging.info(f"Round {r + 1}: Client sparsities updated for next round.")

    logging.info("------------------ FINAL EVALUATION -----------------")
    final_eval_results = server.start_evaluation()
    for i, res in enumerate(final_eval_results):
        logging.info(f"  Final performance of Client {i}: mAP50-95={res.box.map:.4f}")
    logging.info("==================== FEDWEG KITTI EXPERIMENT COMPLETE ====================")

def fedweg_bdd100k_experiment():
    """Simulation of FedWeg with the BDD100k dataset."""
    logging.info("==================== STARTING FEDWEG BDD100K EXPERIMENT ====================")
    
    communication_rounds = 3
    initial_sparsity = 0.2
    super_num_nodes = 4
    min_clients = 3
    client_epochs = 5

    base_path = Path(os.path.dirname(os.path.abspath(__file__)))
    yolo_config = base_path / "yolo_config" / "yolo11n.yaml"
    base_data_path = base_path / "bdd100k"
    prep_data_path = base_path / "prepared_data" / "bdd_clients_fedweg"

    train_data_count = 1000
    val_data_count = 200

    model = YOLO(str(yolo_config)).load("yolo11n.pt")
    logging.warning("FedWeg with BDD100k assumes data is in YOLO format (like KITTI's 'image_2', 'labels' folders).")
    data_class = KittiData(base_data_path=str(base_data_path), prep_data_path=str(prep_data_path), exist_ok=True)
    
    model = warm_up_server_model(model, KittiData)

    strategy = FedWeg(data_class=data_class, initial_sparsity=initial_sparsity, min_clients=min_clients, client_epochs=client_epochs)
    server_config = ServerConfigFedWeg(
        client_model_path=str(yolo_config),
        client_train_data_count=train_data_count,
        client_val_data_count=val_data_count,
        communication_rounds=communication_rounds,
        num_nodes=super_num_nodes,
    )
    server = FedWegServer(model=model, strategy=strategy, config=server_config)

    logging.info("-------------------- FEDWEG (BDD100k) Server Started --------------------")
    server.start_clients()

    for r in range(server.config.communication_rounds):
        logging.info(f"--------------------- COMMUNICATION ROUND {r + 1}/{communication_rounds} -------------------")
        server.fit_clients()
        logging.info(f"Round {r + 1}: Client training complete.")
        server.start_aggregation()
        logging.info(f"Round {r + 1}: Aggregation complete.")
        server.start_evaluation()
        logging.info(f"Round {r + 1}: Evaluation complete.")
        server.update_sparsity()
        logging.info(f"Round {r + 1}: Client sparsities updated.")

    logging.info("------------------ FINAL RESULTS -----------------")
    final_eval_results = server.start_evaluation()
    for i, res in enumerate(final_eval_results):
        logging.info(f"  Final performance of Client {i}: mAP50-95={res.box.map:.4f}")
    logging.info("==================== FEDWEG BDD100K EXPERIMENT COMPLETE ====================")

def fedtag_scene_experiment():
    """Simulation of FedTag with the BDD100k dataset, tagged by scene."""
    logging.info("==================== STARTING FEDTAG BDD100K (SCENE) EXPERIMENT ====================")
    
    communication_rounds = 3
    initial_sparsity = 0.2
    super_num_nodes = 5
    min_clients = 4
    client_epochs = 10

    base_path = Path(os.path.dirname(os.path.abspath(__file__)))
    yolo_config = base_path / "yolo_config" / "yolo11n.yaml"
    base_data_path = base_path / "bdd100k"
    prep_data_path = base_path / "prepared_data" / "bdd_clients_fedtag_scene"

    train_data_count = 1000
    val_data_count = 200

    model = YOLO(str(yolo_config)).load("yolo11n.pt")
    data_class = BddData(base_data_path=str(base_data_path), prep_data_path=str(prep_data_path), exist_ok=True)
    _, scene_dict = data_class.create_tag_dicts()

    keys_to_remove = [k for k, v in scene_dict.items() if len(v) <= train_data_count + val_data_count]
    for key in keys_to_remove:
        scene_dict.pop(key)
    logging.info(f"Usable scenes for training: {list(scene_dict.keys())}")

    model = warm_up_server_model(model=model, data_class=data_class, image_dict=scene_dict)

    strategy = FedTag(data_class=data_class, initial_sparsity=initial_sparsity, min_clients=min_clients, client_epochs=client_epochs)
    server_config = ServerConfigFedTag(
        client_model_path=str(yolo_config),
        client_tag_dict=scene_dict,
        client_train_data_count=train_data_count,
        client_val_data_count=val_data_count,
        communication_rounds=communication_rounds,
        num_nodes=super_num_nodes,
    )
    server = FedTagServer(model=model, strategy=strategy, config=server_config)

    logging.info("-------------------- FEDTAG (BDD100k, Scene) Server Started --------------------")
    server.start_clients()

    for r in range(server.config.communication_rounds):
        logging.info(f"--------------------- COMMUNICATION ROUND {r + 1}/{communication_rounds} -------------------")
        server.fit_clients()
        logging.info(f"Round {r + 1}: Client training complete.")
        server.start_aggregation()
        logging.info(f"Round {r + 1}: Aggregation complete.")
        tag_based_results = server.start_evaluation()
        logging.info(f"Round {r + 1}: Tag-based evaluation complete. mAP scores: {tag_based_results}")
        server.update_sparsity(tag_based_results)
        logging.info(f"Round {r + 1}: Client sparsities updated.")

    logging.info("------------------ FINAL RESULTS -----------------")
    final_results = server.start_evaluation()
    logging.info(f"Final tag-based evaluation results: {final_results}")
    logging.info("==================== FEDTAG BDD100K (SCENE) EXPERIMENT COMPLETE ====================")

def fedtag_weather_experiment():
    """Simulation of FedTag with the BDD100k dataset, tagged by weather."""
    logging.info("==================== STARTING FEDTAG BDD100K (WEATHER) EXPERIMENT ====================")
    
    communication_rounds = 2
    initial_sparsity = 0.2
    super_num_nodes = 2
    min_clients = 2
    client_epochs = 3

    base_path = Path(os.path.dirname(os.path.abspath(__file__)))
    yolo_config = base_path / "yolo_config" / "yolo11n.yaml"
    base_data_path = base_path / "bdd100k"
    prep_data_path = base_path / "prepared_data" / "bdd_clients_fedtag_weather"

    train_data_count = 500
    val_data_count = 100

    model = YOLO(str(yolo_config)).load("yolo11n.pt")
    data_class = BddData(base_data_path=str(base_data_path), prep_data_path=str(prep_data_path), exist_ok=True)
    weather_dict, _ = data_class.create_tag_dicts()

    keys_to_remove = [k for k, v in weather_dict.items() if len(v) <= train_data_count + val_data_count]
    for key in keys_to_remove:
        weather_dict.pop(key)
    if not weather_dict:
        return
    logging.info(f"Usable weather conditions for training: {list(weather_dict.keys())}")

    model = warm_up_server_model(model, data_class, weather_dict)

    strategy = FedTag(data_class=data_class, initial_sparsity=initial_sparsity, min_clients=min_clients, client_epochs=client_epochs)
    server_config = ServerConfigFedTag(
        client_model_path=str(yolo_config),
        client_tag_dict=weather_dict,
        client_train_data_count=train_data_count,
        client_val_data_count=val_data_count,
        communication_rounds=communication_rounds,
        num_nodes=super_num_nodes,
    )
    server = FedTagServer(model=model, strategy=strategy, config=server_config)

    logging.info("-------------------- FEDTAG (BDD100k, Weather) Server Started --------------------")
    server.start_clients()

    for r in range(server.config.communication_rounds):
        logging.info(f"--------------------- COMMUNICATION ROUND {r + 1}/{communication_rounds} -------------------")
        server.fit_clients()
        logging.info(f"Round {r + 1}: Client training complete.")
        server.start_aggregation()
        logging.info(f"Round {r + 1}: Aggregation complete.")
        tag_based_results = server.start_evaluation()
        logging.info(f"Round {r + 1}: Tag-based evaluation complete. mAP scores: {tag_based_results}")
        server.update_sparsity(tag_based_results)
        logging.info(f"Round {r + 1}: Client sparsities updated based on performance.")
        current_sparsities = {f"client_{c.client_id} ({c.tag})": c.sparsity for c in server.clients}
        logging.info(f"  Current client sparsities: {current_sparsities}")

    logging.info("------------------ FINAL RESULTS -----------------")
    final_results = server.start_evaluation()
    logging.info(f"Final tag-based evaluation results: {final_results}")
    logging.info("==================== FEDTAG BDD100K (WEATHER) EXPERIMENT COMPLETE ====================")

if __name__ == "__main__":
    fedtag_weather_experiment()
    # fedtag_bdd100k_scene_experiment()
    # fedweg_kitti_experiment()
    # fedweg_bdd100k_experiment()
