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

def warm_up_server_model(model: YOLO, data_class, base_data_path, prep_data_path):
    """
    Performs a brief training loop on the server model to ensure all layers,
    including biases, are initialized before federated learning begins.
    This resolves potential state_dict key mismatches.
    """
    logging.info("-------------------- WARMING UP SERVER MODEL --------------------")
    # Use a unique directory for server warm-up data to avoid conflicts
    warmup_prep_path = prep_data_path / "server_warmup"
    server_data_provider = data_class(
        base_data_path=str(base_data_path),
        prep_data_path=str(warmup_prep_path),
        exist_ok=True
    )

    # Prepare a small dataset for the server warm-up
    if isinstance(server_data_provider, KittiData):
        warmup_data_yaml = server_data_provider.prepare_client_data(
            client_id=100, train_data_count=16, val_data_count=16
        )
    elif isinstance(server_data_provider, BddData):
        # BddData needs a list of images, so we generate a small one.
        _, scene_dict = server_data_provider.create_tag_dicts()
        image_list = next(iter(scene_dict.values()))
        warmup_data_yaml = server_data_provider.prepare_client_data(
            client_id=100, train_img_list=image_list[:16], val_img_list=image_list[16:32]
        )
    else:
        raise TypeError(f"Unsupported data class for model warm-up: {type(data_class)}")

    # Train for one epoch to initialize all model parameters
    model.train(
        data=warmup_data_yaml,
        epochs=5,
        batch=4,
        imgsz=320,  # Smaller image size for faster warm-up
        project="fed_yolo",
        name="server_model_init",
        exist_ok=True,
        save=False,
        plots=False,
        verbose=False  # Keep the log clean from training details
    )
    logging.info("-------------------- SERVER MODEL WARM-UP COMPLETE --------------------")
    return model

def fedweg_kitti_experiment():
    """Simulation of FedWeg with the KITTI dataset."""
    logging.info("==================== STARTING FEDWEG KITTI EXPERIMENT ====================")
    # --- Configuration ---
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

    # --- Initialization ---
    model = YOLO(str(yolo_config)).load("yolo11n.pt")
    data_class = KittiData(base_data_path=str(base_data_path), prep_data_path=str(prep_data_path), exist_ok=True)

    # --- Server Model Warm-up (Fix for bias layers) ---
    model = warm_up_server_model(model, KittiData, base_data_path, prep_data_path)

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

    # --- Federated Learning Loop ---
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

    # --- Final Evaluation ---
    logging.info("------------------ FINAL EVALUATION -----------------")
    final_eval_results = server.start_evaluation()
    for i, res in enumerate(final_eval_results):
        logging.info(f"  Final performance of Client {i}: mAP50-95={res.box.map:.4f}")
    logging.info("==================== FEDWEG KITTI EXPERIMENT COMPLETE ====================")

def fedweg_bdd100k_experiment():
    """Simulation of FedWeg with the BDD100k dataset."""
    logging.info("==================== STARTING FEDWEG BDD100K EXPERIMENT ====================")
    # --- Configuration ---
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

    # --- Initialization ---
    model = YOLO(str(yolo_config)).load("yolo11n.pt")
    logging.warning("FedWeg with BDD100k assumes data is in YOLO format (like KITTI's 'image_2', 'labels' folders).")
    data_class = KittiData(base_data_path=str(base_data_path), prep_data_path=str(prep_data_path), exist_ok=True)
    
    # --- Server Model Warm-up ---
    model = warm_up_server_model(model, KittiData, base_data_path, prep_data_path)

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

    # --- Federated Learning Loop ---
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

def fedtag_bdd100k_scene_experiment():
    """Simulation of FedTag with the BDD100k dataset, tagged by scene."""
    logging.info("==================== STARTING FEDTAG BDD100K (SCENE) EXPERIMENT ====================")
    # --- Configuration ---
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

    # --- Initialization ---
    model = YOLO(str(yolo_config)).load("yolo11n.pt")
    data_class = BddData(base_data_path=str(base_data_path), prep_data_path=str(prep_data_path), exist_ok=True)
    _, scene_dict = data_class.create_tag_dicts()

    keys_to_remove = [k for k, v in scene_dict.items() if len(v) <= train_data_count + val_data_count]
    for key in keys_to_remove:
        scene_dict.pop(key)
    logging.info(f"Usable scenes for training: {list(scene_dict.keys())}")

    # --- Server Model Warm-up ---
    model = warm_up_server_model(model, BddData, base_data_path, prep_data_path)

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

    # --- Federated Learning Loop ---
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
    # --- Configuration ---
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

    # --- Initialization ---
    model = YOLO(str(yolo_config)).load("yolo11n.pt")
    data_class = BddData(base_data_path=str(base_data_path), prep_data_path=str(prep_data_path), exist_ok=True)
    weather_dict, _ = data_class.create_tag_dicts()

    keys_to_remove = [k for k, v in weather_dict.items() if len(v) <= train_data_count + val_data_count]
    for key in keys_to_remove:
        weather_dict.pop(key)
    logging.info(f"Usable weather conditions for training: {list(weather_dict.keys())}")

    # --- Server Model Warm-up ---
    #model = warm_up_server_model(model, BddData, base_data_path, prep_data_path)

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

    # --- Federated Learning Loop ---
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
    # You can uncomment the experiment you wish to run.
    fedtag_weather_experiment()
    # fedtag_bdd100k_scene_experiment()
    # fedweg_kitti_experiment()
    # fedweg_bdd100k_experiment()
