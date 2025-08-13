# server.py
import io
import os
import torch
import logging
from collections import OrderedDict
from typing import Dict, List
from flask import Flask, request, send_file, jsonify
import ultralytics
from ultralytics import YOLO
from ultralytics.engine.model import Model
from pathlib import Path

# Import configuration
from config import (
    MODEL_NAME,
    GLOBAL_DATA_PATH,
    SERVER_HOST,
    SERVER_PORT,
    CLIENTS,
    GLOBAL_EPOCHS
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_global_model(model:Model, data_path:Path, epochs:int=3) -> List[torch.nn.ParameterDict]:
    """
    Trains the YOLO model using the custom dataset

    Args:
        model (Model): Model object
        epochs (int): Number of rounds to train the data
        data_path (Path): YAML file path
    """
    print("--------------------------------------------------------")
    print("started training the global model")
    model.train(
        data=data_path,
        epochs=epochs,
        save=False,
        project="fed_yolo",
        name="server_pretrain",
        exist_ok=True,
        pretrained=True,
        device=-1,
        warmup_epochs=3,
        cos_lr=True,

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
    print("global model training complete")
    results = model.val(
        save_json=True,
        device=-1,
        project="fed_yolo",
        name="global_model",
    )
    print("----------------------------------------------------------")
    print("global model validation metrics:")
    print("mAP (50-95): ", results.box.map)
    print("mAP (50)   : ", results.box.map50)
    print("mAP (75)   : ", results.box.map75)
    return model.model.state_dict()


class FedServer:
    def __init__(self, global_parameters:torch.nn.ParameterDict, min_clients_for_aggregation: int = 1):
        self.global_parameters = global_parameters
        self.training_round = 0
        self.client_updates = []
        self.client_data_counts = []
        self.min_clients_for_aggregation = max(1, min_clients_for_aggregation)

        logging.info(f"FedServer initialized with model '{MODEL_NAME}'.")
        logging.info(f"Waiting for {self.min_clients_for_aggregation} clients to trigger aggregation.")

    def get_parameters_for_client(self) -> OrderedDict:
        return self.global_parameters

    def aggregate_fit(self, new_parameters: List[OrderedDict], client_data_counts: List[int]):
        """
        Aggregates client model updates and updates the global_parameters.
        It does NOT load the new state into a model here.
        """
        assert len(new_parameters) == len(client_data_counts), \
            "Parameter and data count lists must have the same length"

        logging.info(f"Starting aggregation for round {self.training_round + 1} with {len(new_parameters)} client updates.")
        total_data_count = sum(client_data_counts)

        if total_data_count == 0:
            logging.warning("Total data count is zero. Skipping aggregation.")
            return

        aggregated_state_dict = OrderedDict({k: torch.zeros_like(v) for k, v in new_parameters[0].items()})
        
        for i, client_state_dict in enumerate(new_parameters):
            weight = client_data_counts[i] / total_data_count
            for key in aggregated_state_dict.keys():
                if aggregated_state_dict[key].dtype.is_floating_point:
                    aggregated_state_dict[key] += client_state_dict[key] * weight
                elif i == 0:
                    aggregated_state_dict[key] = client_state_dict[key]

        self.global_parameters = aggregated_state_dict
        self.training_round += 1
        logging.info(f"Aggregation complete. Global parameters updated for round {self.training_round}.")

    def aggregate_evaluate(self, data_path: str):
        """
        Performs evaluation by creating a temporary model instance and loading
        the current global parameters into it.
        """
        logging.info(f"Evaluating aggregated model of round {self.training_round}...")

        # Create a temporary model instance for this evaluation.
        eval_model = YOLO(MODEL_NAME)
        
        # Load the global parameters. Use strict=False because the head will likely mismatch
        # the initial 80-class head of the new model instance.
        eval_model.model.load_state_dict(self.global_parameters, strict=False)
        logging.info("Loaded global parameters into a temporary evaluation model.")

        # The .val() function is smart. It will read the `nc` (number of classes)
        # from the data_path's YAML file and automatically adjust the model's head
        # to match the validation dataset before running evaluation.
        results = eval_model.val(
            data=data_path,
            save_json=True,
            device='cpu',
            project="fed_yolo",
            name=f"eval_round_{self.training_round}",
            exist_ok=True
        )

        logging.info("--- Global Model Validation Metrics ---")
        logging.info(f"mAP (50-95): {results.box.map:.4f}")
        logging.info(f"mAP (50):    {results.box.map50:.4f}")
        logging.info(f"mAP (75):    {results.box.map75:.4f}")
        logging.info("---------------------------------------")

        return {
            "round": self.training_round,
            "map": results.box.map,
            "map50": results.box.map50,
            "map75": results.box.map75
        }


app = Flask(__name__)
model = YOLO(MODEL_NAME)
data_path = Path(GLOBAL_DATA_PATH) / "data.yaml"
pretrained_model = YOLO("yolov8n.pt")
pretrained_state = pretrained_model.model.state_dict()
custom_state = model.model.state_dict()
filtered_state = {k: v for k, v in pretrained_state.items()
                  if k in custom_state and v.shape == custom_state[k].shape}
model.model.load_state_dict(filtered_state, strict=False)
model.model.model[-1] = ultralytics.nn.modules.head.Detect(nc=7, ch=[64, 128, 256])
global_parameters = train_global_model(model=model, data_path=data_path, epochs=GLOBAL_EPOCHS)
fed_server = FedServer(min_clients_for_aggregation=CLIENTS, global_parameters=global_parameters)

@app.route("/get_parameters", methods=["GET"])
def get_parameters():
    parameters = fed_server.get_parameters_for_client()
    buffer = io.BytesIO()
    torch.save(parameters, buffer)
    buffer.seek(0)
    return send_file(buffer, mimetype='application/octet-stream', as_attachment=True, download_name='global_model_params.pth')

@app.route("/submit_update", methods=["POST"])
def submit_update():
    if 'model_file' not in request.files or 'data_count' not in request.form:
        return jsonify({"error": "Missing model file or data count"}), 400

    model_file = request.files['model_file']
    data_count = int(request.form['data_count'])
    client_state_dict = torch.load(io.BytesIO(model_file.read()))

    fed_server.client_updates.append(client_state_dict)
    fed_server.client_data_counts.append(data_count)

    logging.info(f"Received update. Total updates: {len(fed_server.client_updates)}/{fed_server.min_clients_for_aggregation}")

    if len(fed_server.client_updates) >= fed_server.min_clients_for_aggregation:
        # Step 1: Aggregate parameters
        fed_server.aggregate_fit(fed_server.client_updates, fed_server.client_data_counts)
        # Step 2: Evaluate the new global parameters
        eval_path = Path(GLOBAL_DATA_PATH) / "data.yaml"
        metrics = fed_server.aggregate_evaluate(data_path=eval_path)
        
        fed_server.client_updates.clear()
        fed_server.client_data_counts.clear()
        logging.info("Round complete. Ready for next round.")
        return jsonify({"status": "Aggregation and evaluation complete", "metrics": metrics})

    return jsonify({"status": f"Update received. Waiting for {fed_server.min_clients_for_aggregation - len(fed_server.client_updates)} more clients."})

if __name__ == "__main__":
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=False)