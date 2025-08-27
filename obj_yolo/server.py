# server.py
import io
import os
import torch
import logging
from collections import OrderedDict
from typing import List
from flask import Flask, request, send_file, jsonify
from ultralytics import YOLO
from ultralytics.engine.model import Model, Results
from pathlib import Path

# local imports
from strategy import FedAvg, FedWeg

# import configuration
from config import (
    MODEL_PATH,

    CLIENTS_COUNT,

    GLOBAL_DATA_PATH,
    GLOBAL_EPOCHS,

    SERVER_HOST,
    SERVER_PORT,
    STRATEGY
)
MODEL_PATH = os.getenv("MODEL_PATH", "yolo11n_baseline.yaml")
SERVER_HOST = os.getenv("SERVER_HOST", "localhost")
SERVER_PORT = int(os.getenv("SERVER_PORT", 5000))
CLIENTS = int(os.getenv("CLIENTS", 5))
GLOBAL_EPOCHS = int(os.getenv("GLOBAL_EPOCHS", 5))
GLOBAL_DATA_PATH = os.getenv("GLOBAL_DATA_PATH", ".\\yolo_datasets\\global_dataset")
STRATEGY = os.getenv("STRATEGY", "fedavg")

# check if global data path exists
if not Path(GLOBAL_DATA_PATH).exists():
    raise Exception(f"Global data path {GLOBAL_DATA_PATH} does not exist. Please check your configuration.")

is_available = torch.cuda.is_available()
if is_available:
    print(f"CUDA is available. Using GPU for training.")
else:
    print(f"CUDA is not available. Using CPU for training.")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_global_model(model: Model, data_path: str, epochs: int = 3) -> List[torch.nn.ParameterDict]:
    """
    Trains the YOLO model using the custom dataset
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
        copy_paste=0.0,
        workers=0
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

    return model

class ServerApp:
    def __init__(self, model:Model, strategy:str, min_clients:int):
        self.model = model
        self.strategy = strategy
        self.min_clients_agg = min_clients
        self.client_updates = []
        self.client_data_counts = []
        self.client_masks = [] # only used in case of FedWeg algorithm
    
    def get_parameters_for_clients(self) -> OrderedDict:
        return self.model.state_dict()

    def aggregate_fit(self, client_updates, client_data_counts) -> None:
        if self.strategy == "fedavg":
            aggregated = FedAvg.aggregate_fit(client_updates, client_data_counts)
            self.model.load_state_dict(aggregated, strict=False)
        elif self.strategy == "fedweg":
            sparse_updates, masks_list = zip(*client_updates)
            aggregated = FedWeg.aggregate_sparse_updates(
                list(sparse_updates), list(masks_list), client_data_counts, self.model.state_dict()
            )
            self.model.load_state_dict(aggregated, strict=False)
        else:
            raise ValueError(f"Unknown strategy {self.strategy}")
    
    def aggregate_evaluate(self, data_path:Path) -> Results:
        results = self.model.val(data=str(data_path), device=-1)
        logging.info("Global Model Validation Complete")
        logging.info("Global Evalution metrics: ")
        logging.info(f"map  : {results.box.map}")
        logging.info(f"map50: {results.box.map50}")
        return results

app = Flask(__name__)

@app.route("/get_parameters", methods=["GET"])
def get_parameters():
    parameters = ServerApp.get_parameters_for_clients()
    buffer = io.BytesIO()
    torch.save(parameters, buffer)
    buffer.seek(0)
    return send_file(buffer, mimetype='application/octet-stream', as_attachment=True, download_name='global_model_parameters')

@app.route("/submit_update", methods=["POST"])
def submit_update():
    if 'model_file' not in request.files or 'data_count' not in request.form:
        return jsonify({'error': 'missing model file or data count'}), 400
    
    model_file = request.files['model_file']
    data_count = int(request.form['data_count'])
    client_update = torch.load(io.BytesIO(model_file.read()))

    if ServerApp.strategy == "fedavg":
        ServerApp.client_updates.append(client_update)
    else:  # fedweg returns tuple (sparse_update, masks)
        sparse_update, masks = client_update
        ServerApp.client_updates.append((sparse_update, masks))
    ServerApp.client_data_counts.append(data_count)

    logging.info(f"Received update. Total updates: {len(ServerApp.client_updates)}/{ServerApp.min_clients_agg}")

    if len(ServerApp.client_updates) >= ServerApp.min_clients_agg:
        ServerApp.aggregate_fit(ServerApp.client_updates, ServerApp.client_data_counts)
        eval_path = Path(GLOBAL_DATA_PATH) / "data.yaml"
        metrics = ServerApp.aggregate_evaluate(eval_path)
        ServerApp.client_updates.clear()
        ServerApp.client_data_counts.clear()
        return jsonify({"status": "Aggregation and evaluation complete", "metrics": metrics})

    return jsonify({"status": f"Update received. Waiting for more clients"})

if __name__ == "__main__":
    model = YOLO(MODEL_PATH)
    data_path = str(Path(GLOBAL_DATA_PATH) / "data.yaml")

    # pretrain global model
    logging.info("pretraining global model .....")
    model = train_global_model(model, data_path, GLOBAL_EPOCHS)
    logging.info("pretraining complete")

    # start the server
    ServerApp = ServerApp(model=model, strategy=STRATEGY, min_clients=CLIENTS_COUNT)
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=True)
