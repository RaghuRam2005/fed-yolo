# server.py
import io
import os
import torch
import logging
from collections import OrderedDict
from typing import List, Dict
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

def clone_state_dict(state_dict):
    """
    Create a deep clone of state dict with detached, contiguous tensors
    """
    cloned_dict = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            cloned_dict[k] = v.detach().clone().contiguous()
        else:
            cloned_dict[k] = v
    return cloned_dict

class ServerApp:
    def __init__(self, model:Model, strategy:str, min_clients:int):
        self.model = model
        self.strategy = strategy
        self.min_clients_agg = min_clients
        self.client_updates = []
        self.client_data_counts = []
        self.client_masks = [] # only used in case of FedWeg algorithm
    
    def get_parameters_for_clients(self) -> OrderedDict:
        return clone_state_dict(self.model.state_dict())

    def aggregate_fit(self, client_updates, client_data_counts) -> None:
        # Ensure model is in training mode for parameter updates
        self.model.train()
        if self.strategy == "fedavg":
            aggregated = FedAvg.aggregate_fit(client_updates, client_data_counts)
            cloned_aggregated = clone_state_dict(aggregated)
            self.model.load_state_dict(cloned_aggregated, strict=False)
        elif self.strategy == "fedweg":
            sparse_updates, masks_list = zip(*client_updates)
            aggregated = FedWeg.aggregate_sparse_updates(
                list(sparse_updates), list(masks_list), client_data_counts, self.model.state_dict()
            )
            cloned_aggregated = clone_state_dict(aggregated)
            self.model.load_state_dict(cloned_aggregated, strict=False)
        else:
            raise ValueError(f"Unknown strategy {self.strategy}")
        
        # Set model to eval mode after aggregation
        self.model.eval()
        # Disable gradients for evaluation
        for param in self.model.parameters():
            param.requires_grad = False
    
    def aggregate_evaluate(self, data_path: Path) -> Dict:
        """
        Evaluate the aggregated model WITHOUT training
        """
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # Disable gradients completely during evaluation
        with torch.no_grad():
            # Use validation with minimal configuration to avoid training
            results = self.model.val(
                data=str(data_path), 
                device=-1,
                save=False,           # Don't save results
                verbose=False,        # Reduce verbose output
                project="fed_yolo",
                name="federation_eval",
                exist_ok=True
            )
        
        logging.info("Global Model Validation Complete")
        logging.info("Global Evaluation metrics: ")
        logging.info(f"map  : {results.box.map}")
        logging.info(f"map50: {results.box.map50}")

        metrics_dict = {
            "map": float(results.box.map) if results.box.map is not None else 0.0,
            "map50": float(results.box.map50) if results.box.map50 is not None else 0.0,
            "map75": float(results.box.map75) if results.box.map75 is not None else 0.0,
            "fitness": float(results.fitness) if hasattr(results, 'fitness') and results.fitness is not None else 0.0
        }
        
        # Re-enable gradients for next round of federated learning
        for param in self.model.parameters():
            param.requires_grad = True
            
        return metrics_dict

app = Flask(__name__)

@app.route("/get_parameters", methods=["GET"])
def get_parameters():
    # Ensure model is in eval mode when sending parameters
    server_app.model.eval()
    parameters = server_app.get_parameters_for_clients()
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

    if server_app.strategy == "fedavg":
        server_app.client_updates.append(client_update)
    else:
        sparse_update, masks = client_update
        server_app.client_updates.append((sparse_update, masks))
    server_app.client_data_counts.append(data_count)

    logging.info(f"Received update. Total updates: {len(server_app.client_updates)}/{server_app.min_clients_agg}")

    if len(server_app.client_updates) >= server_app.min_clients_agg:
        logging.info("Starting aggregation...")
        server_app.aggregate_fit(server_app.client_updates, server_app.client_data_counts)
        logging.info("Aggregation complete. Starting evaluation...")
        
        eval_path = Path(GLOBAL_DATA_PATH) / "data.yaml"
        metrics = server_app.aggregate_evaluate(eval_path)
        
        # Clear updates for next round
        server_app.client_updates.clear()
        server_app.client_data_counts.clear()
        
        logging.info("Federation round complete.")
        return jsonify({"status": "Aggregation and evaluation complete", "metrics": metrics})

    return jsonify({"status": f"Update received. Waiting for more clients"})

if __name__ == "__main__":
    model = YOLO(MODEL_PATH)
    data_path = str(Path(GLOBAL_DATA_PATH) / "data.yaml")

    logging.info("pretraining global model .....")
    model = train_global_model(model, data_path, GLOBAL_EPOCHS)
    logging.info("pretraining complete")

    # Ensure all parameters are properly set up for federated learning
    for name, param in model.named_parameters():
        if isinstance(param, torch.nn.Parameter):
            param.data = param.data.detach().clone().contiguous()
            param.requires_grad_(True)

    # Set model to eval mode initially
    model.eval()
    
    server_app = ServerApp(model=model, strategy=STRATEGY, min_clients=CLIENTS_COUNT)
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=False)