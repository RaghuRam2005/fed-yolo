# client.py
import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

import torch
from ultralytics.engine.model import Model

from utils import yolo_train, set_parameters, apply_mask_to_model, create_sparse_update

@dataclass
class ClientFitParams:
    """
    Data class to store client parameters for fitting.
    """
    sparsity: float = 0.2
    epochs: int = 2
    lam: float = 1e-5

@dataclass
class ClientManager:
    """
    Data class to store client state and configuration.
    """
    client_id: int
    fit_params: ClientFitParams
    model: Model
    rounds_completed: int = 0

@dataclass
class FitRes:
    """
    Data class for fit results from a client.
    """
    delta: Optional[Dict[str, Tuple]]
    results: Dict[str, Optional[float]]
    data_count: int = 1000

class Client:
    """
    Represents a single client in the federated learning setup.
    It handles model training and evaluation on its local data.
    """
    def __init__(self, client_manager: ClientManager):
        self.client_manager = client_manager

    def update_client_model(self, parameters: List[np.ndarray]) -> None:
        """
        Updates the client's local model with new parameters.
        """
        self.client_manager.model = set_parameters(
            model=self.client_manager.model,
            parameters=parameters,
            name=f"Client {self.client_manager.client_id}"  # Add descriptive name
        )

    def fit(self, data_path: str) -> FitRes:
        """
        Trains the local model on the client's data.
        """
        model = self.client_manager.model
        original_state = {k: v.clone() for k, v in model.model.model.state_dict().items()}

        # Train locally
        model, results, mask = yolo_train(
            model, data_path,
            self.client_manager.fit_params.epochs,
            self.client_manager.fit_params.sparsity,
            self.client_manager.fit_params.lam
        )
        
        # This is the fix for the KeyError
        delta = {}
        with torch.no_grad():
            updated_state = model.model.model.state_dict()
            for key, weights in updated_state.items():
                if key in original_state:
                    if key.endswith(("running_mean", "running_var", "num_batches_tracked")):
                        delta[key] = weights
                    elif weights.shape == original_state[key].shape:
                        delta[key] = weights - original_state[key]
                    else:
                        delta[key] = weights
                else:
                    delta[key] = weights
        
        sparse_weights = apply_mask_to_model(delta, mask)
        sparse_update = create_sparse_update(sparse_weights)

        metrics = {
            "map50-95": results.box.map,
            "map50": results.box.map50,
            "map75": results.box.map75,
            "sparsity": self.client_manager.fit_params.sparsity,
        }

        return FitRes(delta=sparse_update, results=metrics)

    def evaluate(self, paramaters: List[np.ndarray], data_path: str) -> Dict[str, float]:
        """
        Evaluates the given model parameters on the client's local validation data.
        """
        model = set_parameters(
            model=self.client_manager.model,
            parameters=paramaters,
            name=f"Client {self.client_manager.client_id} (pre-eval)" # Add descriptive name
        )
        results = model.val(
            data=data_path,
            save=False,
            device=-1, # Use CPU for evaluation
            verbose=False,
        )
        metrics = {
            "mAP50-95": results.box.map,
            "map50": results.box.map50,
            "map75": results.box.map75,
        }
        return metrics