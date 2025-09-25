#client.py
import os
import logging
from typing import Dict, Optional

from torch import Tensor
from ultralytics.engine.model import Model
from ultralytics.engine.results import Results

from .utils import (
    ClientConfig,
    ClientFitRes,
    client_train,
    set_parameters,
    generate_mask,
    log_pruning_statistics,
    apply_mask_to_model,
    create_sparse_update
)

weather_tags = ["clear", "overcast", "snowy", "rainy", "partial cloudy"] 
scene_tags = ["city street", "highway", "residential"]

class Client:
    def __init__(self, client_id:int, sparsity:float, tag:str):
        self.client_id = client_id
        self.sparsity = sparsity
        self.model:Model = None
        self.rounds_completed = 0
        self.tag = tag
    
    def update_model(self, parameters:Dict[str, Tensor]):
        self.model = set_parameters(model=self.model, parameters=parameters)

    def update_sparsity(self, rank: int, num_clients: int) -> None:
        """
        Updates the client sparsity based on its performance rank.
        A better rank (lower number, e.g., 1) results in lower sparsity.
        
        Args:
            rank (int): The performance rank of the client (1 is best).
            num_clients (int): Total number of clients in the round.
        """
        min_sparsity = 0.2
        max_sparsity = 0.8
        
        # Calculate sparsity step size
        sparsity_step = (max_sparsity - min_sparsity) / (num_clients - 1) if num_clients > 1 else 0
        
        # Better rank (lower number) -> lower sparsity
        new_sparsity = min_sparsity + (rank - 1) * sparsity_step
        
        self.sparsity = min(max(new_sparsity, min_sparsity), max_sparsity)
        
        logging.info(f"Client {self.client_id} (Tag: {self.tag}, Rank: {rank}) sparsity updated to {self.sparsity:.4f}")
    
    def fit(self, client_config:ClientConfig, data_path:str) -> ClientFitRes:
        if not self.model:
            raise AttributeError(f"client {self.client_id} model attribute is not set")
        
        client_parameters = self.model.model.model.state_dict()
        
        results = client_train(
            model=self.model,
            data_path=data_path,
            client_id=self.client_id,
            epochs=client_config.epochs,
        )

        mask = generate_mask(model=self.model, sparsity=self.sparsity)
        log_pruning_statistics(model=self.model, mask=mask)

        delta = {}
        for key, weights in self.model.model.model.named_parameters():
            if key.endswith(('running_mean', 'running_var', 'num_batches_tracked')):
                delta[key] = weights
            else:
                delta[key] = weights - client_parameters[key]
        
        sparse_weights = apply_mask_to_model(delta=delta, mask=mask)
        sparse_update = create_sparse_update(parameters=sparse_weights)
        client_res = ClientFitRes(delta=sparse_update, metrics=results, datacount=1000, sparsity=self.sparsity)
        self.rounds_completed += 1
        return client_res

    def evaluate(self, data_path:str) -> Results:
        metrics = self.model.val(
            data=data_path,
            save_json=True,
            plots=True,
            project="fed_yolo",
            name=f"client_{self.client_id}_val",
            exist_ok=True
        )
        return metrics
