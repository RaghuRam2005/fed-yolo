#client.py
import os
import logging
from typing import Dict, Optional

import torch
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm

from ultralytics.engine.model import Model
from ultralytics.engine.results import Results
from ultralytics.utils.torch_utils import unwrap_model

from .utils import (
    ClientConfig,
    ClientFitRes,
    client_train,
    set_parameters,
    generate_mask,
    log_pruning_statistics,
    apply_mask_to_model,
    create_sparse_update,
    channel_index_masks,
    to_coo
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
    
    def fit(self, client_config: ClientConfig, data_path: str) -> ClientFitRes:
        if not self.model:
            raise AttributeError(f"client {self.client_id} model attribute is not set")

        # Get initial parameters
        client_parameters = self.model.model.model.state_dict()

        # Train the client model
        results = client_train(
            model=self.model,
            data_path=data_path,
            client_id=self.client_id,
            epochs=client_config.epochs,
        )

        # Unwrap model once
        unwrapped_model = unwrap_model(self.model)

        # Compute tau based on BatchNorm weights
        bn_modules = [m for _, m in unwrapped_model.named_modules() if isinstance(m, _BatchNorm)]
        bn_weights = torch.cat([m.weight.abs().detach() for m in bn_modules])
        tau = torch.quantile(bn_weights, q=self.sparsity)

        # Prepare set of BatchNorm weight parameter names
        bn_weight_names = {f"{name}.weight" for name, m in unwrapped_model.named_modules() if isinstance(m, _BatchNorm)}

        # Compute deltas
        delta = {}
        skip_keys = ['running_mean', 'running_var', 'num_batches_tracked']

        for name, param in self.model.model.model.named_parameters():
            if any(skip in name for skip in skip_keys):
                continue  # skip BN statistics

            delta_tensor = param - client_parameters[name]

            if name in bn_weight_names:
                delta[name] = to_coo(delta_tensor, tau)  # sparsify BN weights
            else:
                delta[name] = delta_tensor  # dense delta for other params

        # Sanity check: delta keys must match client_parameters keys (excluding skipped)
        expected_keys = {k for k in client_parameters.keys() if not any(skip in k for skip in skip_keys)}
        if set(delta.keys()) != expected_keys:
            missing = expected_keys - set(delta.keys())
            extra = set(delta.keys()) - expected_keys
            raise ValueError(f"Delta parameters do not match client parameters. Missing: {missing}, Extra: {extra}")

        client_res = ClientFitRes(delta=delta, metrics=results, datacount=1000, sparsity=self.sparsity)
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
