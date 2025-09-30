#client.py
import os
from typing import Dict

import torch
from torch.nn.modules.batchnorm import _BatchNorm

from ultralytics.engine.model import Model
from ultralytics.engine.results import Results
from ultralytics.utils.torch_utils import unwrap_model

from .utils import (
    ClientConfig,
    ClientFitRes,

    client_train,
    to_coo,
)

class Client:
    def __init__(self, client_id:int, sparsity:float):
        self.client_id = client_id
        self.sparsity = sparsity
        self.model:Model = None
        self.rounds_completed = 0
    
    def update_model(self, parameters:Dict[str, torch.Tensor]):
        self.model.model.model.load_state_dict(parameters, strict=True)

    def update_sparsity(self) -> None:
        """
        Updates the client's sparsity parameter based on the number of rounds completed.
        Sparsity increases as the training progresses.
        """
        min_sparsity = 0.2
        max_sparsity = 0.9
        growth_rate = 0.05

        target = min_sparsity + growth_rate * self.rounds_completed
        new_sparsity = min(max(target, min_sparsity), max_sparsity)
        self.sparsity = new_sparsity
    
    def fit(self, client_config:ClientConfig, data_path:str) -> ClientFitRes:
        if not self.model:
            raise AttributeError(f"client {self.client_id} model attribute is not set")
        
        client_parameters = self.model.model.model.state_dict()
        
        # training the local model
        results = client_train(
            model=self.model,
            data_path=data_path,
            client_id=self.client_id,
            epochs=client_config.epochs,
        )

        # unwrap the model
        unwrapped_model = unwrap_model(self.model)

        # compute tau based on BatchNorm weights
        bn_modules = [m for _, m in unwrapped_model.model.model.named_modules() if isinstance(m, _BatchNorm)]
        bn_weights = torch.cat([m.weight.abs().detach() for m in bn_modules])
        tau = torch.quantile(input=bn_weights, q=self.sparsity)

        # Prepare set of BatchNorm weight parameter names
        bn_weight_names = {f"{name}.weight" for name, m in unwrapped_model.model.model.named_modules() if isinstance(m, _BatchNorm)}

        # initialize deltas
        delta = {}

        # compute deltas
        for name, param in self.model.model.model.named_parameters():
            delta_tensor = param - client_parameters[name]

            if name in bn_weight_names:
                delta[name] = to_coo(delta_tensor, tau)
            else:
                delta[name] = delta_tensor

        skip_keys = ['running_mean', 'running_var', 'num_batches_tracked']
        expected_keys = {k for k in client_parameters.keys() if not any(skip in k for skip in skip_keys)}
        if set(delta.keys()) != expected_keys:
            missing = expected_keys - set(delta.keys())
            extra = set(delta.keys()) - expected_keys
            raise ValueError(f"Delta parameters do not match client parameters. Missing: {missing}, Extra: {extra}")        

        client_res = ClientFitRes(delta=delta, metrics=results, datacount=1000, sparsity=self.sparsity)

        self.rounds_completed += 1
        self.update_sparsity()

        return client_res

    def evaluate(self, data_path:str) -> Results:
        metrics = self.model.val(
            data=data_path,
            save_json=True,
            plots=True,
            project="fed_yolo",
            name=f"client_{self.client_id}_val",
        )

        return metrics
