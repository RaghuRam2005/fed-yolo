#client.py
import os
from typing import Dict
from abc import ABC, abstractmethod

import torch
from torch.nn.modules.batchnorm import _BatchNorm

from ultralytics import YOLO
from ultralytics.engine.model import Model
from ultralytics.engine.results import Results
from ultralytics.utils.torch_utils import unwrap_model

from .utils import (
    FitResFedWeg,
    FitConfig,
    client_train,
    to_coo,
    model_state,
)

SKIP_KEYS = ['running_mean', 'running_var', 'num_batches_tracked']

class Client(ABC):
    @abstractmethod
    def update_model(self):
        pass
    
    @abstractmethod
    def prepare_data(self):
        pass

    @abstractmethod
    def fit(self):
        pass
    
    @abstractmethod
    def evaluate(self):
        pass

class FedWegClient(Client):
    def __init__(self, model_path:str, client_id:int, sparsity:float, fitconfig:FitConfig) -> None:
        self.model = YOLO(model_path)
        self.client_id = client_id
        self.sparsity = sparsity
        self.rounds_completed = 0
        self.fitconfig = fitconfig
    
    def update_model(self, parameters:Dict[str, torch.Tensor]) -> None:
        expected_keys = [k for k, _ in self.model.model.model.named_parameters()]
        parameters_keys = [k for k, _ in parameters.items() if not any(k.endswith(skip) for skip in SKIP_KEYS)]
        assert expected_keys == parameters_keys, f"Client {self.client_id} Error: update model parameter keys and model keys doesn't match"
        self.model.model.model.load_state_dict(parameters, strict=True)

    def prepare_data(self):
        

    def fit(
            self
    ):
        init_client_params = model_state(self.model)

        # training the local model
        results = client_train(
                model=self.model,
                data_path=self.fitconfig.data_path,
                client_id=self.client_id,
                epochs=self.fitconfig.epochs,
        )

        # unwrap the model
        unwrapped_model = unwrap_model(self.model)

        # compute tau based on BatchNorm weights
        bn_modules = [m for _, m in unwrapped_model.model.model.named_modules() if isinstance(m, _BatchNorm)]
        bn_weights = torch.cat([m.weight.abs().detach() for m in bn_modules])
        tau = torch.quantile(input=bn_weights, q=self.sparsity)

        # prepare set of BatchNorm weight parameter names
        bn_w_names = {f"{name}.weight" for name, m in unwrapped_model.model.model.named_modules() if isinstance(m, _BatchNorm)}

        # initialize deltas
        delta = {}

        # compute deltas
        for name, param in self.model.model.model.named_parameters():
            delta_tensor = param - init_client_params[name]
            
            if name in bn_w_names:
                delta[name] = to_coo(delta_tensor, tau)
            else:
                delta[name] = delta_tensor

        # validation step
        expected_keys = [k for k, _ in init_client_params.items() if not any(k.endswith(skip) for skip in SKIP_KEYS)]
        assert expected_keys == delta.keys(), f"Client {self.client_id}: delta parameters deos not match client parameter"

        self.rounds_completed += 1
        client_res = FitResFedWeg(delta=delta, metrics=results, sparsity=self.sparsity)
        return client_res

    def evaluate(self) -> Results:
        metrics = self.model.val(
            data=self.fitconfig.data_path,
            save_json=True,
            plots=True,
            project="fed_yolo",
            name=f"client_{self.client_id}_val",
        )

        return metrics

class FedTagClient(FedWegClient):
    def __init__(self, model_path:str, client_id:int, sparsity:float, tag:str) -> None:
        super().__init__(model_path=model_path, client_id=client_id, sparsity=sparsity)
        self.tag = tag
