#client.py
import os
from pathlib import Path
from typing import Dict, List
from abc import ABC, abstractmethod

import torch
from torch.nn.modules.batchnorm import _BatchNorm

from ultralytics import YOLO
from ultralytics.engine.model import Model
from ultralytics.engine.results import Results
from ultralytics.utils.torch_utils import unwrap_model

from .dataset import KittiData, BddData
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
        self.fitconfig = fitconfig
        self.rounds_completed = 0
        self.data_path = None
    
    def update_model(self, parameters:Dict[str, torch.Tensor]) -> None:
        expected_keys = [k for k, _ in self.model.model.model.named_parameters()]
        parameters_keys = [k for k, _ in parameters.items() if not any(k.endswith(skip) for skip in SKIP_KEYS)]
        assert expected_keys == parameters_keys, f"Client {self.client_id} Error: update model parameter keys and model keys doesn't match"
        self.model.model.model.load_state_dict(parameters, strict=True)
    
    def prepare_data(self, data_class:KittiData, train_data_count:int, val_data_count:int) -> None:
        self.data_path = data_class.prepare_client_data(client_id=self.client_id, train_data_count=train_data_count,\
                                                        val_data_count=val_data_count)

    def fit(
            self,
            global_state:Dict
    ) -> FitResFedWeg:
        init_client_params = global_state
        #expected_keys = [name for name, _ in self.model.model.model.named_parameters()]

        # training the local model
        if not self.data_path:
            raise Exception(f"client {self.client_id}:Training data path not found")
        elif not Path(self.data_path).exists():
            raise Exception(f"client {self.client_id}: Data not found at {self.data_path}")
        results = client_train(
                model=self.model,
                data_path=self.data_path,
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
        # assert expected_keys == list(delta.keys()), f"Client {self.client_id}: delta parameters deos not match client parameter,, delta_keys: {delta.keys()}, expected_keys: {expected_keys}"

        self.rounds_completed += 1
        client_res = FitResFedWeg(delta=delta, metrics=results, sparsity=self.sparsity)
        return client_res

    def evaluate(self) -> Results:
        metrics = self.model.val(
            data=self.data_path,
            save_json=True,
            plots=True,
            project="fed_yolo",
            name=f"client_{self.client_id}_val",
        )

        return metrics

class FedTagClient(FedWegClient):
    def __init__(self, model_path:str, client_id:int, sparsity:float, tag:str, fitconfig:FitConfig) -> None:
        super().__init__(model_path=model_path, client_id=client_id, sparsity=sparsity, fitconfig=fitconfig)
        self.tag = tag
    
    def prepare_data(self, data_class:BddData, train_img_list:List, val_img_list:List):
        self.data_path = data_class.prepare_client_data(client_id=self.client_id, train_img_list=train_img_list, \
                                                        val_img_list=val_img_list)
    
