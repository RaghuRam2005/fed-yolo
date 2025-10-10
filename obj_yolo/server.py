#server.py
from typing import List, Dict
from abc import ABC, abstractmethod

import torch

from ultralytics.engine.model import Model
from ultralytics.engine.results import Results

from .utils import ServerConfigFedWeg, model_state, FitConfig
from .strategy import FedWeg, FedTag

class Server(ABC):
    @abstractmethod
    def update_global_model(self):
        pass

    @abstractmethod
    def update_client_models(self):
        pass

    @abstractmethod
    def start_clients(self):
        pass

    @abstractmethod
    def fit_clients(self):
        pass
    
class FedWegServer(Server):
    def __init__(self, model:Model, strategy:FedWeg, config:ServerConfigFedWeg) -> None:
        self.global_model = model
        self.global_state = model_state(self.global_model)
        self.strategy = strategy
        self.config = config
        self.clients = None

    def update_global_model(self, parameters:Dict[str, torch.Tensor]) -> None:
        expected_keys = [k for k, _ in self.global_state.keys() if not k.endswith(('running_mean', 'running_var', 'num_batches_tracked'))]
        param_keys = [k for k, _ in parameters.keys() if not k.endswith(('running_mean', 'running_var', 'num_batches_tracked'))]
        assert expected_keys == param_keys, "keys doesn't match while updating the global model"
        self.global_model.model.model.load_state_dict(parameters)
        self.global_state = model_state(self.global_model)

    def update_client_models(self) -> None:
        if not self.clients:
            return
        for client in self.clients:
            client.update_model(self.global_state)

    def start_clients(self) -> None:
        model_path = self.config.client_model_path
        num_supernodes = self.config.num_nodes
        self.clients = self.strategy.configure_fit(model_path=model_path, num_supernodes=\
                num_supernodes)
        self.update_client_models()

    def fit_clients(self) -> List:
        results = []
        if not self.clients:
            raise Exception("clients not found while running fit")
        for client in self.clients:
            result = client.fit()
            results.append(result)
        self.results = results
        return results
    
    def start_aggregation(self) -> Dict[str, torch.Tensor]:
        if not self.results:
            raise Exception("client results not found while running aggregation")
        agg_state = self.strategy.aggregate_fit(global_state=self.global_state, results=self.results)
        self.update_global_model(parameters=agg_state)
        self.update_client_models()
        return agg_state
    
    def start_evaluation(self) -> List[Results]:
        eval_results = []
        for client in self.clients:
            results = client.evaluate()
            eval_results.append(results)
        return eval_results

class FedTagServer(Server):
    def __init__(self, model:Model, strategy:FedTag, config:):
