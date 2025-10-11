#server.py
from typing import List, Dict
from abc import ABC, abstractmethod

import torch

from ultralytics.engine.model import Model
from ultralytics.engine.results import Results

from .strategy import FedWeg, FedTag
from .utils import (
    ServerConfigFedTag,
    ServerConfigFedWeg,
    model_state,
    update_sparsity_for_all_clients,
)

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
    def start_aggregation(self):
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
                num_supernodes, train_data_count=self.config.client_train_data_count, val_data_count=\
                    self.config.client_val_data_count)
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
    
    def update_sparsity(self):
        for client in self.clients:
            client.sparsity = self.strategy.update_sparsity(rounds_participated=client.rounds_completed)

class FedTagServer(FedWegServer):
    def __init__(self, model:Model, strategy:FedTag, config:ServerConfigFedTag):
        super().__init__(model=model, strategy=strategy, config=config)
    
    def update_client_models(self):
        return super().update_client_models()

    def update_global_model(self):
        return super().update_global_model()
    
    def start_clients(self):
        model_path = self.config.client_model_path
        num_supernodes = self.config.num_nodes
        tags_dict = self.config.client_tag_dict
        self.clients = self.strategy.configure_fit(
            model_path=model_path,
            num_supernodes=num_supernodes,
            tag_dict=tags_dict,
        )
        self.update_client_models()
    
    def fit_clients(self):
        return super().fit_clients()
    
    def start_aggregation(self):
        return super().start_aggregation()

    def start_evaluation(self):
        tags = self.strategy.get_tag_dict()
        tag_based_results = {}
        
        if not self.clients:
            raise Exception("Clients not initialized for evaluation.")

        client_map = {client.client_id: client for client in self.clients}

        for tag, client_ids in tags.items():
            if client_ids:
                selected_client_id = client_ids[0]
                selected_client = client_map.get(selected_client_id)
                
                if selected_client:
                    results = selected_client.evaluate()
                    map_score = results.box.map
                    tag_based_results[tag] = map_score
        return tag_based_results

    def update_sparsity(self, results:Dict[str, float]) -> None:
        update_sparsity_for_all_clients(fedtag=self.strategy, clients=self.clients, results=results)
