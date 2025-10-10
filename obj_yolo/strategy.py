# strategy.py
import random
from typing import List, Dict, Optional
from abc import ABC, abstractmethod

import torch

from ultralytics.engine.results import Results

from .client import FedWegClient, FedTagClient
from .utils import (
    FitConfig,
    FitResFedWeg,
    from_coo,
)

class Strategy(ABC):
    @abstractmethod
    def configure_fit(self):
        pass

    @abstractmethod
    def aggregate_fit(self):
        pass

    @abstractmethod
    def aggregate_evaluate(self):
        pass

class FedWeg(Strategy):
    def __init__(self, client_data_paths:List[str], initial_sparsity:float=0.2, min_clients:int=2, client_epochs:int=10) -> None:
        self.min_clients_for_aggregation = min_clients
        self.initial_sparsity=initial_sparsity
        self.client_data_paths = client_data_paths
        self.client_epochs = client_epochs

    def configure_fit(self, num_supernodes:int, model_path:str) -> List[FedWegClient]:
        if self.min_clients_for_aggregation < num_supernodes:
            raise Exception(f"Min Clients for aggregation ({self.min_clients_for_aggregation}), exceeds the given clients({num_supernodes})")
        
        clients = []
        for x in range(num_supernodes):
            fitconfig = FitConfig(data_path=self.client_data_paths[x], epochs=self.client_epochs)
            client = FedWegClient(model_path=model_path, client_id=x, sparsity=self.initial_sparsity, fitconfig=fitconfig)
            clients.append(client)
        return clients
    
    def aggregate_fit(
            self,
            global_state: Dict[str, torch.Tensor], 
            results: List[FitResFedWeg]
        ) -> Dict[str, torch.Tensor]:
        """
        Aggregate the clients using Inverse sparsity method
        """
        assert len(results) == self.min_clients_for_aggregation, \
                f"Expected at least {self.min_clients_for_aggregation} clients, got {len(results)}"

        # compute inverse sparse weights
        sparsities = [res.sparsity for res in results]
        inv_sparsities = [1.0 / max(s, 0.2) for s in sparsities]
        inv_sparsity_sum = sum(inv_sparsities)

        # Initialize aggregated states
        agg_state = {k: v.clone().detach() for k, v in global_state.items()}

        # Check key consistency
        expected_keys = {k for k in agg_state.keys() if not k.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}
        all_delta_keys = set().union(*(res.delta.keys() for res in results))
        assert expected_keys == all_delta_keys, \
                f"Key mismatch at delta keys and agg keys, missing: {expected_keys - all_delta_keys}, extra: {all_delta_keys - expected_keys}"

        # Aggregate client updates
        for i, res in enumerate(results):
            weight = inv_sparsities[i] / inv_sparsity_sum
            for key, delta in res.delta.items():
                if key.endswith(('running_mean', 'running_var', 'num_batches_tracked')):
                    continue
                if isinstance(delta, torch.Tensor) and delta.is_sparse:
                    dense_delta = from_coo(delta)
                    agg_state[key] += weight * dense_delta
                elif isinstance(delta, torch.Tensor):
                    agg_state[key] += weight * delta
                else:
                    raise TypeError(f"Delta for key {key} is not a torch.Tensor")

        assert agg_state.keys() == expected_keys, \
                f"Key mistch of agg state after aggregation, missing {expected_keys - agg_state.keys()}, extra: {agg_state.keys() - expected_keys}"

        return agg_state
    
    def aggregate_evaluate(self, agg_state:Dict[str, torch.Tensor], clients:Optional[List[FedWegClient]], data_path:str) -> List[Results]:
        agg_results = []
        for client in clients:
            client.update_model(parameters=agg_state)
            result = client.evaluate(data_path=data_path)
            agg_results.append(result)
        return agg_results
    
    def update_sparsity(self, rounds_participated:int) -> float:
        min_sparsity = 0.2
        max_sparsity = 0.8

        new_sparsity = min_sparsity + (rounds_participated * 0.1)
        sparsity = torch.clip([new_sparsity], min=min_sparsity, max=max_sparsity)
        return sparsity[0]

class FedTag(FedWeg):
    def __init__(self, initial_sparsity:float, min_clients:int=2):
        super().__init__(initial_sparsity=initial_sparsity, min_clients=min_clients)
        self.tags:Optional[Dict] = None

    def configure_fit(self, num_supernodes:int, model_path:str, tags:List) -> List[FedTagClient]:
        tag_dict = {}
        clients = []
        for client_id in range(num_supernodes):
            tag = random.choice(tags)
            client = FedTagClient(model_path=model_path, client_id=client_id, \
                    sparsity=self.initial_sparsity, tag=tag)
            tag_clients = tag_dict.get(tag, [])
            tag_clients.append(client_id)
            tag_dict[tag] = tag_clients
            clients.append(client)
        self.tags = tag_dict
        return clients
    
    def update_sparsity(self, rounds_participated:int, change:float) -> float:
        min_sparsity = 0.2
        max_sparsity = 0.8

        new_sparsity = min_sparsity + (rounds_participated * change)
        sparsity = torch.clip([new_sparsity], min=min_sparsity, max=max_sparsity)
        return sparsity[0]

    def get_tag_dict(self):
        return self.tags
    
