# strategy.py
import numpy as np
from typing import List, Dict

import torch

from .utils import (
    ClientConfig,
    ClientFitRes,
    from_coo,
)

class Strategy():
    def __init__(self, min_clients_for_aggregation:int):
        self.min_clients_for_aggregation = min_clients_for_aggregation
        self.rounds_completed = 0
    
    def configure_fit(self, epochs:int):
        return ClientConfig(epochs=epochs)

    def aggregate_fit(self, global_state: Dict[str, torch.Tensor], results: List[ClientFitRes]) -> Dict[str, torch.Tensor]:
        assert len(results) == self.min_clients_for_aggregation, f"Expected at least {self.min_clients_for_aggregation} clients, got {len(results)}"

        # Compute inverse sparsity weights
        sparsities = [res.sparsity for res in results]
        inv_sparsities = [1.0 / max(s, 0.2) for s in sparsities]
        inv_sparsity_sum = sum(inv_sparsities)

        # Initialize aggregated state
        agg_state = {k: v.clone() for k, v in global_state.items()}

        # check key consistency
        expected_keys = {k for k in global_state if not k.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}
        all_delta_keys = set().union(*(res.delta.keys() for res in results))
        missing = expected_keys - all_delta_keys
        extra = all_delta_keys - expected_keys
        if missing or extra:
            raise ValueError(f"Delta parameters mismatch. Missing: {missing}, Extra: {extra}")

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
                
        # Aggregation key check
        expected_keys = {k for k in global_state if not k.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}
        agg_keys = {k for k in agg_state if not k.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}

        if agg_keys != expected_keys:
            missing = expected_keys - agg_keys
            extra = agg_keys - expected_keys
            raise ValueError(f"Aggregated state keys mismatch. Missing: {missing}, Extra: {extra}")

        return agg_state

