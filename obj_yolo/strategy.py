# strategy.py
import numpy as np
from typing import List, Dict

from torch import Tensor

from .utils import (
    ClientConfig,
    ClientFitRes
)

class Strategy():
    def __init__(self, min_clients_for_aggregation:int):
        self.min_clients_for_aggregation = min_clients_for_aggregation
        self.rounds_completed = 0
    
    def configure_fit(self, epochs:int):
        return ClientConfig(epochs=epochs)

    def aggregate_fit(self, global_state:Dict[str, Tensor], results:List[ClientFitRes]) -> Dict[str, Tensor]:
        assert (len(results) == self.min_clients_for_aggregation), \
        f"Not enough or Excessive results than {self.min_clients_for_aggregation}"
        
        sparcities = []
        for result in results:
            sparcities.append(result.sparsity)
        
        inv_sparcities = [1.0 / s for s in sparcities]
        inv_sparcity = sum(inv_sparcities)

        agg_state = {k:v.clone() for k, v in global_state.items()}
        for i, result in enumerate(results):
            delta_w = inv_sparcities[i] / inv_sparcity

            for d_key in result.delta.keys():
                if d_key.endswith(('running_mean', 'running_var', 'num_batches_tracked')):
                    continue
                assert d_key in agg_state.keys(), \
                f"result key at index {i}, key: {d_key} not found , client keys: {agg_state.keys()}"

            for key, weight in agg_state.items():
                if key.endswith('bn.weight'):
                    new_weight = np.zeros_like(weight, dtype=float)
                    ind, params = result.delta[key]
                    p_ind = 0
                    for j in ind:
                        new_weight[j] = params[p_ind]
                        p_ind += 1
                    weight += (delta_w * new_weight)
                    agg_state[key] = Tensor(weight)
                elif key.endswith(('running_mean', 'running_var', 'num_batches_tracked')):
                    continue
                else:
                    weight += (delta_w * result.delta[key])
                    agg_state[key] = Tensor(weight)
            
            for d_key in result.delta.keys():
                if d_key.endswith(('running_mean', 'running_var', 'num_batches_tracked')):
                    continue
                assert d_key in agg_state.keys(), \
                f"After aggregation result key at index {i}, key: {d_key} not found , client keys: {agg_state.keys()}"
        
        return agg_state
