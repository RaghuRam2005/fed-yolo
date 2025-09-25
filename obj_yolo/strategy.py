# strategy.py
import numpy as np
from typing import List, Dict

from torch import Tensor
from ultralytics.engine.results import Results

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
        assert (len(results) >= self.min_clients_for_aggregation), \
        f"Not enough results for aggregation. Expected >= {self.min_clients_for_aggregation}, Got {len(results)}"
        
        total_data = sum(res.datacount for res in results)
        agg_state = {k: v.clone() * 0.0 for k, v in global_state.items()} # Initialize with zeros

        # Perform weighted aggregation of model updates
        for res in results:
            weight = res.datacount / total_data
            for key, value in res.delta.items():
                if key not in agg_state: continue # Skip keys not in the global model

                if isinstance(value, tuple) and key.endswith('bn.weight'):
                    # Reconstruct dense tensor from sparse update
                    indices, params = value
                    update_tensor = agg_state[key].clone() # Use a zero tensor of the correct shape
                    update_tensor[indices] = Tensor(params)
                    agg_state[key] += update_tensor * weight
                elif isinstance(value, Tensor):
                    agg_state[key] += value * weight
        
        # Apply the aggregated delta to the original global state
        final_state = {k: v.clone() for k, v in global_state.items()}
        for key, delta in agg_state.items():
             if key.endswith(('running_mean', 'running_var', 'num_batches_tracked')):
                continue
             final_state[key] += delta

        return final_state

    def position_clients(self, results:Dict[int, Results]) -> Dict[int, int]:
        """
        Ranks clients based on their evaluation metrics (mAP).
        A higher mAP score results in a better (lower) rank.

        Args:
            results (Dict): Maps client_id to their evaluation metrics object.
                            Example: {0: <Results object>, 1: <Results object>}

        Returns:
            Dict[int, int]: Maps each client_id to its rank (1-based).
                            Example: {client_id_best: 1, client_id_worst: 2}
        """
        if not results:
            return {}

        client_scores = {
            client_id: metrics.box.map
            for client_id, metrics in results.items()
        }

        # Sort client_ids by score in descending order (higher mAP is better)
        sorted_clients = sorted(client_scores.items(), key=lambda item: item[1], reverse=True)

        # Assign ranks (1 is the best)
        client_ranks = {
            client_id: rank + 1
            for rank, (client_id, _) in enumerate(sorted_clients)
        }
        return client_ranks
