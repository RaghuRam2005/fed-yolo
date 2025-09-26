# strategy.py
import numpy as np
from typing import List, Dict

import torch
from torch import Tensor
from ultralytics.engine.results import Results

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

    def aggregate_fit(self, global_state: Dict[str, Tensor], results: List[ClientFitRes]) -> Dict[str, Tensor]:
        assert len(results) == self.min_clients_for_aggregation, \
            f"Expected {self.min_clients_for_aggregation} clients, got {len(results)}"

        # Compute inverse sparsity weights
        inv_sparsities = [1.0 / r.sparsity for r in results]
        total_inv_sparsity = sum(inv_sparsities)

        agg_state = {k: v.clone() for k, v in global_state.items()}

        skip_keys = ['running_mean', 'running_var', 'num_batches_tracked']

        for i, res in enumerate(results):
            delta_w = inv_sparsities[i] / total_inv_sparsity

            for key, delta in res.delta.items():
                if any(skip in key for skip in skip_keys):
                    continue

                if isinstance(delta, torch.Tensor) and delta.is_sparse:
                    # Convert sparse COO delta to dense
                    dense_delta = from_coo(delta)
                    agg_state[key] += delta_w * dense_delta
                elif isinstance(delta, torch.Tensor):
                    # Dense update
                    agg_state[key] += delta_w * delta
                else:
                    raise TypeError(f"Delta for key {key} is not a torch.Tensor")

        return agg_state

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
