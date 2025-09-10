# strategy.py
import random
from typing import List, OrderedDict

import torch

from client import ClientManager, FitRes

class Strategy:
    """
    Defines the logic for client selection and aggregation.
    """
    def __init__(self, min_clients_aggregation: int = 2):
        if min_clients_aggregation < 1:
            raise ValueError("min_clients_aggregation must be at least 1.")
        self.min_clients_aggregation = min_clients_aggregation

    @staticmethod
    def update_sparsity(manager: ClientManager) -> None:
        """
        Updates the client's sparsity parameter based on the number of rounds completed.
        Sparsity increases as the training progresses.

        Args:
            manager (ClientManager): The client manager instance to update.
        """
        min_sparsity = 0.2
        max_sparsity = 0.9
        growth_rate = 0.05

        target = min_sparsity + growth_rate * manager.rounds_completed
        new_sparsity = min(max(target, min_sparsity), max_sparsity)
        manager.fit_params.sparsity = new_sparsity

    def sample(self, available_clients: List[ClientManager]) -> List[ClientManager]:
        """
        Selects a random subset of clients for a training round.

        Args:
            available_clients (List[ClientManager]): The list of all clients managed by the server.

        Returns:
            List[ClientManager]: A list of selected client managers.
        """
        num_available = len(available_clients)
        if num_available < self.min_clients_aggregation:
            raise ValueError(
                f"Not enough clients available ({num_available}) to meet the minimum "
                f"aggregation requirement of {self.min_clients_aggregation}."
            )
        
        num_to_sample = min(num_available, self.min_clients_aggregation)
        return random.sample(available_clients, num_to_sample)

    def aggregate_fit(self, results: List[FitRes], global_state: OrderedDict) -> OrderedDict:
        """
        Aggregates client model updates (deltas) into the global model state
        using inverse sparsity weighting.
        """
        if not results:
            return global_state

        sparsities = [res.results.get("sparsity", 0.2) for res in results]
        inv_sparsities = [1.0 / s for s in sparsities]
        inv_sum = sum(inv_sparsities)

        agg_state = {k: v.clone() for k, v in global_state.items()}

        for i, result in enumerate(results):
            client_weight = inv_sparsities[i] / inv_sum
            for key, weight in agg_state.items():
                if key in result.delta.keys():
                    if key.endswith('bn.weight'):
                        mask, params = result.delta[key]
                        for idx, param in zip(mask, params):
                            weight[idx] += client_weight * param
                    else:
                        delta_val = result.delta[key]
                        # Handle floating point vs. integer tensors correctly
                        if torch.is_floating_point(weight):
                            weight += client_weight * delta_val
                        else:
                            # Handle integer tensors like num_batches_tracked
                            weight += (client_weight * delta_val).round().long()
                    agg_state[key] = weight
                else:
                    print(f"{key} not found in result.delta")

        return agg_state