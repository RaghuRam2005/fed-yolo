import random
import numpy as np
from typing import Optional, List, Dict, OrderedDict

import torch
from ultralytics import YOLO

from utils import load_model, set_parameters
from simulation import Simulation
from client import ClientManager, ClientFitParams, FitRes

class Strategy:
    def __init__(self, min_clients_aggregation:int=2):
        self.min_clients_aggregation = min_clients_aggregation
    
    @staticmethod
    def update_sparsity(manager:ClientManager) -> None:
        """
        updates the sparsity of the clients everytime a new client is added
        and also every time a communication round starts

        Args:
            manager (ClientManager): Client Manager instance
        """
        min_sparsity = 0.2
        max_sparsity = 0.9
        growth_rate = 0.05

        # Base sparsity schedule: min_sparsity + growth_rate * rounds
        target = min_sparsity + growth_rate * manager.rounds_completed

        # Ensure we stay within [min_sparsity, max_sparsity]
        new_sparsity = min(max(target, min_sparsity), max_sparsity)

        manager.fit_params.sparsity = new_sparsity


    def sample(self, simulation:Simulation) -> List[int]:
        """
        select required clients from all the clients available

        Args:
            simulation (Simulation): Instance of simulation dataclass to know how many classes are available

        Returns:
            List[int]: List of selected clients
        """
        if simulation.num_supernodes < self.min_clients_aggregation:
            raise Exception("Not enough clients in simulation")
        clients = random.sample(range(simulation.num_supernodes), self.min_clients_aggregation)
        return clients

    def configure_fit(self, simulation:Simulation, global_state:Optional[OrderedDict]) -> List[ClientManager]:
        """
        Initializes client managers for selected clients.

        Args:
            simulation (Simulation): Simulation instance containing configuration and client info.
            global_state (Optional[OrderedDict]): Optional global model state_dict to load into client models.

        Returns:
            List[ClientManager]: List of initialized ClientManager instances for selected clients.
        """
        managers = []
        clients = self.sample(simulation)
        for cid in clients:
            model = YOLO(simulation.yolo_config)
            if global_state is not None:
                model.load_state_dict(global_state, strict=True)
            fit_params = ClientFitParams()
            managers.append(ClientManager(client_id=cid, fit_params=fit_params, model=model))
        return managers

    def aggregate_fit(self, results: List[FitRes], global_state: OrderedDict) -> OrderedDict:
        """
        Aggregates client model updates (deltas) into the global model state using inverse sparsity weighting.
        BatchNorm running statistics are skipped during aggregation, as they are recalibrated locally on clients.

        Args:
            results (List[FitRes]): List of client training results, each containing model deltas and sparsity info.
            global_state (OrderedDict): Current global model state_dict.

        Returns:
            OrderedDict: Updated global model state_dict after aggregation.
        """
        if not results:
            return global_state

        # Collect sparsities
        sparsities = [res.results.get("sparsity", 0.2) for res in results]
        inv_sparsities = [1.0 / s for s in sparsities]
        inv_sum = sum(inv_sparsities)

        # Clone current global state
        agg_state = {k: v.clone() for k, v in global_state.items()}

        def skip_bn_stat(key: str) -> bool:
            return any(stat in key for stat in ["running_mean", "running_var", "num_batches_tracked"])
        
        # Iterate through each client's results
        for i, result in enumerate(results):
            # Calculate the client's weight for this aggregation
            client_weight = inv_sparsities[i] / inv_sum
            
            # Aggregate backbone deltas
            for key, delta_tensor in result.delta["backbone_delta"].items():
                if skip_bn_stat(key):
                    continue
                if key in agg_state:
                    agg_state[key] += client_weight * delta_tensor.to(agg_state[key].device)

            # Aggregate head deltas
            for key, delta_tensor in result.delta["head_delta"].items():
                if skip_bn_stat(key):
                    continue
                if key in agg_state:
                    agg_state[key] += client_weight * delta_tensor.to(agg_state[key].device)

        return agg_state



    def aggreate_evaluate(self, parameters:List[np.ndarray], data_path:str) -> Dict[str, int]:
        """
        Evaluates the YOLO model with aggregated parameters in the server
        (this only happens if there is global data available in server)

        Args:
            parameters (List[np.ndarray]): List of aggregated parameters

        Returns:
            Dict[str, int]: results
        """
        pass
