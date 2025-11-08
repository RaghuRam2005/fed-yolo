""" strategy for federated learning """
import os
from pathlib import Path
from typing import OrderedDict, Tuple, Union, Optional

from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays, FitRes, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

import torch

from ultralytics import YOLO

def get_section_parameters(state_dict:OrderedDict) -> Tuple[dict, dict, dict]:
    """
    Get parameters of each section of the model

    refer to: https://github.com/KCL-BMEIS/UltraFlwr/blob/main
    This section is adopted from UltraFlwr

    Args:
        state_dict (OrderedDict)

    Returns:
        Tuple[dict, dict, dict]
    """
    # Backbone parameters (early layers through conv layers)
    # backbone corresponds to:
    # (0): Conv
    # (1): Conv
    # (2): C3k2
    # (3): Conv
    # (4): C3k2
    # (5): Conv
    # (6): C3k2
    # (7): Conv
    # (8): C3k2
    backbone_weights = {
        k: v for k, v in state_dict.items()
        if not k.startswith(tuple(f'model.{i}' for i in range(9, 24)))
    }

    # Neck parameters
    # The neck consists of the following layers (by index in the Sequential container):
    # (9): SPPF
    # (10): C2PSA
    # (11): Upsample
    # (12): Concat
    # (13): C3k2
    # (14): Upsample
    # (15): Concat
    # (16): C3k2
    # (17): Conv
    # (18): Concat
    # (19): C3k2
    # (20): Conv
    # (21): Concat
    # (22): C3k2
    neck_weights = {
        k: v for k, v in state_dict.items()
        if k.startswith(tuple(f'model.{i}' for i in range(9, 23)))
    }

    # Head parameters (detection head)
    head_weights = {
        k: v for k, v in state_dict.items()
        if k.startswith('model.23')
    }

    return backbone_weights, neck_weights, head_weights

class CustomFedAvg(FedAvg):
    def __init__(self, *, fraction_fit = 1, fraction_evaluate = 1, min_fit_clients = 2, min_evaluate_clients = 2, min_available_clients = 2, evaluate_fn = None, on_fit_config_fn = None, on_evaluate_config_fn = None, accept_failures = True, initial_parameters = None, fit_metrics_aggregation_fn = None, evaluate_metrics_aggregation_fn = None, inplace = True):
        super().__init__(fraction_fit=fraction_fit, fraction_evaluate=fraction_evaluate, min_fit_clients=min_fit_clients, min_evaluate_clients=min_evaluate_clients, min_available_clients=min_available_clients, evaluate_fn=evaluate_fn, on_fit_config_fn=on_fit_config_fn, on_evaluate_config_fn=on_evaluate_config_fn, accept_failures=accept_failures, initial_parameters=initial_parameters, fit_metrics_aggregation_fn=fit_metrics_aggregation_fn, evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn, inplace=inplace)
        BASE_LIB_PATH = os.path.abspath(os.path.dirname(__file__))
        BASE_DIR_PATH = os.path.dirname(BASE_LIB_PATH)
        self.model_path = Path(BASE_DIR_PATH) / "yolo_config" / "yolo11n.yaml"
    
    def __repr__(self):
        rep = f"FedAveraging merged with ultralytics, accept failures = {self.accept_failures}"
        return rep

    def load_and_update_model(self, aggregated_parameters: Parameters) -> YOLO:
        net = YOLO(self.model_path)
        current_state_dict = net.model.state_dict()
        backbone_weights, neck_weights, head_weights = get_section_parameters(state_dict=current_state_dict)
        aggregated_ndarrays = parameters_to_ndarrays(parameters=aggregated_parameters)

        relevant_keys = []
        for k in sorted(current_state_dict.keys()):
            if (k in backbone_weights) or (k in neck_weights) or (k in head_weights):
                relevant_keys.append(k)
        
        if len(aggregated_ndarrays) != len(relevant_keys):
            strategy_name = self.__class__.__name__
            raise ValueError(
                f"Mismatch in aggregated parameter count for strategy {strategy_name}: "
                f"received {len(aggregated_ndarrays)}, expected {len(relevant_keys)}"
            )
        
        params_dict = zip(relevant_keys, aggregated_ndarrays)
        updated_weights = {k : torch.Tensor(v) for k, v in params_dict}
        final_state_dict = current_state_dict.copy()
        final_state_dict.update(updated_weights)
        net.model.load_state_dict(final_state_dict, strict=True)
        return net

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint."""
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            net = self.load_and_update_model(aggregated_parameters)            
            full_parameters = [val.cpu().numpy() for _, val in net.model.state_dict().items()]
            return ndarrays_to_parameters(full_parameters), aggregated_metrics
            
        return aggregated_parameters, aggregated_metrics
