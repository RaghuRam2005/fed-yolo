""" strategy for federated learning """
import os
import time
from logging import INFO
from pathlib import Path
from typing import OrderedDict, Tuple, Union, Optional, Callable

from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays, FitRes, Scalar, log, logger
from flwr.app import ArrayRecord, ConfigRecord
from flwr.server.client_proxy import ClientProxy
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg, Result
from flwr.serverapp.strategy.strategy_utils import log_strategy_start_info

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

    def start(
            self,
            grid: Grid,
            initial_arrays: ArrayRecord,
            num_rounds:int = 3,
            timeout: float = 3600,
            train_config:Optional[ConfigRecord]=None,
            evaluate_config:Optional[ConfigRecord]=None,
            evaluate_fn: Optional[
            Callable[[int, ArrayRecord], Optional[MetricRecord]]
        ] = None,
    ) -> Result:
        self.best_acc_so_far = 0.0

        log(INFO, "Starting %s strategy", self.__class__.__name__)
        log_strategy_start_info(num_rounds, initial_arrays, train_config, evaluate_config)
        self.summary()
        log(INFO, "")

        # Initialize if None
        train_config = ConfigRecord() if train_config is None else train_config
        evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config
        result = Result()

        t_start = time.time()
        # Evaluate starting global parameters
        if evaluate_fn:
            res = evaluate_fn(0, initial_arrays)
            log(INFO, "Initial global evaluation results: %s", res)
            if res is not None:
                result.evaluate_metrics_serverapp[0] = res

        arrays = initial_arrays

        for current_round in range(1, num_rounds + 1):
            log(INFO, "")
            log(INFO, "[ROUND %s/%s]", current_round, num_rounds)

            # -----------------------------------------------------------------
            # --- TRAINING (CLIENTAPP-SIDE) -----------------------------------
            # -----------------------------------------------------------------

            # Call strategy to configure training round
            # Send messages and wait for replies
            train_replies = grid.send_and_receive(
                messages=self.configure_train(
                    current_round,
                    arrays,
                    train_config,
                    grid,
                ),
                timeout=timeout,
            )

            # Aggregate train
            agg_arrays, agg_train_metrics = self.aggregate_train(
                current_round,
                train_replies,
            )

            # Log training metrics and append to history
            if agg_arrays is not None:
                result.arrays = agg_arrays
                arrays = agg_arrays
            if agg_train_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_train_metrics)
                result.train_metrics_clientapp[current_round] = agg_train_metrics
                # Log to W&B
                wandb.log(dict(agg_train_metrics), step=current_round)

            # -----------------------------------------------------------------
            # --- EVALUATION (CLIENTAPP-SIDE) ---------------------------------
            # -----------------------------------------------------------------

            # Call strategy to configure evaluation round
            # Send messages and wait for replies
            evaluate_replies = grid.send_and_receive(
                messages=self.configure_evaluate(
                    current_round,
                    arrays,
                    evaluate_config,
                    grid,
                ),
                timeout=timeout,
            )

            # Aggregate evaluate
            agg_evaluate_metrics = self.aggregate_evaluate(
                current_round,
                evaluate_replies,
            )

            # Log training metrics and append to history
            if agg_evaluate_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_evaluate_metrics)
                result.evaluate_metrics_clientapp[current_round] = agg_evaluate_metrics
                # Log to W&B
                wandb.log(dict(agg_evaluate_metrics), step=current_round)
            # -----------------------------------------------------------------
            # --- EVALUATION (SERVERAPP-SIDE) ---------------------------------
            # -----------------------------------------------------------------

            # Centralized evaluation
            if evaluate_fn:
                log(INFO, "Global evaluation")
                res = evaluate_fn(current_round, arrays)
                log(INFO, "\t└──> MetricRecord: %s", res)
                if res is not None:
                    result.evaluate_metrics_serverapp[current_round] = res
                    # Maybe save to disk if new best is found
                    self._update_best_acc(current_round, res["accuracy"], arrays)
                    # Log to W&B
                    wandb.log(dict(res), step=current_round)

        log(INFO, "")
        log(INFO, "Strategy execution finished in %.2fs", time.time() - t_start)
        log(INFO, "")
        log(INFO, "Final results:")
        log(INFO, "")
        for line in io.StringIO(str(result)):
            log(INFO, "\t%s", line.strip("\n"))
        log(INFO, "")

        return result
