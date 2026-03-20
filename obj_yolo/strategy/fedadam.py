"""Adaptive Federated Optimization using Adam (FedAdam) strategy.

[Reddi et al., 2020]

Paper: arxiv.org/abs/2003.00295
"""
import io
import os
import time
from logging import log, INFO
from pathlib import Path
from typing import Optional, Callable

from flwr.serverapp import Grid
from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    MetricRecord
)
from flwr.serverapp.strategy import FedAdam
from flwr.serverapp.strategy.result import Result
from flwr.serverapp.strategy.strategy_utils import (
    log_strategy_start_info,    
)

from obj_yolo.strategy.strategy_utils import (
    load_and_update_model
)

class CustomFedAdam(FedAdam):
    def __init__(self, *, fraction_train = 1, fraction_evaluate = 1, min_train_nodes = 2, min_evaluate_nodes = 2, min_available_nodes = 2, weighted_by_key = "num-examples", arrayrecord_key = "arrays", configrecord_key = "config", train_metrics_aggr_fn = None, evaluate_metrics_aggr_fn = None, eta = 0.1, eta_l = 0.1, beta_1 = 0.9, beta_2 = 0.99, tau = 0.001, dataset_name:str="kitti"):
        super().__init__(fraction_train=fraction_train, fraction_evaluate=fraction_evaluate, min_train_nodes=min_train_nodes, min_evaluate_nodes=min_evaluate_nodes, min_available_nodes=min_available_nodes, weighted_by_key=weighted_by_key, arrayrecord_key=arrayrecord_key, configrecord_key=configrecord_key, train_metrics_aggr_fn=train_metrics_aggr_fn, evaluate_metrics_aggr_fn=evaluate_metrics_aggr_fn, eta=eta, eta_l=eta_l, beta_1=beta_1, beta_2=beta_2, tau=tau)
        self.base_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
        self.dataset_name = dataset_name
        self.yolo_base_path = Path(self.base_path) / "yolo_config" / f"{dataset_name}_yolo11n.yaml"

    def start(
            self,
            grid:Grid,
            initial_arrays:ArrayRecord,
            num_rounds:int=3,
            timeout:float=3600,
            train_config: Optional[ConfigRecord] = None,
            evaluate_config: Optional[ConfigRecord] = None,
            evaluate_fn: Optional[
                Callable[[int, ArrayRecord], Optional[MetricRecord]]
            ] = None,
    ) -> Result:
        log(INFO, "Starting %s strategy:", self.__class__.__name__)
        log_strategy_start_info(
            num_rounds, initial_arrays, train_config, evaluate_config
        )
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

        for current_round in range(1, num_rounds+1):
            log(INFO, "")
            log(INFO, "[ROUND %s %s]", current_round, num_rounds)

            train_replies = grid.send_and_receive(
                messages=self.configure_train(
                    current_round,
                    arrays,
                    train_config,
                    grid,
                ),
                timeout=timeout,
            )

            agg_arrays, agg_train_metrics = self.aggregate_train(
                current_round, train_replies
            )
            
            if agg_arrays is not None:
                model = load_and_update_model(self.yolo_base_path, agg_arrays)
                model.save(Path(self.base_path) / "flwr_simulation" / f"{self.dataset_name}" / "aggregated_model" / "agg_model.pt")
                result.arrays = agg_arrays
                arrays = agg_arrays
            if agg_train_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_train_metrics)
                result.train_metrics_clientapp[current_round] = agg_train_metrics

            evaluate_replies = grid.send_and_receive(
                messages=self.configure_evaluate(
                    current_round,
                    arrays,
                    evaluate_config,    
                    grid,
                ),
                timeout=timeout,
            )

            agg_evaluate_metrics = self.aggregate_evaluate(
                current_round,
                evaluate_replies,
            )

            # Log training metrics and append to history
            if agg_evaluate_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_evaluate_metrics)
                result.evaluate_metrics_clientapp[current_round] = agg_evaluate_metrics

            # Centralized evaluation
            if evaluate_fn:
                log(INFO, "Global evaluation")
                res = evaluate_fn(current_round, arrays)
                log(INFO, "\t└──> MetricRecord: %s", res)
                if res is not None:
                    result.evaluate_metrics_serverapp[current_round] = res

        log(INFO, "")
        log(INFO, "Strategy execution finished in %.2fs", time.time() - t_start)
        log(INFO, "")
        log(INFO, "Final results:")
        log(INFO, "")
        for line in io.StringIO(str(result)):
            log(INFO, "\t%s", line.strip("\n"))
        log(INFO, "")

        return result
