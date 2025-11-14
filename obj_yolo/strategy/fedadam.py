"""Adaptive Federated Optimization using Adam (FedAdam) strategy.

[Reddi et al., 2020]

Paper: arxiv.org/abs/2003.00295
"""
import io
import os
import time
from logging import INFO
from pathlib import Path
from typing import Optional, Callable, Iterable

from flwr.common import (
    Scalar, 
    log,
    RecordDict,
    ConfigRecord,
    ArrayRecord,
    Message,
    MessageType,
    MetricRecord,
)
from flwr.server import Grid
from flwr.serverapp.strategy import Result
from flwr.serverapp.strategy.fedopt import FedOpt
from flwr.serverapp.strategy.strategy_utils import (
    log_strategy_start_info, 
    sample_nodes,
    aggregate_arrayrecords,
)

from ultralytics import YOLO

from obj_yolo.strategy.strategy_utils import (
    validate_message_reply_consistency,
    load_and_update_model
)

class CustomFedAdam(FedOpt):
    def __init__(self, *, fraction_train = 1, fraction_evaluate = 1, min_train_nodes = 2, min_evaluate_nodes = 2, min_available_nodes = 2, weighted_by_key = "num-examples", arrayrecord_key = "arrays", configrecord_key = "config", train_metrics_aggr_fn = None, evaluate_metrics_aggr_fn = None, eta = 0.1, eta_l = 0.1, beta_1 = 0, beta_2 = 0, tau = 0.001):
        super().__init__(fraction_train=fraction_train, fraction_evaluate=fraction_evaluate, min_train_nodes=min_train_nodes, min_evaluate_nodes=min_evaluate_nodes, min_available_nodes=min_available_nodes, weighted_by_key=weighted_by_key, arrayrecord_key=arrayrecord_key, configrecord_key=configrecord_key, train_metrics_aggr_fn=train_metrics_aggr_fn, evaluate_metrics_aggr_fn=evaluate_metrics_aggr_fn, eta=eta, eta_l=eta_l, beta_1=beta_1, beta_2=beta_2, tau=tau)
        BASE_LIB_PATH = os.path.abspath(os.path.dirname(__file__))
        BASE_DIR_PATH = os.path.dirname(BASE_LIB_PATH)
        self.model_path = Path(BASE_DIR_PATH) / "yolo_config" / "yolo11n.yaml"
        self.untrain_record_key = "untrain_arrays"
        self.untrain_arrays : dict[str, ArrayRecord] = {}
    
    def __repr__(self):
        rep = f"FedAdam merged with ultralytics, accept failures = {self.accept_failures}"
        return rep
