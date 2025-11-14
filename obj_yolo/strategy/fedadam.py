"""Adaptive Federated Optimization using Adam (FedAdam) strategy.

[Reddi et al., 2020]

Paper: arxiv.org/abs/2003.00295
"""
import io
import os
import time
import numpy as np
from logging import INFO
from pathlib import Path
from typing import Optional, Callable, Iterable, OrderedDict

import torch

from flwr.common import (
    Scalar, 
    log,
    RecordDict,
    ConfigRecord,
    ArrayRecord,
    Message,
    MessageType,
    MetricRecord,
    Array
)
from flwr.server import Grid
from flwr.serverapp.strategy import Result, FedAdam
from flwr.serverapp.exception import AggregationError
from flwr.serverapp.strategy.strategy_utils import (
    log_strategy_start_info, 
    sample_nodes,
    aggregate_arrayrecords,
)

from obj_yolo.strategy.strategy_utils import (
    validate_message_reply_consistency,
    load_and_update_model
)

class CustomFedAdam(FedAdam):
    def __init__(self, *, fraction_train = 1, fraction_evaluate = 1, min_train_nodes = 2, min_evaluate_nodes = 2, min_available_nodes = 2, weighted_by_key = "num-examples", arrayrecord_key = "arrays", configrecord_key = "config", train_metrics_aggr_fn = None, evaluate_metrics_aggr_fn = None, eta = 0.1, eta_l = 0.1, beta_1 = 0.9, beta_2 = 0.99, tau = 0.001):
        super().__init__(fraction_train=fraction_train, fraction_evaluate=fraction_evaluate, min_train_nodes=min_train_nodes, min_evaluate_nodes=min_evaluate_nodes, min_available_nodes=min_available_nodes, weighted_by_key=weighted_by_key, arrayrecord_key=arrayrecord_key, configrecord_key=configrecord_key, train_metrics_aggr_fn=train_metrics_aggr_fn, evaluate_metrics_aggr_fn=evaluate_metrics_aggr_fn, eta=eta, eta_l=eta_l, beta_1=beta_1, beta_2=beta_2, tau=tau)
        BASE_LIB_PATH = os.path.abspath(os.path.dirname(__file__))
        BASE_DIR_PATH = os.path.dirname(BASE_LIB_PATH)
        self.model_path = Path(BASE_DIR_PATH) / "yolo_config" / "yolo11n.yaml"
        self.untrain_record_key = "untrain_arrays"
        self.untrain_arrays : dict[str, ArrayRecord] = {}
    
    def __repr__(self):
        rep = f"FedAdam merged with ultralytics, accept failures = {self.accept_failures}"
        return rep
    
    def _construct_messages(
            self, 
            record:RecordDict,
            node_ids:list[int],
            message_type:str
    ) -> Iterable[Message]:
        messages = []
        for node_id in node_ids:
            record[self.untrain_record_key] = self.untrain_arrays.get(
                node_id,
                self.untrain_arrays.get(0, None)
            )
            message = Message(
                content=record,
                message_type=message_type,
                dst_node_id=node_id,
            )
            messages.append(message)
        return messages

    def _check_and_log_replies(
        self, replies: Iterable[Message], is_train: bool, validate: bool = True
    ) -> tuple[list[Message], list[Message]]:
        """Check replies for errors and log them.

        Parameters
        ----------
        replies : Iterable[Message]
            Iterable of reply Messages.
        is_train : bool
            Set to True if the replies are from a training round; False otherwise.
            This impacts logging and validation behavior.
        validate : bool (default: True)
            Whether to validate the reply contents for consistency.

        Returns
        -------
        tuple[list[Message], list[Message]]
            A tuple containing two lists:
            - Messages with valid contents.
            - Messages with errors.
        """
        if not replies:
            return [], []

        # Filter messages that carry content
        valid_replies: list[Message] = []
        error_replies: list[Message] = []
        for msg in replies:
            if msg.has_error():
                error_replies.append(msg)
            else:
                valid_replies.append(msg)

        log(
            INFO,
            "%s: Received %s results and %s failures",
            "aggregate_train" if is_train else "aggregate_evaluate",
            len(valid_replies),
            len(error_replies),
        )

        # Log errors
        for msg in error_replies:
            log(
                INFO,
                "\t> Received error in reply from node %d: %s",
                msg.metadata.src_node_id,
                msg.error.reason,
            )

        # Ensure expected ArrayRecords and MetricRecords are received
        if validate and valid_replies:
            validate_message_reply_consistency(
                replies=[msg.content for msg in valid_replies],
                weighted_by_key=self.weighted_by_key,
                check_arrayrecord=is_train,
            )

        return valid_replies, error_replies
    
    def configure_train(
            self,
            server_round:int, 
            arrays: ArrayRecord,
            config: ConfigRecord,
            grid:Grid,
        ) -> Iterable[Message]:
        if self.fraction_train == 0.0:
            return []
        # Sample nodes
        num_nodes = int(len(list(grid.get_node_ids())) * self.fraction_train)
        sample_size = max(num_nodes, self.min_train_nodes)
        node_ids, num_total = sample_nodes(grid, self.min_available_nodes, sample_size)
        log(
            INFO,
            "configure_train: Sampled %s nodes (out of %s)",
            len(node_ids),
            len(num_total),
        )
        # Always inject current server round
        config["server-round"] = server_round

        # Set Current Arrays
        self.current_arrays = {k: array.numpy() for k, array in arrays.items()}

        # Construct messages
        record = RecordDict(
            {self.arrayrecord_key: arrays, self.configrecord_key: config}
        )
        return self._construct_messages(record, node_ids, MessageType.TRAIN)
    
    def aggregate_train(
            self, server_round:int, replies:Iterable[Message]
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Aggregate ArrayRecords and MetricRecords in the received Messages."""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)

        train_records = []

        for msg in valid_replies:
            train_array = msg.content[self.arrayrecord_key]
            metric_array = msg.content['metrics']
            nid = msg.metadata.src_node_id
            client_untrain = msg.content[self.untrain_record_key]
            self.untrain_arrays[nid] = client_untrain

            record = RecordDict(
                {self.arrayrecord_key: train_array, 'metrics':metric_array}
            )

            train_records.append(record)
        
        aggregated_parameters = aggregate_arrayrecords(
            train_records,
            self.weighted_by_key,
        )

        metric_records = [msg.content for msg in valid_replies]

        aggregated_metrics = self.train_metrics_aggr_fn(
            metric_records,
            self.weighted_by_key,
        )

        if aggregated_parameters is not None:
            net = load_and_update_model(self.model_path, aggregated_parameters)            
            aggregated_records = {k:val.detach() for k, val in net.model.state_dict().items() \
                               if not k.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}
        else:
            return aggregated_parameters, aggregated_metrics
        
        if self.current_arrays is None:
            reason = (
                "Current arrays not set. Ensure that `configure_train` has been "
                "called before aggregation."
            )
            raise AggregationError(reason=reason)
        
        # Compute intermediate variables
        delta_t, m_t, aggregated_ndarrays = self._compute_deltat_and_mt(
            aggregated_records
        )

        # v_t
        if not self.v_t:
            self.v_t = {k: np.zeros_like(v) for k, v in aggregated_ndarrays.items()}
        self.v_t = {
            k: self.beta_2 * v + (1 - self.beta_2) * (delta_t[k] ** 2)
            for k, v in self.v_t.items()
        }

        # Compute the bias-corrected learning rate, `eta_norm` for improving convergence
        # in the early rounds of FL training. This `eta_norm` is `\alpha_t` in Kingma &
        # Ba, 2014 (http://arxiv.org/abs/1412.6980) "Adam: A Method for Stochastic
        # Optimization" in the formula line right before Section 2.1.
        eta_norm = (
            self.eta
            * np.sqrt(1 - np.power(self.beta_2, server_round + 1.0))
            / (1 - np.power(self.beta_1, server_round + 1.0))
        )

        new_arrays = {
            k: x + eta_norm * m_t[k] / (np.sqrt(self.v_t[k]) + self.tau)
            for k, x in self.current_arrays.items()
        }

        return (
            ArrayRecord(OrderedDict({k: Array(v) for k, v in new_arrays.items()})),
            aggregated_metrics,
        )
    
    def configure_evaluate(
            self,
            server_round:int,
            arrays:ArrayRecord,
            config:ConfigRecord,
            grid:Grid,
    ) -> Iterable[Message]:
        if self.fraction_evaluate == 0.0:
            return []

        # Sample nodes
        num_nodes = int(len(list(grid.get_node_ids())) * self.fraction_evaluate)
        sample_size = max(num_nodes, self.min_evaluate_nodes)
        node_ids, num_total = sample_nodes(grid, self.min_available_nodes, sample_size)
        log(
            INFO,
            "configure_evaluate: Sampled %s nodes (out of %s)",
            len(node_ids),
            len(num_total),
        )

        # Always inject current server round
        config["server-round"] = server_round

        # Construct messages
        record = RecordDict(
            {self.arrayrecord_key: arrays, self.configrecord_key: config}
        )
        return self._construct_messages(record, node_ids, MessageType.EVALUATE)

    def start(
            self,
            grid:Grid,
            initial_arrays:ArrayRecord,
            untrainable_parameters:ArrayRecord,
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
        self.untrain_arrays[0] = untrainable_parameters

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
