"""
Flower message-based FedAvg strategy
"""
import io
import os
import time
from logging import INFO
from pathlib import Path
from typing import Optional, Callable, Iterable

from flwr.common import (
    log,
    RecordDict,
    ConfigRecord,
    ArrayRecord,
    Message,
    MessageType,
    MetricRecord,
)

from flwr.server import Grid
from flwr.serverapp.strategy import FedAvg, Result
from flwr.serverapp.strategy.strategy_utils import (
    log_strategy_start_info, 
    sample_nodes,
    aggregate_arrayrecords,
)

from obj_yolo.strategy.strategy_utils import (
    validate_message_reply_consistency,
    load_and_update_model
)

class CustomFedAvg(FedAvg):
    """
    FedAvg class that works with YOLO architecture
    """
    def __init__(self, *, fraction_train = 1, fraction_evaluate = 1, min_train_nodes = 2, min_evaluate_nodes = 2, min_available_nodes = 2):
        super().__init__(fraction_train=fraction_train, fraction_evaluate=fraction_evaluate, min_train_nodes=min_train_nodes, min_evaluate_nodes=min_evaluate_nodes, min_available_nodes=min_available_nodes)
        BASE_LIB_PATH = os.path.abspath(os.path.dirname(__file__))
        BASE_DIR_PATH = os.path.dirname(os.path.dirname(BASE_LIB_PATH))
        self.model_path = Path(BASE_DIR_PATH) / "yolo_config" / "yolo11n.yaml"
        self.untrain_record_key = "untrain_arrays"
        self.untrain_arrays : dict[str, ArrayRecord] = {}
    
    def __repr__(self):
        rep = f"FedAveraging merged with ultralytics, accept failures = {self.accept_failures}"
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

        # Construct messages
        record = RecordDict(
            {self.arrayrecord_key: arrays, self.configrecord_key: config}
        )
        return self._construct_messages(record, node_ids, MessageType.TRAIN)

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message]
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Aggregate model weights using weighted average and store checkpoint."""
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
            aggregated_model_path = Path(Path.cwd() / 'flwr_simulation' / 'aggregated_model')
            if not aggregated_model_path.exists():
                aggregated_model_path.mkdir(parents=True, exist_ok=True)
            net.save(str(aggregated_model_path / "agg_model.pt"))
            full_parameters = {k:val.detach() for k, val in net.model.state_dict().items() \
                               if not k.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}
            return ArrayRecord(full_parameters), aggregated_metrics
            
        return aggregated_parameters, aggregated_metrics

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
