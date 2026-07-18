"""
Flower message-based FedAvg strategy
"""
import os
from pathlib import Path
from typing import Optional, Iterable

from flwr.common import (
    ArrayRecord,
    Message,
    MetricRecord,
)

from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.strategy_utils import (
    aggregate_arrayrecords,
)

from obj_yolo.strategy.strategy_utils import (
    load_and_update_model
)

# pylint: disable=too-many-instance-attributes

class CustomFedAvg(FedAvg):
    """
    FedAvg algorithm that works with Ultralytics library
    
    NOTE:
    - Only works for simulation
    
    Summary:
    - Instead of loading the model using the ArrayRecords, we save the model in a folder
    and load the model form the central folder
    - Check if the model is loading correctly
    """
    def __init__(self, *, fraction_train = 1, fraction_evaluate = 1, min_train_nodes = 2, min_evaluate_nodes = 2, min_available_nodes = 2, dataset_name:str = "kitti"):
        super().__init__(fraction_train=fraction_train, fraction_evaluate=fraction_evaluate, min_train_nodes=min_train_nodes, min_evaluate_nodes=min_evaluate_nodes, min_available_nodes=min_available_nodes)
        self.base_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
        self.dataset_name = dataset_name
        self.yolo_base_path = Path(self.base_path) / "yolo_config" / f"{dataset_name}_yolo11.yaml"
    
    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message]
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Aggregate model weights using weighted average and store checkpoint."""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)
        
        arrays, metrics = None, None
        if valid_replies:
            reply_contents = [msg.content for msg in valid_replies]

            # Aggregate ArrayRecords
            arrays = aggregate_arrayrecords(
                reply_contents,
                self.weighted_by_key,
            )

            # Aggregate MetricRecords
            metrics = self.train_metrics_aggr_fn(
                reply_contents,
                self.weighted_by_key,
            )
            
            agg_model = load_and_update_model(model_path=self.yolo_base_path, aggregated_state=arrays)
            agg_model.save(Path(self.base_path) / "flwr_simulation" / f"{self.dataset_name}" / "aggregated_model" / "agg_model.pt")
        return arrays, metrics
