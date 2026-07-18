"""
FedTag strategy: weather-tag-conditioned sparse federated learning (FedWeg).

Each BDD100K client has a persistent weather tag (assigned during dataset
prep — see obj_yolo.dataset.PrepareData._assign_tags_to_clients). Local
training adds an L1 penalty on BatchNorm gamma factors (channel-level
sparsity, see obj_yolo.train_hooks), with the penalty strength (lambda) for
each tag adjusted every round based on that tag's mAP convergence:
  - converged, top-half tags -> lambda increases (more sparsity)
  - bottom-half tags         -> lambda decreases (less sparsity)

Aggregation is standard FedAvg over the full state dict. Each client keeps
its own BatchNorm running stats locally (obj_yolo.client's personal_bn.pt)
instead of having them aggregated — since the client re-applies its own BN
stats immediately after loading the shared aggregated model every round,
whatever ends up in the shared checkpoint's BN buffers is irrelevant, so no
separate wire transfer or server-side trainable/untrainable split is
needed.

A node's tag is determined client-side (from its partition-id, via
client_tags.json) and reported back in every reply, rather than guessed
server-side — flwr's `node_id` is an opaque per-run identifier and does not
match the small sequential partition-id used when the dataset was prepared,
so the server learns each node's tag organically from replies instead of
trying to pre-map node_id -> tag.
"""
import os
from pathlib import Path
from typing import Optional, Iterable

from flwr.common import (
    ArrayRecord,
    ConfigRecord,
    Message,
    MessageType,
    MetricRecord,
    RecordDict,
)
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.strategy_utils import (
    aggregate_arrayrecords,
    sample_nodes,
)

from obj_yolo.strategy.strategy_utils import load_and_update_model


class FedTag(FedAvg):
    """FedAvg variant with per-weather-tag L1 sparsity scheduling (FedWeg)."""

    def __init__(
        self,
        *,
        fraction_train: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_train_nodes: int = 2,
        min_evaluate_nodes: int = 2,
        min_available_nodes: int = 2,
        dataset_name: str = "bdd100k",
        l1_init: float = 0.2,
        l1_min: float = 0.2,
        l1_max: float = 0.8,
        l1_step: float = 0.02,
        convergence_window: int = 5,
        convergence_threshold: float = 0.001,
    ):
        super().__init__(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
            min_train_nodes=min_train_nodes,
            min_evaluate_nodes=min_evaluate_nodes,
            min_available_nodes=min_available_nodes,
        )
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
        self.dataset_name = dataset_name
        self.yolo_base_path = Path(self.base_path) / "yolo_config" / f"{dataset_name}_yolo11.yaml"

        # node_id -> tag, learned from replies (see module docstring)
        self._node_tag: dict[int, str] = {}

        self.l1_init = l1_init
        self.l1_min = l1_min
        self.l1_max = l1_max
        self.l1_step = l1_step
        self.tag_l1: dict[str, float] = {}

        self.tag_prev_map: dict[str, float] = {}
        self.tag_curr_map: dict[str, float] = {}
        self.tag_conv_count: dict[str, int] = {}
        self.convergence_window = convergence_window
        self.convergence_threshold = convergence_threshold

    def _lambda_for_node(self, node_id: int) -> float:
        tag = self._node_tag.get(node_id)
        if tag is None:
            return self.l1_init
        if tag not in self.tag_l1:
            self.tag_l1[tag] = self.l1_init
        return self.tag_l1[tag]

    def _sample_nodes(self, grid: Grid, fraction: float, min_nodes: int) -> list[int]:
        available = list(grid.get_node_ids())
        sample_size = max(int(len(available) * fraction), min_nodes)
        node_ids, _ = sample_nodes(grid, self.min_available_nodes, sample_size)
        return node_ids

    def _build_messages(
        self,
        arrays: ArrayRecord,
        config: ConfigRecord,
        node_ids: list[int],
        message_type: str,
    ) -> list[Message]:
        messages = []
        for node_id in node_ids:
            content = RecordDict({
                "arrays": arrays,
                "config": config,
                "tags": ConfigRecord({"l1-lambda": self._lambda_for_node(node_id)}),
            })
            messages.append(Message(content=content, message_type=message_type, dst_node_id=node_id))
        return messages

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> list[Message]:
        if self.fraction_train == 0.0:
            return []
        node_ids = self._sample_nodes(grid, self.fraction_train, self.min_train_nodes)
        config["server-round"] = server_round
        return self._build_messages(arrays, config, node_ids, MessageType.TRAIN)

    def configure_evaluate(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> list[Message]:
        if self.fraction_evaluate == 0.0:
            return []
        node_ids = self._sample_nodes(grid, self.fraction_evaluate, self.min_evaluate_nodes)
        config["server-round"] = server_round
        return self._build_messages(arrays, config, node_ids, MessageType.EVALUATE)

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)
        if not valid_replies:
            return None, None

        for msg in valid_replies:
            if "tags" in msg.content and "tag" in msg.content["tags"]:
                self._node_tag[msg.metadata.src_node_id] = str(msg.content["tags"]["tag"])

        reply_contents = [msg.content for msg in valid_replies]
        arrays = aggregate_arrayrecords(reply_contents, self.weighted_by_key)
        metrics = self.train_metrics_aggr_fn(reply_contents, self.weighted_by_key)

        if arrays is not None:
            agg_model = load_and_update_model(model_path=self.yolo_base_path, aggregated_state=arrays)
            agg_model.save(Path(self.base_path) / "flwr_simulation" / f"{self.dataset_name}" / "aggregated_model" / "agg_model.pt")

        return arrays, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> Optional[MetricRecord]:
        valid_replies, _ = self._check_and_log_replies(replies, is_train=False)
        if not valid_replies:
            return None

        tag_maps: dict[str, list[float]] = {}
        for msg in valid_replies:
            if "tags" in msg.content and "tag" in msg.content["tags"]:
                tag = str(msg.content["tags"]["tag"])
                self._node_tag[msg.metadata.src_node_id] = tag
                tag_maps.setdefault(tag, []).append(float(msg.content["metrics"]["eval-map"]))

        self.tag_prev_map = self.tag_curr_map.copy()
        self.tag_curr_map = {tag: sum(vals) / len(vals) for tag, vals in tag_maps.items()}

        # Convergence: consecutive rounds where |delta mAP| <= threshold
        for tag, curr in self.tag_curr_map.items():
            prev = self.tag_prev_map.get(tag)
            if prev is not None and abs(curr - prev) <= self.convergence_threshold:
                self.tag_conv_count[tag] = self.tag_conv_count.get(tag, 0) + 1
            else:
                self.tag_conv_count[tag] = 0

        # Rank tags by mAP, adjust lambda per FedWeg Algorithm 1
        ranked_tags = sorted(self.tag_curr_map, key=self.tag_curr_map.get, reverse=True)
        n = len(ranked_tags)
        for rank, tag in enumerate(ranked_tags):
            old_lambda = self.tag_l1.get(tag, self.l1_init)
            converged = self.tag_conv_count.get(tag, 0) >= self.convergence_window
            top_half = rank < n / 2

            if converged and top_half:
                new_lambda = old_lambda + self.l1_step
            elif not top_half:
                new_lambda = old_lambda - self.l1_step
            else:
                new_lambda = old_lambda

            self.tag_l1[tag] = float(min(self.l1_max, max(self.l1_min, new_lambda)))

        reply_contents = [msg.content for msg in valid_replies]
        return self.evaluate_metrics_aggr_fn(reply_contents, self.weighted_by_key)
