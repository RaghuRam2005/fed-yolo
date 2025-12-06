from pathlib import Path

from flwr.common import RecordDict, ArrayRecord
from flwr.serverapp.exception import InconsistentMessageReplies

from ultralytics import YOLO

def validate_message_reply_consistency(
    replies: list[RecordDict], weighted_by_key:str, check_arrayrecord:bool
) -> None:
    """Validate that replies contain exactly one ArrayRecord and one MetricRecord, and
    that the MetricRecord includes a weight factor key.

    These checks ensure that Message-based strategies behave consistently with
    *Ins/*Res-based strategies.
    """
    # Checking for ArrayRecord consistency
    if check_arrayrecord:
        if any(len(msg.array_records) != 2 for msg in replies):
            raise InconsistentMessageReplies(
                reason="Expected exactly two ArrayRecord in replies. "
                "Skipping aggregation."
            )

        # Ensure all key are present in all ArrayRecords
        record_key = next(iter(replies[0].array_records.keys()))
        all_keys = set(replies[0][record_key].keys())
        if any(set(msg.get(record_key, {}).keys()) != all_keys for msg in replies[1:]):
            raise InconsistentMessageReplies(
                reason="All ArrayRecords must have the same keys for aggregation. "
                "This condition wasn't met. Skipping aggregation."
            )

    # Checking for MetricRecord consistency
    if any(len(msg.metric_records) != 1 for msg in replies):
        raise InconsistentMessageReplies(
            reason="Expected exactly one MetricRecord in replies, but found more. "
            "Skipping aggregation."
        )

    # Ensure all key are present in all MetricRecords
    record_key = next(iter(replies[0].metric_records.keys()))
    all_keys = set(replies[0][record_key].keys())
    if any(set(msg.get(record_key, {}).keys()) != all_keys for msg in replies[1:]):
        raise InconsistentMessageReplies(
            reason="All MetricRecords must have the same keys for aggregation. "
            "This condition wasn't met. Skipping aggregation."
        )

    # Verify the weight factor key presence in all MetricRecords
    if weighted_by_key not in all_keys:
        raise InconsistentMessageReplies(
            reason=f"Missing required key `{weighted_by_key}` in the MetricRecord of "
            "reply messages. Cannot average ArrayRecords and MetricRecords. Skipping "
            "aggregation."
        )

    # Check that it is not a list
    if any(isinstance(msg[record_key][weighted_by_key], list) for msg in replies):
        raise InconsistentMessageReplies(
            reason=f"Key `{weighted_by_key}` in the MetricRecord of reply messages "
            "must be a single value (int or float), but a list was found. Skipping "
            "aggregation."
        )

def load_and_update_model(model_path:Path, aggregated_state:ArrayRecord, untrain_parameters:ArrayRecord) -> YOLO:
    net = YOLO(model_path)
    state_dict = net.model.state_dict().copy()
    state_dict.update(aggregated_state.to_torch_state_dict())
    state_dict.update(untrain_parameters.to_torch_state_dict())
    net.model.load_state_dict(state_dict)
    return net
