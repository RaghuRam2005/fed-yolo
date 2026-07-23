"""
FedAvg aggregation: a weighted average (by each client's local example
count) of client `state_dict()`s, applied uniformly to every tensor
(including BatchNorm running-mean/var/num_batches_tracked buffers) --
matches the behavior of flwr's `aggregate_arrayrecords`, which the old
(removed) `CustomFedAvg`/`FedTag` strategies relied on.
"""
import torch


def fedavg_aggregate(results: list[tuple[dict[str, torch.Tensor], int]]) -> dict[str, torch.Tensor]:
    """
    Args:
        results: list of `(client_state_dict, num_examples)` pairs, one per
            client that participated in this round.

    Returns:
        The weighted-average state_dict, with each tensor cast back to its
        original dtype (averaging is done in float regardless of the
        source tensor's dtype, e.g. for int64 BatchNorm buffers).
    """
    if not results:
        raise ValueError("fedavg_aggregate() called with no client results")

    states, weights = zip(*results)
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("total num_examples across clients must be > 0")

    agg: dict[str, torch.Tensor] = {}
    for key in states[0].keys():
        weighted_sum = sum(s[key].float() * (w / total) for s, w in zip(states, weights))
        agg[key] = weighted_sum.to(states[0][key].dtype)
    return agg
