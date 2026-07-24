"""
Per-block weight/gradient L2-norm statistics -- standard research signals
for diagnosing convergence, exploding/vanishing gradients, and (in the
federated setting) how much a client's local update actually moved its
weights.

Block granularity matches `obj_yolo.metrics.model_summary`'s breakdown
(`b0..b9` backbone, `n12/n15/n16/n18/n19/n21` neck, `detect` head) rather
than per-parameter: hundreds of near-identical per-tensor scalars is noise,
not signal, at the "what's happening to my model" level this is meant for.
"""
import torch
import torch.nn as nn

from obj_yolo.model.yolov8 import YOLOv8

_BACKBONE_BLOCKS = ("b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b9")
_NECK_BLOCKS = ("n12", "n15", "n16", "n18", "n19", "n21")


def _named_blocks(model: YOLOv8) -> dict[str, nn.Module]:
    blocks = {name: getattr(model, name) for name in _BACKBONE_BLOCKS}
    blocks.update({name: getattr(model, name) for name in _NECK_BLOCKS})
    blocks["detect"] = model.detect
    return blocks


def block_weight_norms(model: YOLOv8) -> dict[str, float]:
    """L2 norm of each block's parameters."""
    return {
        name: float(torch.sqrt(sum((p.detach() ** 2).sum() for p in block.parameters())))
        for name, block in _named_blocks(model).items()
    }


def block_grad_norms(model: YOLOv8) -> dict[str, float]:
    """L2 norm of each block's `.grad` (0.0 for a block with no grad yet, e.g. before the first backward())."""
    norms = {}
    for name, block in _named_blocks(model).items():
        grads = [p.grad for p in block.parameters() if p.grad is not None]
        norms[name] = float(torch.sqrt(sum((g.detach() ** 2).sum() for g in grads))) if grads else 0.0
    return norms


def total_grad_norm(model: YOLOv8) -> float:
    """Global L2 norm across all parameters' grads -- the standard single-number
    "is training blowing up" signal (same quantity `clip_grad_norm_` would return)."""
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    return float(torch.sqrt(sum((g.detach() ** 2).sum() for g in grads)))
