"""Model parameter-count / sparsity summaries, reported once per run
(architecture breakdown) and cheaply recomputable every round (sparsity --
0% for plain FedAvg today, but this is exactly the number a future sparse
algorithm like the old FedTag's BatchNorm-gamma L1 penalty would move)."""
import torch.nn as nn

from obj_yolo.model.yolov8 import YOLOv8


def _param_counts(module: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def summarize_model(model: YOLOv8) -> dict:
    total, trainable = _param_counts(model)
    nonzero = sum(int((p != 0).sum().item()) for p in model.parameters())

    backbone = [model.b0, model.b1, model.b2, model.b3, model.b4, model.b5, model.b6, model.b7, model.b8, model.b9]
    neck = [model.n12, model.n15, model.n16, model.n18, model.n19, model.n21]
    breakdown = {
        "backbone": sum(_param_counts(m)[0] for m in backbone),
        "neck": sum(_param_counts(m)[0] for m in neck),
        "detect": _param_counts(model.detect)[0],
    }

    return {
        "scale": model.scale,
        "nc": model.nc,
        "total_params": total,
        "trainable_params": trainable,
        "size_mb": total * 4 / (1024**2),  # float32 assumption
        "nonzero_params": nonzero,
        "sparsity": 1.0 - (nonzero / total) if total else 0.0,
        "breakdown": breakdown,
    }
