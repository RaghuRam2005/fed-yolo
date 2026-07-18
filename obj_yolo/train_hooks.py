"""
Non-invasive Ultralytics training hooks for algorithm-specific loss terms
(e.g. FedTag's BatchNorm-gamma L1 sparsity penalty).

Ultralytics caches a model's loss function as `model.criterion`, lazily built
the first time `BaseModel.loss()` runs. Swapping that attribute inside a
custom Trainer's `get_model()` lets us add an extra loss term for the whole
training run without touching Ultralytics source (no vendoring required).
"""
import torch.nn as nn

from ultralytics.models.yolo.detect import DetectionTrainer


class _L1SparseDetectionLoss:
    """Wraps a base detection loss, adding an L1 penalty on BatchNorm gamma
    (channel-level sparsity), per FedTag/FedWeg."""

    def __init__(self, base_criterion, model: nn.Module, l1_lambda: float):
        self.base_criterion = base_criterion
        self.model = model
        self.l1_lambda = l1_lambda

    def __call__(self, preds, batch):
        loss, loss_items = self.base_criterion(preds, batch)
        l1 = sum(
            m.weight.abs().sum()
            for m in self.model.modules()
            if isinstance(m, nn.BatchNorm2d)
        )
        loss = loss + self.l1_lambda * l1
        return loss, loss_items


def make_l1_sparse_trainer(l1_lambda: float):
    """Return a DetectionTrainer subclass that adds the L1-sparsity penalty."""

    class L1SparseTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            base_criterion = model.init_criterion()
            model.criterion = _L1SparseDetectionLoss(base_criterion, model, l1_lambda)
            return model

    return L1SparseTrainer
