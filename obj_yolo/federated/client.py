"""
One simulated federated client: holds a data partition (its own
`images/`+`labels/` folders, as produced by `obj_yolo.dataset.PrepareData`)
and knows how to `fit()` a copy of the global model on that partition.

"Sending" weights to/from this client is nothing more than a Python function
call passing `state_dict()` tensors -- see the federated-simulation plan's
Context section for why that's sufficient for a single-process simulation.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from obj_yolo.data.yolo_dataset import YoloDataset, collate_fn
from obj_yolo.federated.state_store import ClientStateStore
from obj_yolo.loss.loss import DetectionLoss
from obj_yolo.model.yolov8 import Scale, YOLOv8
from obj_yolo.train import build_optimizer


@dataclass
class FitConfig:
    local_epochs: int = 1
    lr0: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 5e-4
    batch: int = 8
    workers: int = 0


class FedClient:
    """A simulated client bound to one on-disk data partition."""

    def __init__(
        self,
        client_id: str,
        images_dir: str | Path,
        labels_dir: str | Path,
        nc: int,
        scale: Scale = "n",
        imgsz: int = 640,
        device: Optional[torch.device] = None,
    ) -> None:
        self.client_id = client_id
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.nc = nc
        self.scale = scale
        self.imgsz = imgsz
        self.device = device or torch.device("cpu")

        # Built once and reused across rounds -- only its *dataset*, not its
        # weights, is client state; weights come in fresh via fit()'s argument.
        self._dataset = YoloDataset(self.images_dir, self.labels_dir, imgsz=imgsz, augment=True)

    def __len__(self) -> int:
        return len(self._dataset)

    def fit(
        self,
        global_state_dict: dict[str, torch.Tensor],
        config: FitConfig,
        state_store: Optional[ClientStateStore] = None,
    ) -> tuple[dict[str, torch.Tensor], int, dict[str, float]]:
        """Train a fresh model (seeded with `global_state_dict`) on this
        client's local data for `config.local_epochs` epochs.

        Returns the resulting `state_dict()`, the number of local training
        examples (for weighted aggregation), and a small metrics dict.
        """
        model = YOLOv8(nc=self.nc, scale=self.scale).to(self.device)
        model.load_state_dict(global_state_dict)

        # Extension point for stateful algorithms (unused by plain FedAvg):
        # a subclass could read `state_store.get(self.client_id)` here and
        # overlay it onto `model` before training (e.g. personal BN stats).
        _ = state_store.get(self.client_id) if state_store is not None else None

        loader = DataLoader(
            self._dataset,
            batch_size=config.batch,
            shuffle=True,
            num_workers=config.workers,
            collate_fn=collate_fn,
            drop_last=len(self._dataset) > config.batch,
        )
        loss_fn = DetectionLoss(nc=self.nc, stride=model.detect.stride, device=self.device)
        optimizer = build_optimizer(model, "sgd", config.lr0, config.momentum, config.weight_decay)

        model.train()
        epoch_loss = torch.zeros(3, device=self.device)
        n_batches = 0
        for _ in range(config.local_epochs):
            for batch in loader:
                imgs = batch["img"].to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                preds = model(imgs)
                total_loss, loss_items = loss_fn(preds, batch)
                total_loss.backward()
                optimizer.step()
                epoch_loss += loss_items
                n_batches += 1

        # Extension point counterpart: a stateful algorithm would persist
        # whatever it needs here, e.g. state_store.set(self.client_id, ...).

        avg_loss = (epoch_loss / max(n_batches, 1)).tolist()
        metrics = {"box_loss": avg_loss[0], "cls_loss": avg_loss[1], "dfl_loss": avg_loss[2]}
        return model.state_dict(), len(self._dataset), metrics
