"""
Sequential, single-process FedAvg simulation: owns the global model, spawns
each round's client training as a plain in-process loop (one `FedClient` at
a time -- see the federated-simulation plan for why: it keeps only one
model's worth of memory resident at once, no multiprocessing/Ray), and
aggregates results with `obj_yolo.federated.strategy.fedavg_aggregate`.
"""
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from obj_yolo.federated.client import FedClient, FitConfig
from obj_yolo.federated.state_store import ClientStateStore
from obj_yolo.federated.strategy import fedavg_aggregate
from obj_yolo.model.yolov8 import Scale, YOLOv8
from obj_yolo.val import evaluate


class FedAvgSimulation:
    """
    Args:
        clients: the full simulated client pool.
        nc, scale, imgsz: model/config shared by the global model and every client.
    """

    def __init__(
        self,
        clients: list[FedClient],
        nc: int,
        scale: Scale = "n",
        imgsz: int = 640,
        device: Optional[torch.device] = None,
        seed: int = 42,
    ) -> None:
        if not clients:
            raise ValueError("FedAvgSimulation needs at least one client")
        self.clients = clients
        self.nc = nc
        self.scale = scale
        self.imgsz = imgsz
        self.device = device or torch.device("cpu")
        self.rng = random.Random(seed)

        self.global_model = YOLOv8(nc=nc, scale=scale).to(self.device)
        self.state_store = ClientStateStore()

    def _sample_clients(self, fraction_fit: float) -> list[FedClient]:
        k = max(1, round(len(self.clients) * fraction_fit))
        return self.rng.sample(self.clients, k)

    def run(
        self,
        num_rounds: int,
        local_epochs: int = 1,
        lr0: float = 0.01,
        batch: int = 8,
        fraction_fit: float = 1.0,
        val_loader: Optional[DataLoader] = None,
        out_dir: str | Path = "runs/federated",
    ) -> dict[str, float]:
        out_dir = Path(out_dir)
        weights_dir = out_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)

        fit_config = FitConfig(local_epochs=local_epochs, lr0=lr0, batch=batch)
        best_map = 0.0

        for rnd in range(1, num_rounds + 1):
            sampled = self._sample_clients(fraction_fit)
            global_state = {k: v.detach().clone() for k, v in self.global_model.state_dict().items()}

            results: list[tuple[dict[str, torch.Tensor], int]] = []
            for client in sampled:
                state_dict, num_examples, metrics = client.fit(global_state, fit_config, self.state_store)
                results.append((state_dict, num_examples))
                print(
                    f"[round {rnd}/{num_rounds}] client={client.client_id} "
                    f"n={num_examples} box={metrics['box_loss']:.4f} "
                    f"cls={metrics['cls_loss']:.4f} dfl={metrics['dfl_loss']:.4f}"
                )

            agg_state = fedavg_aggregate(results)
            self.global_model.load_state_dict(agg_state)
            torch.save(agg_state, weights_dir / "global_last.pt")

            if val_loader is not None:
                metrics = evaluate(self.global_model, val_loader, self.device)
                print(f"[round {rnd}/{num_rounds}] val mAP50-95={metrics['map']:.4f} mAP50={metrics['map_50']:.4f}")
                if metrics["map"] > best_map:
                    best_map = metrics["map"]
                    torch.save(agg_state, weights_dir / "global_best.pt")

        return {"best_map": best_map}
