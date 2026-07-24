"""
Sequential, single-process FedAvg simulation: owns the global model, spawns
each round's client training as a plain in-process loop (one `FedClient` at
a time -- see the federated-simulation plan for why: it keeps only one
model's worth of memory resident at once, no multiprocessing/Ray), and
aggregates results with `obj_yolo.federated.strategy.fedavg_aggregate`.

Each round is fault-isolated per client: a client that raises during `fit()`
is logged and excluded from that round's aggregation rather than crashing
the whole simulation. If every sampled client fails, the round is skipped
(previous global weights carry over) rather than aggregating nothing.
"""
import logging
import random
import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from obj_yolo.federated.client import FedClient, FitConfig
from obj_yolo.federated.state_store import ClientStateStore
from obj_yolo.federated.strategy import fedavg_aggregate
from obj_yolo.metrics.grad_stats import block_weight_norms
from obj_yolo.metrics.run_tracker import RunTracker, state_dict_bytes
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
        """Sample `round(len(clients) * fraction_fit)` clients, clamped to
        `[1, len(clients)]` (Ultralytics-style hyperparameter clamping,
        rather than rejecting out-of-range values outright)."""
        n = len(self.clients)
        clamped = max(1e-6, min(1.0, fraction_fit))
        if clamped != fraction_fit:
            logging.warning(f"fraction_fit={fraction_fit} out of (0, 1]; clamped to {clamped}")
        k = max(1, min(n, round(n * clamped)))
        return self.rng.sample(self.clients, k)

    def run(
        self,
        num_rounds: int,
        fit_config: FitConfig,
        fraction_fit: float = 1.0,
        val_loader: Optional[DataLoader] = None,
        out_dir: str | Path = "runs/federated",
        tracker: Optional[RunTracker] = None,
    ) -> dict[str, float]:
        tracker = tracker or RunTracker(out_dir, config={"num_rounds": num_rounds})

        best_map = 0.0
        successful_rounds = 0
        tracker.log_model_summary(self.global_model, round_num=0)

        for rnd in range(1, num_rounds + 1):
            sampled = self._sample_clients(fraction_fit)
            global_state = {k: v.detach().clone() for k, v in self.global_model.state_dict().items()}
            download_bytes = state_dict_bytes(global_state) * len(sampled)

            results: list[tuple[dict[str, torch.Tensor], int]] = []
            upload_bytes = 0
            for client in sampled:
                t0 = time.time()
                try:
                    state_dict, num_examples, metrics, grad_stats = client.fit(
                        global_state, fit_config, self.state_store
                    )
                except Exception as e:
                    logging.exception(f"[round {rnd}] client {client.client_id} failed, skipping")
                    tracker.log_client_error(rnd, client.client_id, repr(e))
                    continue
                duration = time.time() - t0
                upload_bytes += state_dict_bytes(state_dict)

                local_eval = None
                try:
                    local_eval = client.evaluate_local(
                        state_dict, batch=fit_config.batch,
                        conf_thres=fit_config.eval_conf_thres, iou_thres=fit_config.eval_iou_thres,
                    )
                except Exception as e:
                    logging.exception(f"[round {rnd}] client {client.client_id} local eval failed")
                    tracker.log_client_error(rnd, client.client_id, f"local eval: {e!r}")

                results.append((state_dict, num_examples))
                tracker.log_client_round(rnd, client.client_id, num_examples, metrics, duration, local_eval, grad_stats)
                print(
                    f"[round {rnd}/{num_rounds}] client={client.client_id} "
                    f"n={num_examples} box={metrics['box_loss']:.6f} "
                    f"cls={metrics['cls_loss']:.6f} dfl={metrics['dfl_loss']:.6f} "
                    f"grad_norm={grad_stats['total_grad_norm']:.6f}"
                )

            if not results:
                logging.warning(f"[round {rnd}] all sampled clients failed; keeping previous global weights")
                continue

            successful_rounds += 1
            agg_state = fedavg_aggregate(results)
            self.global_model.load_state_dict(agg_state)
            torch.save(agg_state, tracker.weights_dir / "global_last.pt")
            tracker.log_model_summary(self.global_model, round_num=rnd)

            # Global model has no gradients of its own (FedAvg aggregation
            # isn't a backprop step) -- weight norms + histograms only.
            tracker.log_grad_stats(rnd, block_weight_norms(self.global_model), tag="global")
            tracker.log_weight_histograms(self.global_model, rnd)

            if val_loader is not None:
                t0 = time.time()
                metrics = evaluate(
                    self.global_model, val_loader, self.device,
                    conf_thres=fit_config.eval_conf_thres, iou_thres=fit_config.eval_iou_thres,
                )
                duration = time.time() - t0
                total_examples = sum(n for _, n in results)
                tracker.log_central_round(
                    rnd, metrics, duration, len(results), total_examples, download_bytes, upload_bytes
                )
                print(
                    f"[round {rnd}/{num_rounds}] val P={metrics['precision']:.6f} R={metrics['recall']:.6f} "
                    f"F1={metrics['f1']:.6f} mAP50={metrics['map50']:.6f} mAP50-95={metrics['map']:.6f}"
                )
                if metrics["map"] > best_map:
                    best_map = metrics["map"]
                    torch.save(agg_state, tracker.weights_dir / "global_best.pt")

        result = {"best_map": best_map, "successful_rounds": successful_rounds, "num_rounds": num_rounds}
        tracker.finalize(result)

        if successful_rounds == 0:
            raise RuntimeError(
                f"federated simulation completed 0/{num_rounds} successful rounds -- "
                f"every sampled client failed in every round; see {tracker.run_dir / 'errors.csv'}"
            )
        return result
