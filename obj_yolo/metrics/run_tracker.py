"""
Structured per-run artifacts: a self-contained `project/<run_name>/`
directory holding a reproducibility manifest, CSV metrics streams (overall,
per-class, per-IoU-threshold, weight/gradient norms, faults), and
(optionally) TensorBoard scalars + histograms + an hparams comparison tab --
shared by both the centralized (`obj_yolo/train.py`) and federated
(`obj_yolo/federated/simulation.py`) training entrypoints. A one-row-per-run
`<project>/index.csv` accumulates across every run under that project so
past research runs can be scanned/sorted without opening each one.
"""
import csv
import json
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from obj_yolo.metrics.detection_metrics import IOU_THRESHOLDS
from obj_yolo.metrics.model_summary import summarize_model

CLIENT_FIELDS = [
    "round", "client_id", "num_examples", "box_loss", "cls_loss", "dfl_loss", "duration_s",
    "total_grad_norm", "local_precision", "local_recall", "local_f1", "local_map50", "local_map",
]
CENTRAL_FIELDS = [
    "round", "precision", "recall", "f1", "map50", "map", "duration_s",
    "num_clients", "total_examples", "download_bytes", "upload_bytes", "cumulative_bytes",
]
PER_CLASS_FIELDS = ["round", "class_id", "p", "r", "f1", "ap50", "ap"]
IOU_BREAKDOWN_FIELDS = ["round"] + [f"{t:.2f}" for t in IOU_THRESHOLDS]
GRAD_STATS_FIELDS = ["round", "block", "weight_norm", "grad_norm"]
MODEL_SUMMARY_FIELDS = ["round", "total_params", "trainable_params", "nonzero_params", "sparsity", "size_mb"]
ERROR_FIELDS = ["round", "client_id", "error"]
INDEX_FIELDS = [
    "run_name", "timestamp", "run_dir", "scale", "imgsz", "lr0", "batch",
    "best_map", "total_wall_time_s", "total_communication_bytes",
]


def state_dict_bytes(state_dict: dict[str, torch.Tensor]) -> int:
    """Simulated wire size of a state_dict, for reporting FL communication cost
    even though nothing actually crosses a network in this single-process simulation."""
    return sum(t.numel() * t.element_size() for t in state_dict.values())


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def _flatten_for_hparams(d: dict[str, Any]) -> dict[str, Any]:
    """TensorBoard's add_hparams only accepts int/float/str/bool/None values --
    JSON-stringify anything else (nested dicts like config's own "_env")."""
    flat = {}
    for k, v in d.items():
        if v is None or isinstance(v, (int, float, str, bool)):
            flat[k] = v if v is not None else "None"
        else:
            flat[k] = json.dumps(v, default=str)
    return flat


class RunTracker:
    """
    Args:
        project: root directory for runs (e.g. `runs/train`, `runs/federated`).
        run_name: subdirectory name; defaults to a timestamp.
        config: full CLI/run configuration, snapshotted into `config.json`
            alongside environment info (torch version, CUDA availability,
            git commit) for reproducibility, and later used for the
            TensorBoard hparams tab and the cross-run index row.
        use_tensorboard: write scalars/histograms/hparams to `<run_dir>/tb/`
            via `torch.utils.tensorboard.SummaryWriter`.
        log_histograms: additionally write per-parameter weight/gradient
            histograms (heavier: slower + larger `tb/` event files).
    """

    def __init__(
        self,
        project: str | Path,
        run_name: Optional[str] = None,
        config: Optional[dict] = None,
        use_tensorboard: bool = True,
        log_histograms: bool = True,
    ) -> None:
        self.project = Path(project)
        run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name
        self.run_dir = self.project / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.weights_dir = self.run_dir / "weights"
        self.weights_dir.mkdir(exist_ok=True)
        self.log_histograms_enabled = log_histograms

        self.config = dict(config or {})
        self.config["_env"] = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "git_commit": _git_commit(),
            "timestamp": datetime.now().isoformat(),
        }
        (self.run_dir / "config.json").write_text(json.dumps(self.config, indent=2, default=str))

        self._init_csv("client_metrics.csv", CLIENT_FIELDS)
        self._init_csv("centralized_metrics.csv", CENTRAL_FIELDS)
        self._init_csv("per_class_metrics.csv", PER_CLASS_FIELDS)
        self._init_csv("iou_breakdown.csv", IOU_BREAKDOWN_FIELDS)
        self._init_csv("grad_stats.csv", GRAD_STATS_FIELDS)
        self._init_csv("model_summary_history.csv", MODEL_SUMMARY_FIELDS)
        self._init_csv("errors.csv", ERROR_FIELDS)

        self.tb = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb = SummaryWriter(str(self.run_dir / "tb"))
            except ImportError as e:
                logging.warning(f"tensorboard unavailable ({e}); continuing without TensorBoard logging")

        self._cumulative_bytes = 0
        self._t_start = time.time()

    # -- CSV plumbing -----------------------------------------------------
    def _init_csv(self, name: str, fields: list[str]) -> None:
        self._init_csv_at(self.run_dir / name, fields)

    @staticmethod
    def _init_csv_at(path: Path, fields: list[str]) -> None:
        if not path.exists():
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()

    def _append_csv(self, name: str, fields: list[str], row: dict) -> None:
        self._append_csv_at(self.run_dir / name, fields, row)

    @staticmethod
    def _append_csv_at(path: Path, fields: list[str], row: dict) -> None:
        with open(path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow({k: row.get(k, "") for k in fields})

    # -- model summary / sparsity ------------------------------------------
    def log_model_summary(self, model, round_num: int = 0) -> dict:
        """Writes/overwrites `model_summary.json` (architecture breakdown,
        cheap and unchanged round-to-round for plain FedAvg) and appends a
        row to `model_summary_history.csv` (sparsity is the number worth
        watching over time once a sparse algorithm is in use)."""
        summary = summarize_model(model)
        (self.run_dir / "model_summary.json").write_text(json.dumps(summary, indent=2))
        self._append_csv(
            "model_summary_history.csv",
            MODEL_SUMMARY_FIELDS,
            {
                "round": round_num,
                "total_params": summary["total_params"],
                "trainable_params": summary["trainable_params"],
                "nonzero_params": summary["nonzero_params"],
                "sparsity": summary["sparsity"],
                "size_mb": summary["size_mb"],
            },
        )
        if self.tb:
            self.tb.add_scalar("model/sparsity", summary["sparsity"], round_num)
            self.tb.add_scalar("model/total_params", summary["total_params"], round_num)
        return summary

    # -- weight / gradient statistics --------------------------------------
    def log_grad_stats(
        self,
        round_num: int,
        block_weight_norms: dict[str, float],
        block_grad_norms: Optional[dict[str, float]] = None,
        total_grad_norm: Optional[float] = None,
        tag: str = "global",
    ) -> None:
        """Per-block weight/gradient L2 norms (see `obj_yolo.metrics.grad_stats`).
        `block_grad_norms`/`total_grad_norm` are omitted for the federated
        *global* model, which is never itself backpropagated (aggregation
        isn't a gradient step) -- only weight norms make sense there."""
        block_grad_norms = block_grad_norms or {}
        for block, w_norm in block_weight_norms.items():
            g_norm = block_grad_norms.get(block, "")
            self._append_csv(
                "grad_stats.csv", GRAD_STATS_FIELDS, {"round": round_num, "block": block, "weight_norm": w_norm, "grad_norm": g_norm}
            )
            if self.tb:
                self.tb.add_scalar(f"{tag}/weight_norm/{block}", w_norm, round_num)
                if block in block_grad_norms:
                    self.tb.add_scalar(f"{tag}/grad_norm/{block}", block_grad_norms[block], round_num)
        if self.tb and total_grad_norm is not None:
            self.tb.add_scalar(f"{tag}/grad_norm/total", total_grad_norm, round_num)

    def log_weight_histograms(self, model: nn.Module, round_num: int) -> None:
        if not (self.tb and self.log_histograms_enabled):
            return
        for name, param in model.named_parameters():
            self.tb.add_histogram(f"weights/{name}", param.detach().cpu(), round_num)

    def log_grad_histograms(self, model: nn.Module, round_num: int) -> None:
        if not (self.tb and self.log_histograms_enabled):
            return
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.tb.add_histogram(f"grads/{name}", param.grad.detach().cpu(), round_num)

    # -- per-client (federated) --------------------------------------------
    def log_client_round(
        self,
        round_num: int,
        client_id: str,
        num_examples: int,
        train_metrics: dict,
        duration_s: float,
        local_eval: Optional[dict] = None,
        grad_stats: Optional[dict] = None,
    ) -> None:
        """`grad_stats`, if given: `{"block_weight_norms": {...}, "block_grad_norms":
        {...}, "total_grad_norm": float}` from `obj_yolo.metrics.grad_stats`
        -- logged as per-block TensorBoard scalars (namespaced per client) and
        a single `total_grad_norm` column in `client_metrics.csv` (full
        per-block detail in a CSV per client per round would be a lot of
        rows; TensorBoard is the right place to browse that)."""
        row = {
            "round": round_num,
            "client_id": client_id,
            "num_examples": num_examples,
            "box_loss": train_metrics.get("box_loss"),
            "cls_loss": train_metrics.get("cls_loss"),
            "dfl_loss": train_metrics.get("dfl_loss"),
            "duration_s": duration_s,
            "total_grad_norm": grad_stats.get("total_grad_norm") if grad_stats else "",
        }
        if local_eval:
            row.update(
                {
                    "local_precision": local_eval["precision"],
                    "local_recall": local_eval["recall"],
                    "local_f1": local_eval["f1"],
                    "local_map50": local_eval["map50"],
                    "local_map": local_eval["map"],
                }
            )
        self._append_csv("client_metrics.csv", CLIENT_FIELDS, row)

        if self.tb:
            for k in ("box_loss", "cls_loss", "dfl_loss"):
                if train_metrics.get(k) is not None:
                    self.tb.add_scalar(f"client/{client_id}/{k}", train_metrics[k], round_num)
            if local_eval:
                for k in ("precision", "recall", "f1", "map50", "map"):
                    self.tb.add_scalar(f"client/{client_id}/local_{k}", local_eval[k], round_num)
            if grad_stats:
                for block, w in grad_stats.get("block_weight_norms", {}).items():
                    self.tb.add_scalar(f"client/{client_id}/weight_norm/{block}", w, round_num)
                for block, g in grad_stats.get("block_grad_norms", {}).items():
                    self.tb.add_scalar(f"client/{client_id}/grad_norm/{block}", g, round_num)
                if grad_stats.get("total_grad_norm") is not None:
                    self.tb.add_scalar(f"client/{client_id}/grad_norm/total", grad_stats["total_grad_norm"], round_num)

    def log_client_error(self, round_num: int, client_id: str, error: str) -> None:
        self._append_csv("errors.csv", ERROR_FIELDS, {"round": round_num, "client_id": client_id, "error": error})

    # -- centralized / global-model evaluation -----------------------------
    def log_central_round(
        self,
        round_num: int,
        metrics: dict,
        duration_s: float,
        num_clients: int = 1,
        total_examples: int = 0,
        download_bytes: int = 0,
        upload_bytes: int = 0,
    ) -> None:
        self._cumulative_bytes += download_bytes + upload_bytes
        row = {
            "round": round_num,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "map50": metrics["map50"],
            "map": metrics["map"],
            "duration_s": duration_s,
            "num_clients": num_clients,
            "total_examples": total_examples,
            "download_bytes": download_bytes,
            "upload_bytes": upload_bytes,
            "cumulative_bytes": self._cumulative_bytes,
        }
        self._append_csv("centralized_metrics.csv", CENTRAL_FIELDS, row)

        for cls_id, stats in metrics.get("per_class", {}).items():
            self._append_csv(
                "per_class_metrics.csv",
                PER_CLASS_FIELDS,
                {"round": round_num, "class_id": cls_id, **stats},
            )
            if self.tb:
                for k, v in stats.items():
                    self.tb.add_scalar(f"central/class_{cls_id}/{k}", v, round_num)

        map_per_iou = metrics.get("map_per_iou", {})
        if map_per_iou:
            self._append_csv("iou_breakdown.csv", IOU_BREAKDOWN_FIELDS, {"round": round_num, **map_per_iou})
            if self.tb:
                for thr, v in map_per_iou.items():
                    self.tb.add_scalar(f"central/map_per_iou/{thr}", v, round_num)

        if self.tb:
            for k in ("precision", "recall", "f1", "map50", "map"):
                self.tb.add_scalar(f"central/{k}", metrics[k], round_num)
            self.tb.add_scalar("central/cumulative_bytes", self._cumulative_bytes, round_num)

    # -- hparams comparison tab --------------------------------------------
    def log_hparams(self, final_metrics: dict[str, float]) -> None:
        if not self.tb:
            return
        hparams = _flatten_for_hparams(self.config)
        metrics = {f"hparam/{k}": float(v) for k, v in final_metrics.items() if isinstance(v, (int, float))}
        self.tb.add_hparams(hparams, metrics)

    # -- run finalization / cross-run index --------------------------------
    def finalize(self, summary: dict) -> None:
        summary = dict(summary)
        summary["total_wall_time_s"] = time.time() - self._t_start
        summary["total_communication_bytes"] = self._cumulative_bytes
        (self.run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))

        self.log_hparams(summary)

        self._init_csv_at(self.project / "index.csv", INDEX_FIELDS)
        self._append_csv_at(
            self.project / "index.csv",
            INDEX_FIELDS,
            {
                "run_name": self.run_name,
                "timestamp": self.config.get("_env", {}).get("timestamp", ""),
                "run_dir": str(self.run_dir),
                "scale": self.config.get("scale", ""),
                "imgsz": self.config.get("imgsz", ""),
                "lr0": self.config.get("lr0", ""),
                "batch": self.config.get("batch", ""),
                "best_map": summary.get("best_map", ""),
                "total_wall_time_s": summary["total_wall_time_s"],
                "total_communication_bytes": summary["total_communication_bytes"],
            },
        )

        if self.tb:
            self.tb.close()
