"""
Federated (simulated) YOLOv8 training entrypoint -- basic FedAvg over a
sequential, single-process client simulation (see `obj_yolo/federated/`).

Expects the `<data-root>/client_<i>/{images,labels}/{train,val}` layout that
`obj_yolo.dataset.PrepareData` already produces (call it with
`clientCount=N` to generate it):

    uv run python -m obj_yolo.train_federated --data-root dataset/clients --num-clients 5 --nc 13 --config configs/federated.yaml
    uv run python -m obj_yolo.train_federated \\
        --data-root dataset/clients --num-clients 5 --nc 13 \\
        --rounds 20 --local-epochs 2 --scale n --imgsz 640 --batch 8 \\
        --val-images dataset/clients/client_0/images/val \\
        --val-labels dataset/clients/client_0/labels/val

Every hyperparameter has a single source of truth in
`obj_yolo.config.FEDERATED_DEFAULTS`; `--config <file.yaml>` overrides it,
explicit CLI flags override the file (see `obj_yolo/config.py`).
"""
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from obj_yolo.config import FEDERATED_DEFAULTS, build_aug_hyp, load_config
from obj_yolo.config_validation import validate_client_root, validate_imgsz, validate_positive, validate_val_pair
from obj_yolo.data.yolo_dataset import YoloDataset, collate_fn
from obj_yolo.federated.client import FedClient, FitConfig
from obj_yolo.federated.simulation import FedAvgSimulation
from obj_yolo.metrics.run_tracker import RunTracker


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simulated federated YOLOv8 training (basic FedAvg).")
    S = argparse.SUPPRESS
    p.add_argument("--data-root", default=S, help="root containing client_<i>/ folders; may also be a 'data_root' key in --config")
    p.add_argument("--num-clients", type=int, default=S, help="may also be a 'num_clients' key in --config")
    p.add_argument("--nc", type=int, default=S, help="number of classes; may also be an 'nc' key in --config")
    p.add_argument("--config", default=None, help="YAML file of defaults; CLI flags override it")
    p.add_argument("--rounds", type=int, default=S)
    p.add_argument("--local-epochs", type=int, default=S)
    p.add_argument("--fraction-fit", type=float, default=S)
    p.add_argument("--scale", choices=["n", "s", "m", "l", "x"], default=S)
    p.add_argument("--imgsz", type=int, default=S)
    p.add_argument("--batch", type=int, default=S)
    p.add_argument("--lr0", type=float, default=S)
    p.add_argument("--momentum", type=float, default=S)
    p.add_argument("--weight-decay", type=float, default=S)
    p.add_argument("--box-gain", type=float, default=S)
    p.add_argument("--cls-gain", type=float, default=S)
    p.add_argument("--dfl-gain", type=float, default=S)
    p.add_argument("--tal-topk", type=int, default=S)
    p.add_argument("--tal-alpha", type=float, default=S)
    p.add_argument("--tal-beta", type=float, default=S)
    p.add_argument("--eval-conf-thres", type=float, default=S)
    p.add_argument("--eval-iou-thres", type=float, default=S)
    p.add_argument("--hsv-h", type=float, default=S)
    p.add_argument("--hsv-s", type=float, default=S)
    p.add_argument("--hsv-v", type=float, default=S)
    p.add_argument("--degrees", type=float, default=S)
    p.add_argument("--translate", type=float, default=S)
    p.add_argument("--aug-scale", type=float, default=S, help="affine-transform scale gain (not model scale)")
    p.add_argument("--shear", type=float, default=S)
    p.add_argument("--fliplr", type=float, default=S)
    p.add_argument("--mosaic", type=float, default=S)
    p.add_argument("--val-images", default=S, help="optional central held-out val images dir")
    p.add_argument("--val-labels", default=S, help="optional central held-out val labels dir")
    p.add_argument("--val-batch", type=int, default=S)
    p.add_argument("--project", default=S)
    p.add_argument("--run-name", default=S)
    p.add_argument("--tensorboard", dest="tensorboard", action="store_true", default=S)
    p.add_argument("--no-tensorboard", dest="tensorboard", action="store_false", default=S)
    p.add_argument("--log-histograms", dest="log_histograms", action="store_true", default=S)
    p.add_argument("--no-log-histograms", dest="log_histograms", action="store_false", default=S)
    p.add_argument("--device", default=S)
    p.add_argument("--seed", type=int, default=S)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args, FEDERATED_DEFAULTS)

    try:
        for required in ("data_root", "num_clients", "nc"):
            if required not in cfg:
                raise ValueError(f"--{required.replace('_', '-')} must be set (CLI flag or '{required}' key in --config)")
        validate_imgsz(cfg["imgsz"])
        validate_positive(cfg["rounds"], "--rounds")
        validate_val_pair(cfg.get("val_images"), cfg.get("val_labels"))
        validate_client_root(cfg["data_root"], cfg["num_clients"])
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    device = torch.device(cfg["device"])
    data_root = Path(cfg["data_root"])
    tracker = RunTracker(
        cfg["project"], run_name=cfg["run_name"], config=cfg,
        use_tensorboard=cfg["tensorboard"], log_histograms=cfg["log_histograms"],
    )

    aug_hyp = build_aug_hyp(cfg)
    clients = []
    for i in range(cfg["num_clients"]):
        client_dir = data_root / f"client_{i}"
        clients.append(
            FedClient(
                client_id=f"client_{i}",
                images_dir=client_dir / "images" / "train",
                labels_dir=client_dir / "labels" / "train",
                nc=cfg["nc"],
                scale=cfg["scale"],
                imgsz=cfg["imgsz"],
                device=device,
                hyp=aug_hyp,
            )
        )

    val_loader = None
    if cfg.get("val_images") and cfg.get("val_labels"):
        val_ds = YoloDataset(cfg["val_images"], cfg["val_labels"], imgsz=cfg["imgsz"], augment=False)
        val_loader = DataLoader(val_ds, batch_size=cfg["val_batch"], shuffle=False, collate_fn=collate_fn)

    fit_config = FitConfig(
        local_epochs=cfg["local_epochs"], lr0=cfg["lr0"], momentum=cfg["momentum"], weight_decay=cfg["weight_decay"],
        batch=cfg["batch"],
        box_gain=cfg["box_gain"], cls_gain=cfg["cls_gain"], dfl_gain=cfg["dfl_gain"],
        tal_topk=cfg["tal_topk"], tal_alpha=cfg["tal_alpha"], tal_beta=cfg["tal_beta"],
        eval_conf_thres=cfg["eval_conf_thres"], eval_iou_thres=cfg["eval_iou_thres"],
    )

    sim = FedAvgSimulation(clients, nc=cfg["nc"], scale=cfg["scale"], imgsz=cfg["imgsz"], device=device, seed=cfg["seed"])
    try:
        result = sim.run(
            num_rounds=cfg["rounds"],
            fit_config=fit_config,
            fraction_fit=cfg["fraction_fit"],
            val_loader=val_loader,
            out_dir=cfg["project"],
            tracker=tracker,
        )
    except RuntimeError as e:
        # e.g. every client failed every round -- artifacts are already
        # written by tracker.finalize() before this is raised.
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Federated training complete. best mAP50-95={result['best_map']:.6f}")


if __name__ == "__main__":
    main()
