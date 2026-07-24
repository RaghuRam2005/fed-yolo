"""
Centralized YOLOv8 training entrypoint (single machine, no federated learning
-- see project README for status).

    uv run python -m obj_yolo.train --data dataset/client_0/data.yaml --config configs/train.yaml
    uv run python -m obj_yolo.train --data dataset/client_0/data.yaml --scale n --epochs 100 --imgsz 640 --batch 16

Every hyperparameter has a single source of truth in
`obj_yolo.config.TRAIN_DEFAULTS`; `--config <file.yaml>` overrides it,
explicit CLI flags override the file (see `obj_yolo/config.py`).

Optimizer param grouping (weight-decay on conv weights only, none on biases
or BatchNorm weights), warmup+cosine LR schedule, and EMA of weights are all
Ultralytics-default training conventions, reimplemented here directly on top
of the from-scratch model/loss (see `obj_yolo/model`, `obj_yolo/loss`).
"""
import argparse
import copy
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from obj_yolo.config import TRAIN_DEFAULTS, build_aug_hyp, load_config
from obj_yolo.config_validation import validate_data_yaml, validate_imgsz, validate_positive
from obj_yolo.data.yolo_dataset import YoloDataset, collate_fn
from obj_yolo.loss.loss import DetectionLoss
from obj_yolo.metrics.grad_stats import block_grad_norms, block_weight_norms, total_grad_norm
from obj_yolo.metrics.run_tracker import RunTracker
from obj_yolo.model.yolov8 import YOLOv8
from obj_yolo.val import evaluate


class ModelEMA:
    """Exponential moving average of model weights (decoupled from optimizer state)."""

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1 - self.decay)


def build_optimizer(model: nn.Module, name: str, lr: float, momentum: float, weight_decay: float) -> optim.Optimizer:
    """3 param groups: biases (no decay), BatchNorm weights (no decay), conv/linear weights (decay)."""
    g_bias, g_bn, g_weight = [], [], []
    for module in model.modules():
        for pname, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            if pname == "bias":
                g_bias.append(p)
            elif pname == "weight" and isinstance(module, nn.BatchNorm2d):
                g_bn.append(p)
            else:
                g_weight.append(p)

    if name == "sgd":
        optimizer = optim.SGD(g_bias, lr=lr, momentum=momentum, nesterov=True)
    else:
        optimizer = optim.AdamW(g_bias, lr=lr, betas=(momentum, 0.999))
    optimizer.add_param_group({"params": g_weight, "weight_decay": weight_decay})
    optimizer.add_param_group({"params": g_bn, "weight_decay": 0.0})
    return optimizer


def cosine_lr_lambda(epoch: int, epochs: int, warmup_epochs: float, lrf: float) -> float:
    """Linear warmup to 1.0, then cosine decay to `lrf` (fraction of lr0), as a
    multiplicative factor for `torch.optim.lr_scheduler.LambdaLR`."""
    if epoch < warmup_epochs:
        return (epoch + 1) / max(1.0, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1.0, epochs - warmup_epochs)
    return lrf + (1 - lrf) * (1 + math.cos(math.pi * progress)) / 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLOv8 from scratch in PyTorch (no ultralytics).")
    S = argparse.SUPPRESS
    p.add_argument("--data", default=S, help="path to data.yaml (train/val/nc/names); may also be set via --config's 'data' key")
    p.add_argument("--config", default=None, help="YAML file of defaults; CLI flags override it")
    p.add_argument("--scale", choices=["n", "s", "m", "l", "x"], default=S)
    p.add_argument("--epochs", type=int, default=S)
    p.add_argument("--imgsz", type=int, default=S)
    p.add_argument("--batch", type=int, default=S)
    p.add_argument("--workers", type=int, default=S)
    p.add_argument("--lr0", type=float, default=S)
    p.add_argument("--lrf", type=float, default=S, help="final LR as a fraction of lr0")
    p.add_argument("--momentum", type=float, default=S)
    p.add_argument("--weight-decay", type=float, default=S)
    p.add_argument("--warmup-epochs", type=float, default=S)
    p.add_argument("--optimizer", choices=["sgd", "adamw"], default=S)
    p.add_argument("--amp", dest="amp", action="store_true", default=S)
    p.add_argument("--no-amp", dest="amp", action="store_false", default=S)
    p.add_argument("--ema-decay", type=float, default=S)
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
    cfg = load_config(args, TRAIN_DEFAULTS)

    try:
        if not cfg.get("data"):
            raise ValueError("--data must be set (either as a CLI flag or a 'data' key in --config)")
        validate_imgsz(cfg["imgsz"])
        validate_positive(cfg["epochs"], "--epochs")
        data_cfg = validate_data_yaml(cfg["data"])
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    torch.manual_seed(cfg["seed"])
    device = torch.device(cfg["device"])
    amp_enabled = cfg["amp"] and device.type == "cuda"
    nc = int(data_cfg["nc"])

    tracker = RunTracker(
        cfg["project"], run_name=cfg["run_name"], config=cfg,
        use_tensorboard=cfg["tensorboard"], log_histograms=cfg["log_histograms"],
    )

    model = YOLOv8(nc=nc, scale=cfg["scale"]).to(device)
    tracker.log_model_summary(model, round_num=0)

    aug_hyp = build_aug_hyp(cfg)
    train_dir = Path(data_cfg["train"])
    train_ds = YoloDataset(
        train_dir, Path(str(train_dir).replace("images", "labels")), imgsz=cfg["imgsz"], augment=True, hyp=aug_hyp
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch"],
        shuffle=True,
        num_workers=cfg["workers"],
        collate_fn=collate_fn,
        drop_last=True,
    )

    val_loader = None
    if data_cfg.get("val"):
        val_dir = Path(data_cfg["val"])
        val_ds = YoloDataset(
            val_dir, Path(str(val_dir).replace("images", "labels")), imgsz=cfg["imgsz"], augment=False
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg["batch"], shuffle=False, num_workers=cfg["workers"], collate_fn=collate_fn
        )

    loss_fn = DetectionLoss(
        nc=nc, stride=model.detect.stride, device=device,
        box_gain=cfg["box_gain"], cls_gain=cfg["cls_gain"], dfl_gain=cfg["dfl_gain"],
        tal_topk=cfg["tal_topk"], tal_alpha=cfg["tal_alpha"], tal_beta=cfg["tal_beta"],
    )
    optimizer = build_optimizer(model, cfg["optimizer"], cfg["lr0"], cfg["momentum"], cfg["weight_decay"])
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda e: cosine_lr_lambda(e, cfg["epochs"], cfg["warmup_epochs"], cfg["lrf"])
    )
    scaler = torch.amp.GradScaler(device="cuda", enabled=amp_enabled)
    ema = ModelEMA(model, decay=cfg["ema_decay"])

    weights_dir = tracker.weights_dir

    best_map = 0.0
    for epoch in range(cfg["epochs"]):
        model.train()
        epoch_loss = torch.zeros(3, device=device)
        t0 = time.time()

        for batch in train_loader:
            imgs = batch["img"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                preds = model(imgs)
                total_loss, loss_items = loss_fn(preds, batch)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)  # internally unscales grads before stepping
            scaler.update()
            ema.update(model)
            epoch_loss += loss_items

        # Grad stats from the just-completed last batch: scaler.step() above
        # already unscaled .grad, so these are real (not loss-scaled) norms.
        w_norms = block_weight_norms(model)
        g_norms = block_grad_norms(model)
        g_total = total_grad_norm(model)
        tracker.log_grad_stats(epoch + 1, w_norms, g_norms, g_total, tag="train")
        tracker.log_weight_histograms(model, epoch + 1)
        tracker.log_grad_histograms(model, epoch + 1)

        scheduler.step()
        n_batches = len(train_loader)
        box_l, cls_l, dfl_l = (epoch_loss / n_batches).tolist()
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch + 1}/{cfg['epochs']}  box={box_l:.6f} cls={cls_l:.6f} dfl={dfl_l:.6f}  "
            f"lr={lr:.6f}  grad_norm={g_total:.6f}"
        )

        torch.save(model.state_dict(), weights_dir / "last.pt")
        tracker.log_model_summary(model, round_num=epoch + 1)

        if val_loader is not None:
            metrics = evaluate(ema.ema, val_loader, device, conf_thres=cfg["eval_conf_thres"], iou_thres=cfg["eval_iou_thres"])
            duration = time.time() - t0
            tracker.log_central_round(
                epoch + 1, metrics, duration, num_clients=1, total_examples=len(train_ds)
            )
            print(
                f"  val P={metrics['precision']:.6f} R={metrics['recall']:.6f} F1={metrics['f1']:.6f} "
                f"mAP50={metrics['map50']:.6f} mAP50-95={metrics['map']:.6f}"
            )
            if metrics["map"] > best_map:
                best_map = metrics["map"]
                torch.save(ema.ema.state_dict(), weights_dir / "best.pt")

    tracker.finalize({"best_map": best_map, "epochs": cfg["epochs"]})
    print(f"Training complete. best mAP50-95={best_map:.6f}. Weights saved to {weights_dir}")


if __name__ == "__main__":
    main()
