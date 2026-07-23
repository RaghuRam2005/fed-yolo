"""
Centralized YOLOv8 training entrypoint (single machine, no federated learning
-- see project README for status).

    uv run python -m obj_yolo.train --data dataset/client_0/data.yaml --scale n --epochs 100 --imgsz 640 --batch 16

Optimizer param grouping (weight-decay on conv weights only, none on biases
or BatchNorm weights), warmup+cosine LR schedule, and EMA of weights are all
Ultralytics-default training conventions, reimplemented here directly on top
of the from-scratch model/loss (see `obj_yolo/model`, `obj_yolo/loss`).
"""
import argparse
import copy
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from obj_yolo.data.yolo_dataset import YoloDataset, collate_fn
from obj_yolo.loss.loss import DetectionLoss
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
    p.add_argument("--data", required=True, help="path to data.yaml (train/val/nc/names)")
    p.add_argument("--scale", default="n", choices=["n", "s", "m", "l", "x"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--lrf", type=float, default=0.01, help="final LR as a fraction of lr0")
    p.add_argument("--momentum", type=float, default=0.937)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--warmup-epochs", type=float, default=3.0)
    p.add_argument("--optimizer", default="sgd", choices=["sgd", "adamw"])
    p.add_argument("--amp", dest="amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--project", default="runs/train")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    amp_enabled = args.amp and device.type == "cuda"

    with open(args.data) as f:
        data_cfg = yaml.safe_load(f)
    nc = int(data_cfg["nc"])

    model = YOLOv8(nc=nc, scale=args.scale).to(device)

    train_dir = Path(data_cfg["train"])
    train_ds = YoloDataset(
        train_dir, Path(str(train_dir).replace("images", "labels")), imgsz=args.imgsz, augment=True
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        drop_last=True,
    )

    val_loader = None
    if data_cfg.get("val"):
        val_dir = Path(data_cfg["val"])
        val_ds = YoloDataset(
            val_dir, Path(str(val_dir).replace("images", "labels")), imgsz=args.imgsz, augment=False
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, collate_fn=collate_fn
        )

    loss_fn = DetectionLoss(nc=nc, stride=model.detect.stride, device=device)
    optimizer = build_optimizer(model, args.optimizer, args.lr0, args.momentum, args.weight_decay)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda e: cosine_lr_lambda(e, args.epochs, args.warmup_epochs, args.lrf)
    )
    scaler = torch.amp.GradScaler(device="cuda", enabled=amp_enabled)
    ema = ModelEMA(model, decay=args.ema_decay)

    weights_dir = Path(args.project) / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    best_map = 0.0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = torch.zeros(3, device=device)

        for batch in train_loader:
            imgs = batch["img"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                preds = model(imgs)
                total_loss, loss_items = loss_fn(preds, batch)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            epoch_loss += loss_items

        scheduler.step()
        n_batches = len(train_loader)
        box_l, cls_l, dfl_l = (epoch_loss / n_batches).tolist()
        lr = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch + 1}/{args.epochs}  box={box_l:.4f} cls={cls_l:.4f} dfl={dfl_l:.4f}  lr={lr:.6f}")

        torch.save(model.state_dict(), weights_dir / "last.pt")

        if val_loader is not None:
            metrics = evaluate(ema.ema, val_loader, device)
            print(f"  val mAP50-95={metrics['map']:.4f}  mAP50={metrics['map_50']:.4f}")
            if metrics["map"] > best_map:
                best_map = metrics["map"]
                torch.save(ema.ema.state_dict(), weights_dir / "best.pt")

    print(f"Training complete. best mAP50-95={best_map:.4f}. Weights saved to {weights_dir}")


if __name__ == "__main__":
    main()
