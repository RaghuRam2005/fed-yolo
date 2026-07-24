"""
Single source of truth for every tunable hyperparameter, plus the
CLI-flag-over-YAML-file-over-code-default merge used by both
`obj_yolo/train.py` and `obj_yolo/train_federated.py`.

Both CLIs build their `argparse.ArgumentParser` with `default=argparse.SUPPRESS`
on every optional flag, so `vars(args)` only contains keys the user *actually
typed* -- that's what makes the three-way merge below unambiguous:

    code default  <  --config file  <  explicit CLI flag

Note on naming: the augmentation affine-transform "scale" hyperparameter
(`obj_yolo.data.augment.DEFAULT_HYP`'s `"scale"` key, e.g. 0.5) is exposed
here as `aug_scale` to avoid colliding with the model-size `scale` flag
("n"/"s"/"m"/"l"/"x"). `build_aug_hyp()` translates it back to `DEFAULT_HYP`'s
own key name when constructing the dict `YoloDataset` actually expects.
"""
import argparse
from pathlib import Path
from typing import Any, Optional

import torch
import yaml

# -- loss/assignment/eval knobs shared by both centralized and federated training --
_SHARED_DEFAULTS: dict[str, Any] = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "scale": "n",
    "imgsz": 640,
    "lr0": 0.01,
    "momentum": 0.937,
    "weight_decay": 5e-4,
    "box_gain": 7.5,
    "cls_gain": 0.5,
    "dfl_gain": 1.5,
    "tal_topk": 10,
    "tal_alpha": 0.5,
    "tal_beta": 6.0,
    "eval_conf_thres": 0.001,
    "eval_iou_thres": 0.6,
    # augmentation (obj_yolo.data.augment.DEFAULT_HYP, "scale" renamed "aug_scale" -- see module docstring)
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "aug_scale": 0.5,
    "shear": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "project": "runs/train",
    "run_name": None,
    "tensorboard": True,
    "log_histograms": True,
    "seed": 42,
}

TRAIN_DEFAULTS: dict[str, Any] = {
    **_SHARED_DEFAULTS,
    "epochs": 100,
    "batch": 16,
    "workers": 4,
    "lrf": 0.01,
    "warmup_epochs": 3.0,
    "optimizer": "sgd",
    "amp": True,
    "ema_decay": 0.9999,
    "project": "runs/train",
}

FEDERATED_DEFAULTS: dict[str, Any] = {
    **_SHARED_DEFAULTS,
    "rounds": 20,
    "local_epochs": 1,
    "fraction_fit": 1.0,
    "batch": 8,
    "val_batch": 8,
    "project": "runs/federated",
}

AUG_HYP_KEYS = ("hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "aug_scale", "shear", "fliplr", "mosaic")


def build_aug_hyp(cfg: dict) -> dict[str, float]:
    """Translate config's `aug_scale` back to `DEFAULT_HYP`'s own `"scale"` key."""
    hyp = {k: cfg[k] for k in AUG_HYP_KEYS if k in cfg}
    if "aug_scale" in hyp:
        hyp["scale"] = hyp.pop("aug_scale")
    return hyp


def load_config(args: argparse.Namespace, defaults: dict[str, Any]) -> dict[str, Any]:
    """Merge `defaults < --config YAML file < explicit CLI flags`.

    `args` must have been parsed with every optional flag's `default` set to
    `argparse.SUPPRESS`, so unset flags are simply absent from `vars(args)`.
    """
    cfg = dict(defaults)
    config_path: Optional[str] = getattr(args, "config", None)
    if config_path:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"--config file not found: {path}")
        with open(path) as f:
            file_cfg = yaml.safe_load(f) or {}
        cfg.update(file_cfg)
    cfg.update({k: v for k, v in vars(args).items() if k != "config"})
    return cfg
