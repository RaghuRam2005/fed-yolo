import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict, OrderedDict, Any

import torch
from torch import nn

from ultralytics import YOLO
from ultralytics.engine.model import Model
from ultralytics.engine.results import Results
from ultralytics.engine.trainer import BaseTrainer

def set_parameters(model: Model, parameters: List[np.ndarray]):
    """
    Set the model's parameters from a list of NumPy arrays,
    excluding BatchNorm running statistics (running_mean, running_var, num_batches_tracked).
    
    Args:
        model (Model): YOLO model
        parameters (List[np.ndarray]): List of parameter arrays matching model.state_dict() order
    """
    # Build state dict from given parameters
    params_dict = zip(model.state_dict().keys(), parameters)
    new_state_dict = OrderedDict()

    for k, v in params_dict:
        if any(stat in k for stat in ["running_mean", "running_var", "num_batches_tracked"]):
            # keep original BN buffers
            new_state_dict[k] = model.state_dict()[k]
        else:
            # update trainable params
            new_state_dict[k] = torch.tensor(v, dtype=model.state_dict()[k].dtype)

    # Load into model
    model.load_state_dict(new_state_dict, strict=True)
    return model

def load_model(config:str) -> Optional[List[np.ndarray]]:
    """
    Loads the model from the path specified
    if path is not specified load a pretrained model from ultralytics

    Returns:
        Optional[List[np.ndarray]]: parameters of the model loaded
    """
    model = None
    if Path(config).exists():
        model = YOLO(config)
    else:
        model = YOLO("yolo11n.pt")
    ndarray = get_whole_parameters(model)
    return ndarray

def get_whole_parameters(model:Model) -> Optional[List[np.ndarray]]:
    """
    Extracts the parameters from the model and gives in numpy format

    Args:
        model (Model): Instance of pytorch model

    Returns:
        Optional[List[np.ndarray]]: List of model parameters
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def prune_and_make_mask(backbone: nn.Module, sparsity: float) -> Dict[str, torch.Tensor]:
    """Computes a pruning mask based on the magnitudes of BatchNorm gamma parameters."""
    if sparsity <= 0.0:
        return {}

    gammas = [m.weight.data.abs().flatten() for m in backbone.modules() if isinstance(m, nn.BatchNorm2d) and hasattr(m, "weight")]
    if not gammas:
        return {}

    all_gammas = torch.cat(gammas)
    q = min(max(float(sparsity), 0.0), 0.99)
    threshold = torch.quantile(all_gammas, q).item()

    mask_by_name: Dict[str, torch.Tensor] = {}
    for name, module in backbone.named_modules():
        if isinstance(module, nn.BatchNorm2d) and hasattr(module, "weight"):
            mask = (module.weight.data.abs() > threshold).float()
            mask_by_name[f"{name}.weight"] = mask.clone()
            mask_by_name[f"{name}.bias"] = mask.clone()
    return mask_by_name

def apply_mask_to_backbone(backbone: nn.Module, mask_by_name: Dict[str, torch.Tensor]):
    """Applies the pruning mask to the backbone's parameters."""
    with torch.no_grad():
        for name, param in backbone.named_parameters():
            if name in mask_by_name:
                param.mul_(mask_by_name[name].to(param.device))

def export_sparse_delta(original_weights: OrderedDict, model: nn.Module, mask_by_name: Dict[str, torch.Tensor]):
    """
    Compute masked deltas for backbone, dense deltas for head.
    Skip BN running stats (they will be recomputed globally).
    """
    new_state = model.state_dict()
    deltas = {k: new_state[k] - original_weights[k] for k in original_weights if k in new_state}

    delta = {"backbone_delta": {}, "head_delta": {}}

    # Backbone: apply mask only to trainable BN params
    backbone = model.model[:11]
    for name, module in backbone.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            weight_key = f"{name}.weight"
            bias_key = f"{name}.bias"

            mask = mask_by_name.get(weight_key)
            if mask is None:
                continue

            if weight_key in deltas:
                delta["backbone_delta"][weight_key] = deltas[weight_key] * mask
            if bias_key in deltas:
                delta["backbone_delta"][bias_key] = deltas[bias_key] * mask

            # skip running_mean and running_var (buffers)

    # Head: full dense deltas
    head = model.model[11:]
    for name, param in head.named_parameters():
        if name in deltas:
            delta["head_delta"][name] = deltas[name]

    return delta


class L1_BN_reg:
    """Applies L1 regularization to BatchNorm gamma parameters during training."""
    def __init__(self, lam: float = 1e-5):
        self.lam = lam

    def __call__(self, trainer: BaseTrainer):
        l1_penalty = torch.tensor(0.0, device=trainer.model.device)
        for module in trainer.model.modules():
            if isinstance(module, nn.BatchNorm2d) and hasattr(module, "weight") and module.weight is not None:
                l1_penalty += torch.abs(module.weight).sum()
        (self.lam * l1_penalty).backward()

def on_train_start(trainer:BaseTrainer):
    """Initialize sparsity trainer and log initial parameters"""
    backbone = trainer.model.model[:11]
    total_params = sum(p.numel() for p in backbone.parameters())
    bn_params = sum(m.weight.numel() for m in backbone.modules()
                   if isinstance(m, nn.BatchNorm2d) and m.weight is not None)

    print(f"Training started:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  BatchNorm gamma parameters: {bn_params:,}")

class on_train_end_pruning:
    """Callback to prune the model's backbone at the end of training."""
    def __init__(self, sparsity: float = 0.2):
        self.sparsity = sparsity

    def __call__(self, backbone:Model):
        logging.info(f"Applying pruning with sparsity {self.sparsity:.3f}")
        mask = prune_and_make_mask(backbone, self.sparsity)
        return mask

# def on_val_end(validator:BaseValidator):
#     """ returns the loss acculated during the validation """
#     if not losses:
#         raise ValueError("No losses variable found while calling on_val_end")
#     losses = validator.loss

def yolo_train(model:Model, data_path:str, epochs:int, sparsity:float, lam:float) -> Tuple[Model, Results, List[int]]:
    """YOLO model training function.""" 
    prune_parameters = on_train_end_pruning(sparsity)
    l1_loss_func = L1_BN_reg(lam)

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("optimizer_step", l1_loss_func)

    results = model.train(
        data=data_path,
        epochs=epochs,
        batch=8,
        save=False,
        plots=False,
        device=-1,
        verbose=False,
        project="fed_yolo_runs",
        name=f"client_train",
        exist_ok=True,
        cos_lr=True,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
    )
    backbone = model.model[:11]
    mask = prune_parameters(backbone)
    apply_mask_to_backbone(backbone)

    return model, results, mask
