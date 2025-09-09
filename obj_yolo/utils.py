import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import torch
from torch import nn
from torch.nn import Parameter

from ultralytics import YOLO
from ultralytics.engine.model import Model
from ultralytics.engine.results import Results
from ultralytics.engine.trainer import BaseTrainer

def set_parameters(model: Model, parameters: Dict[str, Parameter], name: str = "Unnamed Model"):
    """
    Set the model's parameters, with detailed logging for debugging.
    
    Args:
        model (Model): YOLO model to update.
        parameters (Dict[str, Parameter]): A dictionary of parameter tensors.
        name (str): The name of the model being updated for clear logs.
    """
    # --- Start of new debug statements ---
    logging.info(f"--- Updating parameters for '{name}' ---")

    original_model_state = model.model.model.state_dict()
    
    target_keys = set(original_model_state.keys())
    incoming_keys = set(parameters.keys())
    
    missing_keys = 0
    mismatch_keys = 0

    for key in target_keys:
        if key not in incoming_keys:
            missing_keys += 1
            logging.warning(f"missing key found: {key}")
        else:
            if original_model_state[key].shape != parameters[key].shape:
                mismatch_keys += 1
                logging.warning(f"mismtach key found: {key}")
                logging.warning(f"target key shape: {target_keys[key].shape}")
                logging.warning(f"target key shape: {incoming_keys[key].shape}")
    logging.info("------------------ SUMMARY -------------------- ")
    logging.info(f"Total missing keys: {missing_keys}")
    logging.info(f"Total shape mismatches: {mismatch_keys}")

    # --- End of new debug statements ---

    first_key = next(iter(parameters))
    if first_key.startswith('model.model.'):
        model.load_state_dict(parameters, strict=False)
    elif first_key.startswith('model.'):
        model.model.load_state_dict(parameters,strict=False)
    else:
        model.model.model.load_state_dict(parameters, strict=False)
    
    logging.info(f"--- Parameter update for '{name}' complete ---")
    return model

def load_model(config:str) -> Dict[str, Parameter]:
    """
    Loads the model from the path specified
    if path is not specified load a pretrained model from ultralytics

    Returns:
        Dict[str, Parameters]: state_dict of the model loaded
    """
    model = None
    if Path(config).exists():
        model = YOLO(config)
    else:
        model = YOLO("yolov11n.pt")
    parameters = get_whole_parameters(model)
    return parameters

def get_whole_parameters(model:Model) -> Optional[Dict[str, Parameter]]:
    """
    Extracts the parameters from the model and gives in numpy format

    Args:
        model (Model): Instance of pytorch model

    Returns:
        Optional[Dict[str, Parameter]]: List of model parameters
    """
    cloned_dict = {k:v.clone() for k, v in model.model.model.state_dict().items()}
    return cloned_dict

def generate_mask(model:Model, sparsity:float) -> Dict[str, torch.Tensor]:    
    """
    Generates a mask for batch normalization weights in a model based on a specified sparsity level.
    This function collects all batch normalization weights (identified by 'bn.weight' in their names),
    computes their absolute values, and determines a threshold such that a given percentage (sparsity)
    of weights are considered "inactive" (below the threshold). It then creates a mask for each batch
    normalization weight tensor, where elements above the threshold are marked as True (active) and
    those below as False (inactive).

    Args:
        model (Model): The model containing layers with batch normalization weights.
        sparsity (float): The fraction (between 0 and 1) of weights to be masked out (set as inactive).

    Returns:
        Dict[str, torch.Tensor]: A dictionary mapping batch normalization weight parameter names to
            boolean masks indicating which weights are above the sparsity threshold.
    """
    layers = model.model.model
    scaling_factors = []
    with torch.no_grad():
        for key, weights in layers.named_parameters():
            if key.endswith('bn.weight'):
                scaling_factors.append(weights.abs())
            
        all_scaling_factors = torch.cat(scaling_factors)
        threshold = np.percentile(all_scaling_factors.detach().numpy(), sparsity * 100)

        mask = {}
        for key, weights in layers.named_parameters():
            if key.endswith('bn.weight'):
                temp = (weights.abs() > threshold)
                mask[key] = temp

    return mask

def apply_mask_to_model(delta:Dict[str, torch.Tensor], mask:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Apply a binary mask to a dictionary of parameter deltas, producing pruned (sparse) deltas.

    Args:
        delta (Dict[str, torch.Tensor]): Dictionary of parameter deltas to be masked.
        mask (Dict[str, torch.Tensor]): Dictionary mapping parameter names to binary masks (torch.Tensor).

    Returns:
        Dict[str, torch.Tensor]: Dictionary mapping parameter names to masked (pruned) tensor values.

    Notes:
        - Only parameters with corresponding keys in the mask dictionary are masked.
        - Masking is performed without gradient tracking (torch.no_grad()).
        - The returned dictionary contains only the masked parameters.
    """
    sparse_params = {}

    with torch.no_grad():
        for key, weights in delta.items():
            if key in mask.keys():
                bool_mask = mask[key].type(weights.dtype)
                pruned_weights = weights * bool_mask
                sparse_params[key] = pruned_weights
            else:
                sparse_params[key] = weights

    return sparse_params

def create_sparse_update(parameters:Dict[str, torch.Tensor]) -> Dict[str, Optional[Tuple]]:
    """
    Converts a dictionary of parameter tensors into a sparse representation.

    For each parameter tensor, this function identifies non-zero elements and records their indices and values.
    The result is a dictionary mapping each parameter name to a tuple containing:
        - A list of indices where the tensor has non-zero values.
        - A list of the corresponding non-zero values.

    Args:
        parameters (Dict[str, torch.Tensor]): 
            A dictionary where keys are parameter names and values are PyTorch tensors.

    Returns:
        Dict[str, Tuple]:
            A dictionary mapping each parameter name to a tuple of (mask, sparse), where:
                - mask (List[int]): Indices of non-zero elements in the tensor.
                - sparse (List[float]): Values of the non-zero elements.
    """
    sparse_update = {}
    for key, weights in parameters.items():
        if key.endswith('bn.weight'):
            # Vectorized sparse representation
            nonzero_indices = (weights != 0).nonzero(as_tuple=True)[0].tolist()
            nonzero_values = weights[nonzero_indices].cpu().tolist()
            sparse_update[key] = (nonzero_indices, nonzero_values)
        else:
            sparse_update[key] = weights
    return sparse_update

def log_pruning_statistics(model: Model, mask: Dict[str, torch.Tensor]):
    """Logs the statistics of the pruning mask to show which weights are zeroed out."""
    logging.info("--- Pruning Statistics ---")
    model_dict = model.model.model.state_dict()
    
    with torch.no_grad():
        for key, bn_mask in mask.items():
            if key in model_dict.keys():
                logging.info(f"key: {key}")
                logging.info(f"non zero parameters (True): {sum(v for v in bn_mask)}")
                logging.info(f"non zero parameters (False): {sum(not v for v in bn_mask)}")
            else:
                logging.warning(f"key: {key} not found in model state dict")            

class L1_BN_reg:
    """Applies L1 regularization to BatchNorm gamma parameters during training."""
    def __init__(self, lam: float = 1e-5):
        self.lam = lam

    def __call__(self, trainer: BaseTrainer):
        l1_penalty = torch.tensor(0.0)
        for key, weights in trainer.model.model.state_dict().items():
            if key.endswith('bn.weight'):
                l1_penalty += torch.abs(weights).sum()
                weights += trainer.loss + (self.lam * l1_penalty)
            trainer.model.model.state_dict()[key] = weights
        # for module in trainer.model.modules():
        #     if isinstance(module, nn.BatchNorm2d) and hasattr(module, "weight") and module.weight is not None:
        #         l1_penalty += torch.abs(module.weight).sum()
        # (self.lam * l1_penalty).backward()

def on_train_start(trainer:BaseTrainer):
    """Initialize sparsity trainer and log initial parameters"""
    backbone = trainer.model.model[:11]
    total_params = sum(p.numel() for p in backbone.parameters())
    bn_params = sum(m.weight.numel() for m in backbone.modules()
                   if isinstance(m, nn.BatchNorm2d) and m.weight is not None)

    print(f"Training started:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  BatchNorm gamma parameters: {bn_params:,}")

def yolo_train(model:Model, data_path:str, epochs:int, sparsity:float, lam:float) -> Tuple[Dict[str, torch.tensor_split], Results, List[int]]:
    """YOLO model training function.""" 
    l1_regularizer = L1_BN_reg(lam=lam)

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_batch_end", l1_regularizer)

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

    # generate mask
    mask = generate_mask(model=model, sparsity=sparsity)
    
    # Add the new logging call to show pruning results
    log_pruning_statistics(model=model, mask=mask)

    return model, results, mask
