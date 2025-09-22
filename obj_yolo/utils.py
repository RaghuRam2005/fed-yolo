# utils.py
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm

from ultralytics.engine.model import Model
from ultralytics.engine.results import Results

@dataclass
class ClientConfig:
    """
    Client Configuration data class
    """
    epochs:int=10

@dataclass
class ClientFitRes:
    """
    ClientFitRes is a data class representing the result of a client's fit operation.
    """
    delta:Dict[str, Optional[Tuple]]
    metrics:Results
    sparsity:float
    datacount:int = 1000

def client_train(model:Model, data_path:str, client_id:int, epochs:int) -> Results:
    # training the local model
    results = model.train(
        data = data_path,
        epochs=epochs,
        device=-1,
        batch=8,
        imgsz=640,
        save=False, # client doesn't have much storage
        cache='disk',
        project="fed_yolo",
        name=f"client_{client_id}_train",
        exist_ok=True,
        pretrained="yolo11n.pt",
        optimizer='auto',
        seed=32,
        deterministic=True,
        freeze=0,
        plots=True, # useful for verification
    )

    return results

def generate_mask(model:Model, sparsity:float) -> Dict[str, Tensor]:
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
        for _, module in layers.named_modules():
            if isinstance(module, _BatchNorm):
                scaling_factors.append(module.weight.abs())

        scaling_factors = torch.cat(scaling_factors)
        threshold = torch.quantile(input=scaling_factors, q=sparsity)

        mask = {}
        for key, weight in layers.named_parameters():
            if key.endswith('bn.weight'):
                temp = (weight.abs() > threshold)
                mask[key] = temp
    
    return mask

def log_pruning_statistics(model:Model, mask:Dict[str, Tensor]) -> None:
    layers = model.model.model

    logging.info("----------------- Pruning Statistics ------------------")
    for key, _ in layers.named_parameters():
        if key.endswith('bn.weight'):
            inactive_params = (~mask[key]).sum().item()
            active_params = mask[key].sum().item()
            logging.info(f'Key              : {key}')
            logging.info(f'Active Parameters: {active_params}')
            logging.info(f'Inactive Params  : {inactive_params}')

def apply_mask_to_model(delta:Dict[str, Tensor], mask:Dict[str, Tensor]) -> Dict[str, Tensor]:
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
    sparse_parameters = {}

    for key, weights in delta.items():
        if key.endswith('bn.weight') and key not in mask.keys():
            raise Exception(f"Key {key} not found")
        elif key.endswith('bn.weight') and key in mask.keys():
            bool_mask = mask[key].type(weights.dtype)
            assert(bool_mask.shape == weights.shape), f"Weights shape and mask shape doesn't match at {key}"
            weights = (weights * bool_mask)
        sparse_parameters[key] = weights
    
    assert delta.keys() == sparse_parameters.keys(), \
        "Keys mismatch between delta and sparse parameters"

     # check consistency in value shapes
    for key in delta.keys():
        assert delta[key].shape == sparse_parameters[key].shape, \
            f"Shape mismatch after masking at {key}: {delta[key].shape} vs {sparse_parameters[key].shape}"
        
    return sparse_parameters

def create_sparse_update(parameters:Dict[str, Tensor]) -> Dict[str, Optional[Tuple]]:
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
            nonzero_indices = (weights != 0).nonzero(as_tuple=True)[0].tolist()
            nonzero_values = weights[nonzero_indices].cpu().tolist()
            sparse_update[key] = (nonzero_indices, nonzero_values)
        else:
            sparse_update[key] = weights
    
    assert(sparse_update.keys() == parameters.keys()), \
    "sparse update keys and the parameter keys do not match"

    return sparse_update

def set_parameters(model:Model, parameters:Dict[str, Tensor]) -> Model:
    """
    Set the model's parameters, with detailed logging for debugging.
    
    Args:
        model (Model): YOLO model to update.
        parameters (Dict[str, Parameter]): A dictionary of parameter tensors.
    """
    model.model.model.load_state_dict(parameters, strict=True)
    return model
