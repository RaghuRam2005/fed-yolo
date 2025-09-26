# utils.py
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
from torch.nn.modules.batchnorm import _BatchNorm

from ultralytics.engine.model import Model
from ultralytics.engine.results import Results
from ultralytics.utils.torch_utils import unwrap_model

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
    """
    Trains a local YOLO model for a specific client using the provided dataset.
    Args:
        model (Model): The YOLO model instance to be trained.
        data_path (str): Path to the training data.
        client_id (int): Unique identifier for the client.
        epochs (int): Number of training epochs.
    Returns:
        Results: Training results containing metrics and artifacts.
    """
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

def channel_index_masks(model:Model, tau:float) -> Dict[str, torch.Tensor]:
    """
    Generates masks for the channel indices of BatchNorm layers in a model, 
    selecting channels whose absolute weight values exceed a given threshold.

    Args:
        model (Model): The neural network model containing BatchNorm layers.
        tau (float): Threshold value for selecting channels based on BatchNorm weights.

    Returns:
        Dict[str, torch.Tensor]: A dictionary mapping BatchNorm layer names to tensors 
        containing the indices of channels whose absolute weights are greater than tau.
    """
    bn = [(n, m) for n, m in unwrap_model(model).named_modules() if isinstance(m, _BatchNorm)]
    return {n: (m.weight.abs() > tau).nonzero(as_tuple=False).squeeze(1).to(torch.int32) for n, m in bn}

def to_coo(t: torch.Tensor, tau:float) -> torch.Tensor:
    """
    Converts a dense tensor to a sparse COO tensor by thresholding absolute values.

    Args:
        t (torch.Tensor): The input dense tensor.
        tau (float): Threshold value; elements with absolute value greater than tau are kept.

    Returns:
        torch.Tensor: A sparse COO tensor containing only the elements of `t` whose absolute value exceeds `tau`.
    """
    m = t.abs() > tau
    idx = m.nonzero(as_tuple=False).T
    vals = t[m].float()
    return torch.sparse_coo_tensor(idx, vals, size=t.shape).coalesce()

def from_coo(s: torch.Tensor) -> torch.Tensor:
    """
    Converts a sparse COO tensor to a dense tensor.

    Args:
        s (torch.Tensor): A sparse tensor in COO format.

    Returns:
        torch.Tensor: The dense representation of the input tensor.
    """
    return s.to_dense()
