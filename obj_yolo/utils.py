# util.py
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import torch
from torch.nn.modules.batchnorm import _BatchNorm

from ultralytics.engine.model import Model
from ultralytics.engine.results import Results
from ultralytics.utils.torch_utils import unwrap_model

@dataclass
class DataConfigTag:
    weather_dict:Dict[str, List]
    scene_dict:Dict[str, List]

@dataclass
class ServerConfigFedWeg:
    """
    Sever Configuration
    """
    client_model_path:str
    client_train_data_count:int
    client_val_data_count:int
    communication_rounds:int=2
    num_nodes:int=2

@dataclass
class ServerConfigFedTag:
    client_model_path:str
    client_tag_dict:Dict
    client_train_data_count:int
    client_val_data_count:int
    communication_rounds:int=2
    num_nodes:int=2

@dataclass
class FitConfig:
    """
    Fit configuration class for clients
    """
    epochs:int=10

@dataclass
class FitResFedWeg:
    """
    FitRes is a data class representing the result of a client's fit operation.
    """
    delta:Dict[str, Optional[Tuple]]
    metrics:Results
    sparsity:float
    
def model_state(model:Model) -> Dict[str, torch.Tensor]:
    unwrapped_model = unwrap_model(model)
    return unwrapped_model.model.model.state_dict()

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
        l1_lambda=1e-4,
    )
    return results

def to_coo(t:torch.Tensor, tau:float) -> torch.Tensor:
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
    return torch.sparse_coo_tensor(indices=idx, values=vals, size=t.shape).coalesce()

def from_coo(t:torch.Tensor) -> torch.Tensor:
    """
    Converts a sparse COO tensor to a dense tensor.

    Args:
        s (torch.Tensor): A sparse tensor in COO format.

    Returns:
        torch.Tensor: The dense representation of the input tensor.
    """
    return t.to_dense()

def update_sparsity_for_all_clients(fedtag, clients, results) -> None:
    sorted_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
    result_keys = list(sorted_results.keys())
    mid = len(result_keys) // 2
    change_value_dict = {k:0.1 if i < mid else -0.1 for i, k in enumerate(result_keys)}
    for client in clients:
        client.sparsity = fedtag.update_sparsity(current_sparsity=client.sparsity, change=change_value_dict[client.tag])

