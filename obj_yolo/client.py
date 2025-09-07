import numpy as np
from typing import List, Optional, Dict
from dataclasses import dataclass

from ultralytics.engine.model import Model

from utils import yolo_train, export_sparse_delta, set_parameters

@dataclass
class ClientFitParams:
    """
    data class to store client parameters
    """
    sparsity:float = 0.2
    epochs:int = 2
    lam:float = 1e-5

@dataclass
class ClientManager:
    """
    data class to store the client
    """
    client_id:int
    rounds_completed:int=0
    fit_params: ClientFitParams
    model:Model

@dataclass
class FitRes:
    """
    data class fit results
    """
    delta:Optional[Dict[str, np.ndarray]]
    results:Dict[str, Optional[float]]
    data_count:int = 1000

class Client:
    def __init__(self, client_manager:ClientManager):
        self.client_manager = client_manager
    
    def update_client_model(self, parameters:List[np.ndarray]):
        self.client_manager.model = set_parameters(
            model=self.client_manager.model,
            parameters=parameters
        )
    
    def fit(self, data_path:str) -> FitRes:
        """
        data class fit results

        Args:
            data_path (str): data path for the client

        Returns:
            FitRes: results of the training
        """
        model = self.client_manager.model
        original_state = {k: v.clone() for k, v in model.state_dict().items()}

        # train locally
        model, results, mask = yolo_train(
            model, data_path, 
            self.client_manager.fit_params.epochs, 
            self.client_manager.fit_params.sparsity, 
            self.client_manager.fit_params.lam
        )

        delta = export_sparse_delta(original_weights=original_state, model=model, mask=mask)

        metrics = {
            "map50-95" : results.box.map,
            "map50" : results.box.map50,
            "map75":results.box.map75,
            "sparsity" : self.client_manager.fit_params.sparsity,
        }

        return FitRes(delta=delta, results=metrics)

    def evaluate(self, paramaters:List[np.ndarray], data_path:str) -> Dict[str, float]:
        """
        Evaluate the client using the aggregated parameters

        Args:
            paramaters (List[np.ndarray]): aggregated parameters after aggregate_fit
            data_path (str): path for yaml file
        """
        model = set_parameters(model=self.client_manager.model, parameters=paramaters)
        results = model.val(
            data_path=data_path,
            save=False,
            device=-1,
            verbose=False,
        )
        metrics = {
            "mAP50-95" : results.box.map,
            "map50": results.box.map50,
            "map75" : results.box.map75,
        }
        return metrics
