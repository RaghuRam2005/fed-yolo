#client.py
import os

from ultralytics.engine.model import Model
from ultralytics.engine.results import Results

from .utils import (
    ClientConfig,
    ClientFitRes,

    client_train,

    generate_mask,
    log_pruning_statistics,
    apply_mask_to_model,
    create_sparse_update
)

class Client:
    def __init__(self, client_id:int, sparsity:float):
        self.client_id = client_id
        self.sparsity = sparsity
        self.model:Model = None
        self.rounds_completed = 0
    
    def set_model(self, model:Model):
        self.model = model

    def update_sparsity(self) -> None:
        """
        Updates the client's sparsity parameter based on the number of rounds completed.
        Sparsity increases as the training progresses.
        """
        min_sparsity = 0.2
        max_sparsity = 0.9
        growth_rate = 0.05

        target = min_sparsity + growth_rate * self.rounds_completed
        new_sparsity = min(max(target, min_sparsity), max_sparsity)
        self.sparsity = new_sparsity
    
    def fit(self, client_config:ClientConfig, data_path:str) -> ClientFitRes:
        if not self.model:
            raise AttributeError(f"client {self.client_id} model attribute is not set")
        
        client_parameters = self.model.model.model.state_dict()
        
        # training the local model
        results = client_train(
            model=self.model,
            data_path=data_path,
            client_id=self.client_id,
            epochs=client_config.epochs,
        )

        mask = generate_mask(model=self.model, sparsity=self.sparsity)

        log_pruning_statistics(model=self.model, mask=mask)

        delta = {}
        for key, weights in self.model.model.model.named_parameters():
            if key.endswith(('running_mean', 'running_var', 'num_batches_tracked')):
                delta[key] = weights
            else:
                delta[key] = client_parameters[key] - weights
        
        for d_key in delta.keys():
            if d_key.endswith(('running_mean', 'running_var', 'num_batches_tracked')):
                continue
            assert d_key in client_parameters.keys(), \
            f"delta key: {d_key} not found, client keys: {client_parameters.keys()}"

        sparse_weights = apply_mask_to_model(delta=delta, mask=mask)

        for s_key in sparse_weights.keys():
            if s_key.endswith(('running_mean', 'running_var', 'num_batches_tracked')):
                continue
            assert s_key in client_parameters.keys(), \
            f"sparse weight key: {s_key} not found, client keys: {client_parameters.keys()}"


        sparse_update = create_sparse_update(parameters=sparse_weights)

        assert(sparse_update.keys() == sparse_weights.keys()), \
        "sparse update keys and sparse weight keys doesn't match"

        client_res = ClientFitRes(delta=sparse_update, metrics=results, datacount=1000, sparsity=self.sparsity)

        self.rounds_completed += 1
        self.update_sparsity()

        return client_res

    def evaluate(self, data_path:str) -> Results:
        metrics = self.model.val(
            data=data_path,
            save_json=True,
            plots=True,
            project="fed_yolo",
            name=f"client_{self.client_id}_val",
        )

        return metrics
