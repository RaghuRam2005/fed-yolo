""" Server App for federated learning """
import os
from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp

from ultralytics import YOLO
from ultralytics.utils.torch_utils import unwrap_model

from obj_yolo.strategy.fedavg import CustomFedAvg
from obj_yolo.strategy.fedadam import CustomFedAdam

server_app = ServerApp()

@server_app.main()
def main(grid:Grid, context:Context) -> None:
    """
    main entry point for server

    Args:
        grid (Grid)
        context (Context)
    """
    BASE_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

    # Read run config
    dataset_name = context.run_config["dataset-name"]
    fraction_train: int = int(context.run_config["fraction-train"])
    num_rounds: int = int(context.run_config["num-server-rounds"])
    lr: float = float(context.run_config["lr"])

    # Load global model
    yolo_model_config = Path(BASE_PATH) / "yolo_config" / f"{dataset_name}_yolo11n.yaml"
    model_path = Path(BASE_PATH) / "flwr_simulation" / f"{dataset_name}" / "aggregated_model" / "agg_model.pt"

    global_model = YOLO(yolo_model_config).load('yolo11n.pt')
    global_model.save(model_path)
    
    parameters = global_model.state_dict()
    
    arrays = ArrayRecord(parameters, keep_input=True)

    # Initialize FedAvg strategy
    strategy = CustomFedAvg(
        fraction_train=fraction_train, 
        fraction_evaluate=1, 
        min_train_nodes=3,
        min_evaluate_nodes=3, 
        min_available_nodes=3,
        dataset_name=str(dataset_name)
    )
    #strategy = CustomFedAdam(
    #    fraction_train=fraction_train,
    #    fraction_evaluate=1.0,
    #    min_train_nodes=2,
    #    min_evaluate_nodes=2,
    #    min_available_nodes=2,
    #    eta=0.1,
    #    eta_l=0.3,
    #    beta_1=0.9,
    #    beta_2=0.9,
    #    tau=0.001,
    #)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")
