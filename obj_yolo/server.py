""" Server App for federated learning """
import os
from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp

from ultralytics import YOLO
from ultralytics.utils.torch_utils import unwrap_model

from obj_yolo.strategy import CustomFedAvg

server_app = ServerApp()

@server_app.main()
def main(grid:Grid, context:Context) -> None:
    """
    main entry point for server

    Args:
        grid (Grid)
        context (Context)
    """

    # Read run config
    fraction_fit: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]

    # Load global model
    BASE_LIB_PATH = os.path.abspath(os.path.dirname(__file__))
    BASE_DIR_PATH = os.path.dirname(BASE_LIB_PATH)
    YOLO_CONFIG = Path(BASE_DIR_PATH) / "yolo_config" / "yolo11n.yaml"

    global_model = YOLO(YOLO_CONFIG).load('yolo11n.pt')

    # record global model state
    unwrapped_model = unwrap_model(global_model)
    state_dict = unwrapped_model.model.state_dict()
    tensor_state_dict = {
        k: v.cpu().numpy()
        for k, v in state_dict.items()
        if isinstance(v, torch.Tensor)
    }
    arrays = ArrayRecord(tensor_state_dict)

    # Initialize FedAvg strategy
    strategy = CustomFedAvg(fraction_fit=fraction_fit, fraction_evaluate=1.0, min_fit_clients=2, min_evaluate_clients=2, min_available_clients=2)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")
