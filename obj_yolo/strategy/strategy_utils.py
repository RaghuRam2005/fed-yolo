from pathlib import Path

from flwr.common import ArrayRecord

from ultralytics import YOLO

def load_and_update_model(model_path:Path, aggregated_state:ArrayRecord) -> YOLO:
    """
    Check if the aggregated YOLO model records are of same size as the
    Original Model records

    Args:
        model_path (Path): yolo model base path (yaml file)
        aggregated_state (ArrayRecord): aggregated state data

    Returns:
        YOLO: aggregated YOLO model
    """
    net = YOLO(model_path)
    state_dict = net.state_dict().copy()
    state_dict.update(aggregated_state.to_torch_state_dict())
    net.load_state_dict(state_dict=state_dict)
    return net
