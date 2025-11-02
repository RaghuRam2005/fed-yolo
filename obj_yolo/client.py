""" Client App for federated learning """
import os
from pathlib import Path

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from obj_yolo.utils import train as train_fn
from obj_yolo.utils import test as test_fn

from ultralytics import YOLO
from ultralytics.utils.torch_utils import unwrap_model

client_app = ClientApp()

@client_app.train()
def train(msg:Message, context:Context):
    """
    Train the model using the local data for each client

    Args:
        msg (Message)
        context (Context)
    """

    # search for data and YOLO config files
    BASE_LIB_PATH = os.path.abspath(os.path.dirname(__file__))
    BASE_DIR_PATH = os.path.dirname(BASE_LIB_PATH)
    YOLO_CONFIG = Path(BASE_DIR_PATH) / "yolo_config" / "yolo11n.yaml"

    # load new model instance everytime we run a client train
    model = YOLO(YOLO_CONFIG)
    model.load_state_dict(msg.content['arrays'].to_torch_state_dict())

    # load configuration
    partition_id = context.node_config["partition-id"]
    data_count = context.run_config["train-data-count"]
    data_path = Path(BASE_DIR_PATH) / "dataset" / "clients" / f"client_{partition_id}" / "data.yaml"
    if not data_path.exists():
        raise Exception(f"Data Not prepared Exception: Client-{partition_id}")

    # train the model
    train_metrics = train_fn(
        partition_id=partition_id,
        model=model,
        data_path=data_path,
        local_epochs=context.run_config['local-epochs'],
        lr0=msg.content["config"]["lr"],
    )

    # construct the state dict of the model
    unwrapped_model = unwrap_model(model)
    state_dict = unwrapped_model.state_dict()
    detached_weights = {
        k: v.detach()
        for k, v in state_dict.items()
        if isinstance(v, torch.Tensor)
    }

    # construct record and store them
    model_record = ArrayRecord(detached_weights)
    metrics = {
        'train-map':train_metrics,
        'num-examples':data_count,
    }
    metrics_record = MetricRecord(metrics)
    content = RecordDict({"arrays" : model_record, "metrics" : metrics_record})
    return Message(content=content, reply_to=msg)

@client_app.evaluate()
def evaluate(msg:Message, context:Context):
    # search for data and YOLO config files
    BASE_LIB_PATH = os.path.abspath(os.path.dirname(__file__))
    BASE_DIR_PATH = os.path.dirname(BASE_LIB_PATH)
    YOLO_CONFIG = Path(BASE_DIR_PATH) / "yolo_config" / "yolo11n.yaml"

    # load new model instance everytime we run a client train
    model = YOLO(YOLO_CONFIG)
    model.load_state_dict(msg.content['arrays'].to_torch_state_dict())

    # load the data
    partition_id = context.node_config["partition-id"]
    data_count = context.run_config["val-data-count"]
    data_path = Path(BASE_DIR_PATH) / "dataset" / "clients" / f"client_{partition_id}" / "data.yaml"
    if not data_path.exists():
        raise Exception(f"Data Not prepared Exception: Client-{partition_id}")
    
    eval_metrics = test_fn(
        partition_id=partition_id,
        model=model,
        data_path=data_path,
    )

    # construct validation metric records
    metrics = {
        "eval-map" : eval_metrics,
        "num-examples" : data_count,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics":metric_record})
    return Message(content=content, reply_to=msg)
