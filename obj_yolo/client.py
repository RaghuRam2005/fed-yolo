""" Client App for federated learning """
import os
from pathlib import Path

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from obj_yolo.utils import train as train_fn
from obj_yolo.utils import test as test_fn
from obj_yolo.utils import eval_train as train_val_fn

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
    new_arrays = msg.content['arrays'].to_torch_state_dict()
    untrainable_parameters = msg.content['untrain_arrays'].to_torch_state_dict()
    if not untrainable_parameters:
        state_dict = model.model.state_dict().copy()
        state_dict.update(new_arrays)
    else:
        state_dict = model.model.state_dict().copy()
        state_dict.update(new_arrays)
        state_dict.update(untrainable_parameters)
    model.model.load_state_dict(state_dict, strict=True)

    # load configuration
    partition_id = context.node_config["partition-id"]
    data_path = Path(BASE_DIR_PATH) / "dataset" / "clients" / f"client_{partition_id}" / "data.yaml"
    if not data_path.exists():
        raise Exception(f"Data Not prepared Exception: Client-{partition_id}")
    data_count = len(os.listdir(Path(BASE_DIR_PATH) / "dataset" / "clients" / f"client_{partition_id}" / "images" / "train"))

    # train the model
    train_metrics = train_fn(
        partition_id=partition_id,
        model=model,
        data_path=data_path,
        local_epochs=context.run_config['local-epochs'],
        lr0=msg.content["config"]["lr"],
        mu=msg.content["config"]["mu"],
    )

    # construct the state dict of the model
    unwrapped_model = unwrap_model(model)
    state_dict = unwrapped_model.model.state_dict()
    trainable_parameters = {}
    untrainable_parameters = {}
    for k, val in state_dict.items():
        if k.endswith(('running_mean', 'running_var', 'num_batches_tracked')):
            untrainable_parameters[k] = val
        if isinstance(val, torch.Tensor):
            trainable_parameters[k] = val
    
    # construct record and store them
    model_record = ArrayRecord(trainable_parameters)
    parameter_record = ArrayRecord(untrainable_parameters)
    metrics = {
        'train-map':train_metrics,
        'num-examples':data_count,
    }
    metrics_record = MetricRecord(metrics)
    content = RecordDict({"arrays" : model_record, "metrics" : metrics_record, "untrain_arrays" : parameter_record})
    return Message(content=content, reply_to=msg)

@client_app.evaluate()
def evaluate(msg:Message, context:Context):
    # search for data and YOLO config files
    BASE_LIB_PATH = os.path.abspath(os.path.dirname(__file__))
    BASE_DIR_PATH = os.path.dirname(BASE_LIB_PATH)
    YOLO_CONFIG = Path(BASE_DIR_PATH) / "yolo_config" / "yolo11n.yaml"

    # load a new model instance
    model = YOLO(YOLO_CONFIG)
    new_arrays = msg.content['arrays'].to_torch_state_dict()
    untrainable_parameters = msg.content['untrain_arrays'].to_torch_state_dict()
    if not untrainable_parameters:
        state_dict = model.model.state_dict().copy()
        state_dict.update(new_arrays)
    else:
        state_dict = model.model.state_dict().copy()
        state_dict.update(new_arrays)
        state_dict.update(untrainable_parameters)
    model.model.load_state_dict(state_dict, strict=True)

    # load the data
    partition_id = context.node_config["partition-id"]
    data_path = Path(BASE_DIR_PATH) / "dataset" / "clients" / f"client_{partition_id}" / "data.yaml"
    if not data_path.exists():
        raise Exception(f"Data Not prepared Exception: Client-{partition_id}")
    data_count = len(os.listdir(Path(BASE_DIR_PATH) / "dataset" / "clients" / f"client_{partition_id}" / "images" / "train"))
    
    # we are training model for warming up after loading the aggregation state
    #eval_train = train_val_fn(
    #    partition_id=partition_id,
    #    model=model,
    #    data_path=data_path,
    #    local_epochs=2,
    #    lr0=0.01,
    #)
    
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
