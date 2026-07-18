""" Client App for federated learning """
import os
import logging
from pathlib import Path

import torch

from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from obj_yolo.utils import train as train_fn
from obj_yolo.utils import test as test_fn
from obj_yolo.utils import eval_train as train_val_fn
from obj_yolo.dataset import PrepareData

from ultralytics import YOLO
from ultralytics.utils.torch_utils import unwrap_model

# logging while running the client
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

client_app = ClientApp()

# BatchNorm buffer keys kept personalized per client (FedTag), never aggregated
_BN_BUFFER_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked")


def _personal_bn_path(base_path: str, dataset_name: str, partition_id: int) -> Path:
    return Path(base_path) / "flwr_simulation" / f"{dataset_name}" / f"client_{partition_id}" / "personal_bn.pt"


def _apply_personal_bn(model: YOLO, bn_path: Path) -> None:
    """Overlay this client's own BatchNorm running stats onto the shared model (FedTag)."""
    if not bn_path.exists():
        return
    personal_bn = torch.load(bn_path)
    state_dict = model.model.state_dict()
    state_dict.update(personal_bn)
    model.model.load_state_dict(state_dict)


def _save_personal_bn(model: YOLO, bn_path: Path) -> None:
    """Persist this client's trained BatchNorm running stats (FedTag)."""
    unwrapped_model = unwrap_model(model)
    state_dict = unwrapped_model.state_dict()
    bn_state = {k: v for k, v in state_dict.items() if k.endswith(_BN_BUFFER_SUFFIXES)}
    bn_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bn_state, bn_path)


def _own_weather_tag(base_path: str, partition_id: int) -> str:
    """Look up this client's persistent weather tag from client_tags.json (FedTag)."""
    clients_path = Path(base_path) / "dataset" / "clients"
    try:
        client_tags = PrepareData.load_client_tags(clients_path)
    except FileNotFoundError:
        return "clear"
    return client_tags.get(str(partition_id), "clear")


@client_app.train()
def train(msg:Message, context:Context):
    """
    Train the model using the local data for each client

    Args:
        msg (Message)
        context (Context)
    """
    dataset_name = context.run_config["dataset-name"]
    BASE_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    
    # search the YOLO model file
    model_path = Path(BASE_PATH ) / "flwr_simulation" / f"{dataset_name}" / "aggregated_model" / "agg_model.pt"
    if not model_path.exists():
        logging.error(f"Aggregated model not found at {model_path}")
        raise FileNotFoundError(f"model.pt not found at {model_path}")
        
    # load the model
    model = YOLO(model_path)

    # load configuration
    partition_id = context.node_config["partition-id"]
    epochs = context.run_config["local-epochs"]
    lr0 = context.run_config["lr0"]
    
    # count the images in the data folder
    img_data_path = Path(BASE_PATH) / "dataset" / "clients" / f"client_{partition_id}" / "images" / "train"
    if not img_data_path.exists():
        logging.error(f"Image file not found at {img_data_path}")
        raise FileNotFoundError(f"Images not found at {img_data_path}")
    data_count = len(list(img_data_path.glob("*")))
    
    # load data.yaml file
    data_path = Path(BASE_PATH) / "dataset" / "clients" / f"client_{partition_id}" / "data.yaml"
    if not data_path.exists():
        logging.error(f"DATA.yaml does not exist at {data_path}")
        raise FileNotFoundError(f"data.yaml not found at {data_path}")

    # FedTag sends a per-node "tags" record (current l1-lambda for this node's
    # tag); other strategies don't, so this whole block is a no-op for them.
    # The client determines its OWN tag from client_tags.json (keyed by its
    # partition-id) rather than the server assigning it, since flwr's node_id
    # doesn't match the partition-id used when the dataset was prepared.
    tag_record = msg.content["tags"] if "tags" in msg.content else None
    l1_lambda = float(tag_record["l1-lambda"]) if tag_record is not None else 0.0
    my_tag = _own_weather_tag(BASE_PATH, partition_id) if tag_record is not None else None
    bn_path = _personal_bn_path(BASE_PATH, dataset_name, partition_id)
    if tag_record is not None:
        _apply_personal_bn(model, bn_path)

    # train the model
    train_metrics = train_fn(
        partition_id=int(partition_id),
        model=model,
        data_path=data_path,
        local_epochs=int(epochs),
        lr0 = float(lr0),
        l1_lambda=l1_lambda,
    )

    if tag_record is not None:
        _save_personal_bn(model, bn_path)

    # construct the state dict of the model
    unwrapped_model = unwrap_model(model)
    state_dict = unwrapped_model.state_dict()

    # save the trained model
    model_path = Path(BASE_PATH) / "flwr_simulation" / f"{dataset_name}" / f"client_{partition_id}" / "model.pt"
    model.save(model_path)

    # construct record and store them
    model_record = ArrayRecord(state_dict)
    metrics = {
        'train-map':train_metrics,
        'num-examples':data_count,
    }
    metrics_record = MetricRecord(metrics)
    content = RecordDict({"arrays" : model_record, "metrics" : metrics_record})
    if tag_record is not None:
        content["tags"] = ConfigRecord({"tag": my_tag, "l1-lambda": l1_lambda})
    return Message(content=content, reply_to=msg)

@client_app.evaluate()
def evaluate(msg:Message, context:Context):
    """
    Evalute the aggregated metrics using the Federated clients

    Args:
        msg (Message): message object from algorithm
        context (Context): context object from server
    """
    dataset_name = context.run_config["dataset-name"]
    BASE_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    
    # search the YOLO model file
    model_path = Path(BASE_PATH ) / "flwr_simulation" / f"{dataset_name}" / "aggregated_model" / "agg_model.pt"
    if not model_path.exists():
        logging.error(f"Aggregated model not found at {model_path}")
        raise FileNotFoundError(f"model.pt not found at {model_path}")
        
    # load the model
    model = YOLO(model_path)

    # load configuration
    partition_id = context.node_config["partition-id"]
    
    # count the images in the data folder
    img_data_path = Path(BASE_PATH) / "dataset" / "clients" / f"client_{partition_id}" / "images" / "train"
    if not img_data_path.exists():
        logging.error(f"Image file not found at {img_data_path}")
        raise FileNotFoundError(f"Images not found at {img_data_path}")
    data_count = len(list(img_data_path.glob("*")))
    
    # load data.yaml file
    data_path = Path(BASE_PATH) / "dataset" / "clients" / f"client_{partition_id}" / "data.yaml"
    if not data_path.exists():
        logging.error(f"DATA.yaml does not exist at {data_path}")
        raise FileNotFoundError(f"data.yaml not found at {data_path}")

    # FedTag sends a per-node "tags" record; other strategies don't.
    tag_record = msg.content["tags"] if "tags" in msg.content else None
    my_tag = _own_weather_tag(BASE_PATH, partition_id) if tag_record is not None else None
    if tag_record is not None:
        _apply_personal_bn(model, _personal_bn_path(BASE_PATH, dataset_name, partition_id))

    # we are training model for warming up after loading the aggregation state
    eval_train = train_val_fn(
        partition_id=int(partition_id),
        model=model,
        data_path=data_path,
        local_epochs=3,
        lr0=0.001,
    )

    logging.info(f"Evaluation Training completed, now starting the evaluation, results: {eval_train}")

    eval_metrics = test_fn(
        partition_id=int(partition_id),
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
    if tag_record is not None:
        content["tags"] = ConfigRecord({"tag": my_tag, "l1-lambda": float(tag_record["l1-lambda"])})
    return Message(content=content, reply_to=msg)
