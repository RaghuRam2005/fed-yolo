""" Client App for federated learning """
import os
import logging
from pathlib import Path

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from obj_yolo.utils import train as train_fn
from obj_yolo.utils import test as test_fn
from obj_yolo.utils import eval_train as train_val_fn

from ultralytics import YOLO
from ultralytics.utils.torch_utils import unwrap_model

# logging while running the client
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

client_app = ClientApp()

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

    # train the model
    train_metrics = train_fn(
        partition_id=int(partition_id),
        model=model,
        data_path=data_path,
        local_epochs=int(epochs),
        lr0 = float(lr0)
    )

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
    return Message(content=content, reply_to=msg)
